"""Full SmolVLA fine-tuning entry point (Eval 1 and similarly-shaped tasks).

Differences from `overfit_test.py`:
  * No `pass_criteria` gate — this is real training, not a sanity check.
  * Uses all episodes by default (`episodes: null`).
  * Pushes the final checkpoint to HuggingFace Hub if `cfg.hf.push` is set.
  * Image augmentations from `cfg.data.augmentations` are NOT applied yet;
    a warning is logged so the user knows. SmolVLA's preprocessor pipeline
    doesn't expose a hook for image transforms — adding them requires a
    custom IterableDataset wrapper. Track as a TODO.

Usage:
    python scripts/train.py --config configs/train/full_eval1.yaml

Or via slurm:
    sbatch scripts/train.slurm configs/train/full_eval1.yaml scripts/train.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_env(s):
    """Expand $VAR / ${VAR} in a string. Used for HF repo IDs in YAML."""
    if isinstance(s, str):
        return os.path.expandvars(s)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[train] config: {args.config}")
    print(f"[train] experiment: {cfg['experiment_name']}")

    # Run-id'd output dir, same convention as overfit_test.py.
    run_id = time.strftime("%Y%m%d-%H%M%S")
    if "SLURM_JOB_ID" in os.environ:
        run_id = f"{run_id}_job{os.environ['SLURM_JOB_ID']}"
    base_out = Path(cfg["train"]["output_dir"])
    cfg["train"]["output_dir"] = str(base_out / run_id)
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] run_id: {run_id}")
    print(f"[train] output_dir: {out_dir}")

    # ---- wandb ----
    log_cfg = cfg.get("logging") or {}
    use_wandb = bool(log_cfg.get("use_wandb", False)) and log_cfg.get("mode", "online") != "disabled"
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=log_cfg.get("project", "Lerobot"),
            entity=log_cfg.get("entity"),
            name=log_cfg.get("name") or f"{cfg['experiment_name']}-{run_id}",
            id=run_id,
            group=cfg["experiment_name"],
            tags=log_cfg.get("tags"),
            mode=log_cfg.get("mode", "online"),
            dir=str(out_dir),
            config=cfg,
        )
        print(f"[train] wandb: {wandb_run.url if wandb_run.url else '(offline)'}")

    # Lazy imports so argparse stays fast.
    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    torch.manual_seed(cfg["seed"])
    # TF32 matmul: ~3-5% free on A100 fp32 paths (bf16 already main path).
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    dcfg = cfg["data"]
    tcfg = cfg["train"]
    mcfg = cfg["model"]

    # ---- Augmentation TODO ----
    if dcfg.get("augmentations"):
        print(
            f"[train] WARNING: augmentations {dcfg['augmentations']} are configured "
            "but NOT applied — image-aug hook is not yet wired into SmolVLA's "
            "preprocessor. Train still runs without them."
        )

    # `data.sources` is an optional list of per-source dicts (each accepts the
    # same keys — repo_id, root, episodes, val — that the top-level dcfg does).
    # When present, datasets are concatenated for training; the FIRST source's
    # LeRobotDatasetMetadata is used for the normalizer stats. Action/state
    # shapes must match across sources (verified at runtime by lerobot).
    sources = dcfg.get("sources")
    primary = sources[0] if sources else dcfg
    repo_id = primary.get("repo_id") or "local/eval1_merged"
    root = primary.get("root")
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)

    # ---- Policy config ----
    policy_cfg = SmolVLAConfig(
        pretrained_path=mcfg["base"],
        chunk_size=mcfg["chunk_size"],
        n_action_steps=mcfg["chunk_size"],
        device=tcfg["device"],
        freeze_vision_encoder=mcfg["freeze_vision_encoder"],
    )

    # ---- Dataset ----
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    seed = int(cfg.get("seed", 42))

    def _build_one(src: dict) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset | None, list[int] | None, str, str | None]:
        """Build a (train, val_or_None) pair for one source. The val split is opt-in
        per source via `val` (color-stratified by tasks.parquet). Skipped when the
        source pins `episodes` (single-episode debug or carry-over subsets).

        Optional `prompt_augment` field on a source enables per-sample prompt
        rewriting (direct/ordinal/relational/negation) using the bowl arrangement
        for that episode. Val datasets are NOT augmented — eval loss should track
        the canonical prompt distribution to be comparable across runs."""
        s_repo = src.get("repo_id") or "local/eval1_merged"
        s_root = src.get("root")
        s_val_cfg = (src.get("val") or {}) if src.get("episodes") is None else {}
        train_eps = src.get("episodes")
        val_eps = None
        if s_val_cfg:
            from src.data.splits import episodes_by_color, train_val_episode_split
            by = episodes_by_color(s_repo, s_root)
            train_eps, val_eps = train_val_episode_split(
                by,
                per_color=s_val_cfg.get("per_color"),
                fraction=s_val_cfg.get("fraction"),
                min_train_per_color=int(s_val_cfg.get("min_train_per_color", 3)),
                seed=seed,
            )
            print(f"[train] {s_repo}: train_eps={len(train_eps)} val_eps={len(val_eps)} -> {val_eps}")
        # `force_cache_sync=True` works around a lerobot v0.5.1 bug: when a
        # second LeRobotDataset is built against the same repo with a
        # different `episodes` slice (here: train_ds first, then val_ds),
        # `try_load()` finds the train parquets already on disk, applies the
        # val episode-index filter, gets zero rows, and raises
        # `ValueError("Instruction 'train' corresponds to no data!")` — which
        # `try_load`'s except clause does NOT catch (only FileNotFoundError /
        # NotADirectoryError). Forcing a cache sync skips that broken path
        # and goes straight to selective hub fetch, which is idempotent.
        train_ds: torch.utils.data.Dataset = LeRobotDataset(
            repo_id=s_repo, root=s_root, episodes=train_eps,
            delta_timestamps=delta_timestamps, force_cache_sync=True,
        )
        val_ds: torch.utils.data.Dataset | None = None
        if val_eps:
            val_ds = LeRobotDataset(
                repo_id=s_repo, root=s_root, episodes=val_eps,
                delta_timestamps=delta_timestamps, force_cache_sync=True,
            )

        aug = src.get("prompt_augment")
        if aug:
            from src.data.prompt_aug import PromptAugmentingDataset, load_arrangements
            arr_path = aug.get("arrangements")
            if not arr_path:
                raise ValueError(f"{s_repo}: prompt_augment requires `arrangements` path")
            arrs = load_arrangements(arr_path, s_repo)
            if not arrs:
                raise ValueError(f"{s_repo}: no arrangement entries found in {arr_path}")
            train_ds = PromptAugmentingDataset(train_ds, arrs, seed=seed)
            print(f"[train] {s_repo}: prompt_augment ON ({len(arrs)} episodes mapped)")
        return train_ds, val_ds, train_eps, s_repo, s_root

    # Per-source train_eps + (repo_id, root) saved so we can build phase labels
    # against the SAME slice we just handed to the LeRobotDataset.
    source_specs: list[tuple[list[int] | None, str, str | None]] = []
    if sources:
        train_parts: list[LeRobotDataset] = []
        val_parts: list[LeRobotDataset] = []
        for src in sources:
            td, vd, t_eps, t_repo, t_root = _build_one(src)
            train_parts.append(td)
            source_specs.append((t_eps, t_repo, t_root))
            if vd is not None:
                val_parts.append(vd)
        dataset = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
        val_dataset = (
            None if not val_parts else (val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts))
        )
        print(
            f"[train] multi-source: {len(sources)} sources  "
            f"train_frames={len(dataset)}  val_frames={len(val_dataset) if val_dataset else 0}"
        )
    else:
        dataset, val_dataset, t_eps, t_repo, t_root = _build_one(dcfg)
        source_specs.append((t_eps, t_repo, t_root))
        print(f"[train] dataset frames: {len(dataset)}  episodes: {ds_meta.total_episodes}")

    # Optional phase-weighted sampler. Off by default — see plan
    # flower-vla-smol-vla-flickering-puddle.md. Handles single-source and
    # ConcatDataset (multi-source) uniformly via per-source label compute.
    phase_cfg = tcfg.get("phase_sampling") or {}
    sampler = None
    if phase_cfg.get("enabled"):
        from src.data.phase_labels import compute_phase_labels, summarize
        from src.data.sampler import (
            make_phase_weighted_sampler, concat_phase_labels,
            assert_dataset_alignment, assert_concat_alignment,
        )
        kwargs = dict(
            open_frac=float(phase_cfg.get("open_frac", 0.6)),
            close_frac=float(phase_cfg.get("close_frac", 0.4)),
            min_amplitude=float(phase_cfg.get("min_amplitude", 5.0)),
            post_close_margin=int(phase_cfg.get("post_close_margin", 3)),
        )
        parts = []
        for (eps, p_repo, p_root) in source_specs:
            # SmolVLA path: LeRobotDataset's HF cache is in standard location;
            # if root is None, resolve via huggingface_hub snapshot path.
            label_root = Path(p_root) if p_root else None
            if label_root is None:
                from huggingface_hub import snapshot_download
                label_root = Path(snapshot_download(
                    repo_id=p_repo, repo_type="dataset", revision="v3.0",
                    allow_patterns=["meta/*", "data/**"],
                ))
            part = compute_phase_labels(
                repo_id=p_repo, root=label_root, episodes=eps, **kwargs,
            )
            print(f"[train] phase_sampling: {summarize(part, label=p_repo)}")
            parts.append(part)
        if isinstance(dataset, ConcatDataset):
            phase_labels = concat_phase_labels(parts)
            assert_concat_alignment(dataset, parts, n_check=16)
        else:
            phase_labels = parts[0]
            assert_dataset_alignment(dataset, phase_labels, n_check=16)
        if len(phase_labels) != len(dataset):
            raise RuntimeError(
                f"phase_sampling: labels length {len(phase_labels)} != dataset length "
                f"{len(dataset)} — iteration order divergence."
            )
        sampler = make_phase_weighted_sampler(
            phase_labels,
            weight_pregrasp=float(phase_cfg.get("weight_pregrasp", 2.0)),
            replacement=bool(phase_cfg.get("replacement", True)),
            seed=int(cfg.get("seed", 42)),
        )

    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=tcfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=True,
        persistent_workers=tcfg.get("num_workers", 4) > 0,
    )

    val_loader = None
    if val_dataset is not None:
        # Cap val workers — a few are plenty for a small set, leaves CPUs for train.
        val_workers = min(2, int(tcfg.get("num_workers", 4)))
        val_loader = DataLoader(
            val_dataset,
            batch_size=tcfg["batch_size"],
            shuffle=False,
            num_workers=val_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=val_workers > 0,
        )
        print(f"[train] val frames: {len(val_dataset)}  batches: {len(val_loader)}")

    # ---- Policy + processors ----
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.train()
    print(f"[train] params: {sum(p.numel() for p in policy.parameters()):,}")

    # Optional torch.compile. Bench (job 2701026) shows +9% samples/s and
    # -20% VRAM at the cost of ~5 min cold-start; SmolVLA's flow-matching
    # forward has graph breaks (Beta sample, .item() in loss path) so we use
    # default mode rather than reduce-overhead — same gain, no graph drama.
    if tcfg.get("compile", False):
        print("[train] torch.compile enabled (mode=default) — first step will be slow")
        policy = torch.compile(policy)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=mcfg["base"],
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": tcfg["device"]},
            "normalizer_processor": {
                "stats": ds_meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": ds_meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )

    # ---- Optim + LR schedule ----
    optim = torch.optim.AdamW(
        policy.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        fused=torch.cuda.is_available(),
    )
    warmup = int(tcfg.get("warmup_steps", 0))
    total_steps = int(tcfg["num_steps"])

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        # cosine to 10% of base after warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        import math
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    grad_accum = int(tcfg.get("grad_accum_steps", 1))

    # ---- GPU sampler (for per-step system/* metrics in wandb) ----
    from src.utils.gpu_metrics import GpuSampler
    gpu_sampler = GpuSampler()  # gracefully no-ops on non-CUDA / no-driver hosts

    # ---- Sidecar metadata helper (for rollout-side wandb linkage) ----
    from src.utils.checkpoint_meta import write_checkpoint_meta
    from src.utils.run_metadata import git_sha as _git_sha_helper
    _git_sha_str = _git_sha_helper()

    def run_eval() -> float | None:
        """Full no-grad pass over val set; returns mean loss or None if no val set."""
        if val_loader is None:
            return None
        policy.eval()
        losses: list[float] = []
        with torch.no_grad():
            for vbatch in val_loader:
                vbatch = preprocessor(vbatch)
                vloss, _ = policy.forward(vbatch)
                losses.append(float(vloss.detach().cpu()))
        policy.train()
        return sum(losses) / max(1, len(losses))

    # ---- Eval cadence (decoupled from save_every so we can poll val finely
    # without paying the disk cost of checkpointing every time). Defaults to
    # save_every for backwards compat with older configs. ----
    eval_every = int(tcfg.get("eval_every", tcfg["save_every"]))

    # ---- Early-stop config (opt-in via cfg.train.early_stop) ----
    # Conservative defaults: only fires after `min_steps` AND only when val
    # has plateaued or risen for `patience` consecutive eval cycles. The
    # best-so-far checkpoint is always saved separately to `<out>/best/`
    # alongside `step_N/` periodic saves so we never lose the lowest-val
    # checkpoint to overfit drift.
    es_cfg = tcfg.get("early_stop") or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 4))
    es_min_delta = float(es_cfg.get("min_delta", 0.005))
    es_min_steps = int(es_cfg.get("min_steps", 5000))
    if es_enabled:
        print(f"[train] early-stop: enabled  patience={es_patience}  "
              f"min_delta={es_min_delta}  min_steps={es_min_steps}  "
              f"eval_every={eval_every}")
    best_val_loss = float("inf")
    best_step = -1
    no_improve_evals = 0

    # ---- Loop ----
    step = 0
    last_loss = float("inf")
    loss_window: list[float] = []
    t0 = time.time()
    optim.zero_grad()
    should_stop = False

    while step < total_steps and not should_stop:
        for batch in loader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                optim.step()
                sched.step()
                optim.zero_grad()

            last_loss = float(loss.detach().cpu())
            loss_window.append(last_loss)
            if len(loss_window) > 50:
                loss_window.pop(0)

            if step % tcfg["log_every"] == 0:
                avg = sum(loss_window) / len(loss_window)
                lr = optim.param_groups[0]["lr"]
                elapsed = time.time() - t0
                steps_per_s = (step + 1) / max(elapsed, 1e-6)
                samples_per_s = steps_per_s * tcfg["batch_size"]
                eta_min = (total_steps - step) / max(steps_per_s, 1e-6) / 60
                print(
                    f"[train] step={step:6d}/{total_steps}  "
                    f"loss={last_loss:.4f}  avg50={avg:.4f}  "
                    f"lr={lr:.2e}  {steps_per_s:.2f} steps/s  ETA={eta_min:.1f}min"
                )
                if use_wandb:
                    payload = {
                        "train/loss": last_loss,
                        "train/loss_avg50": avg,
                        "train/lr": lr,
                        "train/steps_per_s": steps_per_s,
                        "train/samples_per_s": samples_per_s,
                    }
                    payload.update(gpu_sampler.sample())
                    wandb.log(payload, step=step)
            if step > 0 and step % tcfg["save_every"] == 0:
                ckpt_dir = out_dir / f"step_{step}"
                policy.save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)
                write_checkpoint_meta(
                    ckpt_dir, wandb_run, cfg, step, _git_sha_str,
                )

            if step > 0 and step % eval_every == 0:
                val_loss = run_eval()
                if val_loss is not None:
                    print(f"[train] step={step:6d}  eval/loss={val_loss:.4f}")
                    if use_wandb:
                        wandb.log({"eval/loss": val_loss}, step=step)
                    # "improved" controls whether to refresh the best/ ckpt;
                    # "significant" (improved by ≥ min_delta) controls whether
                    # to reset the no-improve counter. Decoupling these means a
                    # slow trickle of sub-min_delta improvements still saves the
                    # lowest-val ckpt while patience continues to count down.
                    improved = val_loss < best_val_loss
                    significant = val_loss < best_val_loss - es_min_delta
                    if improved:
                        best_val_loss = val_loss
                        best_step = step
                        best_dir = out_dir / "best"
                        policy.save_pretrained(best_dir)
                        preprocessor.save_pretrained(best_dir)
                        postprocessor.save_pretrained(best_dir)
                        # Stamp metadata so downstream knows what's inside `best/`.
                        # Kept for backwards-compat; superseded by wandb_metadata.json.
                        (best_dir / "best_val_meta.json").write_text(
                            json.dumps({"val_loss": float(val_loss),
                                        "step": int(step),
                                        "experiment": cfg["experiment_name"]}, indent=2)
                        )
                        write_checkpoint_meta(
                            best_dir, wandb_run, cfg, step, _git_sha_str,
                            extra={"val_loss": float(val_loss), "is_best": True},
                        )
                        print(f"[train] new best  val_loss={val_loss:.4f} @ step {step} -> {best_dir}")
                        if use_wandb:
                            wandb.log({"eval/best_loss": val_loss,
                                       "eval/best_step": step}, step=step)
                    if significant:
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                        print(f"[train] no-improve evals: {no_improve_evals}/"
                              f"{es_patience} (best={best_val_loss:.4f} @ {best_step})")
                    if (es_enabled and no_improve_evals >= es_patience
                            and step >= es_min_steps):
                        print(f"[train] EARLY STOP at step {step}: "
                              f"{no_improve_evals} consecutive evals without "
                              f"≥{es_min_delta} improvement. "
                              f"Best val_loss={best_val_loss:.4f} @ step {best_step}.")
                        if use_wandb:
                            wandb.summary["early_stop_step"] = step
                            wandb.summary["early_stop_best_step"] = best_step
                            wandb.summary["early_stop_best_loss"] = best_val_loss
                        should_stop = True

            step += 1
            if step >= total_steps or should_stop:
                break

    # ---- Final save ----
    final_dir = out_dir / "final"
    try:
        policy.save_pretrained(final_dir)
        preprocessor.save_pretrained(final_dir)
        postprocessor.save_pretrained(final_dir)
        write_checkpoint_meta(
            final_dir, wandb_run, cfg, max(0, step - 1), _git_sha_str,
            extra={"is_final": True},
        )
        print(f"[train] saved final checkpoint -> {final_dir}")

        final_val = run_eval()
        if final_val is not None:
            print(f"[train] final eval/loss={final_val:.4f}")
            if use_wandb:
                wandb.log({"eval/loss": final_val}, step=max(0, step - 1))
                wandb.summary["final_eval_loss"] = final_val

        if use_wandb:
            avg = sum(loss_window) / max(1, len(loss_window))
            wandb.summary["final_loss"] = last_loss
            wandb.summary["final_loss_avg50"] = avg
    finally:
        gpu_sampler.shutdown()
        if use_wandb and wandb_run is not None:
            wandb.finish()

    # ---- HF Hub push ----
    # Default target is best/ (lowest val_loss seen during training). final/
    # is the post-loop snapshot, which on long runs is overfit; we'd prefer
    # not to publish that. Fall back to final/ if best/ is missing — happens
    # for runs without a val split (overfit.yaml/bench.yaml) where best/ is
    # never written. The user can force this with `hf.upload: final` if they
    # really want the last-step ckpt instead.
    hf_cfg = cfg.get("hf") or {}
    if hf_cfg.get("push"):
        repo_id_hf = expand_env(hf_cfg["repo_id"])
        private = bool(hf_cfg.get("private", True))
        upload_pref = str(hf_cfg.get("upload", "best")).lower()
        best_dir = out_dir / "best"
        if upload_pref == "final":
            push_dir, push_kind = final_dir, "final"
        elif best_dir.exists():
            push_dir, push_kind = best_dir, "best"
        else:
            print("[train] note: no best/ ckpt (likely no val split); pushing final/ instead.")
            push_dir, push_kind = final_dir, "final"
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id_hf, private=private, repo_type="model", exist_ok=True)
            commit_msg = f"upload {cfg['experiment_name']} {run_id} ({push_kind})"
            print(f"[train] uploading {push_kind}/ to https://huggingface.co/{repo_id_hf} (private={private})")
            api.upload_folder(
                folder_path=str(push_dir),
                repo_id=repo_id_hf,
                repo_type="model",
                commit_message=commit_msg,
            )
            print(f"[train] HF upload complete: https://huggingface.co/{repo_id_hf}")
        except Exception as e:
            print(f"[train] HF upload FAILED: {e!r}")
            print("[train] checkpoint is still saved locally — re-upload manually with:")
            print("  conda activate lerobot")
            print(f"  hf auth login   # PrajnaYang token, WRITE scope")
            print(f"  hf repos create {repo_id_hf} --type model")
            print(f"  hf upload {repo_id_hf} {push_dir} . --repo-type=model")
            sys.exit(2)
    else:
        print("[train] HF push disabled (cfg.hf.push is false).")

    print("[train] done.")


if __name__ == "__main__":
    main()
