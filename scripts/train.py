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
    from torch.utils.data import DataLoader

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    torch.manual_seed(cfg["seed"])

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

    # ---- Dataset metadata ----
    repo_id = dcfg.get("repo_id") or "local/eval1_merged"
    root = dcfg.get("root")
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
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=dcfg.get("episodes"),  # null = all
        delta_timestamps=delta_timestamps,
    )
    print(f"[train] dataset frames: {len(dataset)}  episodes: {ds_meta.total_episodes}")

    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=True,
        persistent_workers=tcfg.get("num_workers", 4) > 0,
    )

    # ---- Policy + processors ----
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.train()
    print(f"[train] params: {sum(p.numel() for p in policy.parameters()):,}")

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

    # ---- Loop ----
    step = 0
    last_loss = float("inf")
    loss_window: list[float] = []
    t0 = time.time()
    optim.zero_grad()

    while step < total_steps:
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
                eta_min = (total_steps - step) / max(steps_per_s, 1e-6) / 60
                print(
                    f"[train] step={step:6d}/{total_steps}  "
                    f"loss={last_loss:.4f}  avg50={avg:.4f}  "
                    f"lr={lr:.2e}  {steps_per_s:.2f} steps/s  ETA={eta_min:.1f}min"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": last_loss,
                            "train/loss_avg50": avg,
                            "train/lr": lr,
                            "train/steps_per_s": steps_per_s,
                        },
                        step=step,
                    )
            if step > 0 and step % tcfg["save_every"] == 0:
                ckpt_dir = out_dir / f"step_{step}"
                policy.save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)

            step += 1
            if step >= total_steps:
                break

    # ---- Final save ----
    final_dir = out_dir / "final"
    try:
        policy.save_pretrained(final_dir)
        preprocessor.save_pretrained(final_dir)
        postprocessor.save_pretrained(final_dir)
        print(f"[train] saved final checkpoint -> {final_dir}")

        if use_wandb:
            avg = sum(loss_window) / max(1, len(loss_window))
            wandb.summary["final_loss"] = last_loss
            wandb.summary["final_loss_avg50"] = avg
    finally:
        if use_wandb and wandb_run is not None:
            wandb.finish()

    # ---- HF Hub push ----
    hf_cfg = cfg.get("hf") or {}
    if hf_cfg.get("push"):
        repo_id_hf = expand_env(hf_cfg["repo_id"])
        private = bool(hf_cfg.get("private", True))
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id_hf, private=private, repo_type="model", exist_ok=True)
            commit_msg = f"upload {cfg['experiment_name']} {run_id}"
            print(f"[train] uploading to https://huggingface.co/{repo_id_hf} (private={private})")
            api.upload_folder(
                folder_path=str(final_dir),
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
            print(f"  hf upload {repo_id_hf} {final_dir} . --repo-type=model")
            sys.exit(2)
    else:
        print("[train] HF push disabled (cfg.hf.push is false).")

    print("[train] done.")


if __name__ == "__main__":
    main()
