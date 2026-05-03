"""Overfit a single episode to verify the training pipeline works end-to-end.

The most important script in week 1. If this doesn't reach near-zero loss,
something is wrong with: data loading, action normalization, model loading,
device placement, or loss computation. FIX IT BEFORE SCALING UP.

Usage:
    python scripts/overfit_test.py --config configs/train/overfit.yaml

Implementation notes:
- We use lerobot's official factory functions (`make_policy`,
  `make_pre_post_processors`, `resolve_delta_timestamps`) instead of
  hand-rolling the data + model setup. SmolVLA expects tokenized prompts
  (`OBS_LANGUAGE_TOKENS`) and normalized state/action that the raw
  `LeRobotDataset` does NOT produce; the preprocessor pipeline does.
- The training step itself is intentionally tiny so it stays pdb-able.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

# Allow `import src.*` when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def plot_action_overlay(policy, dataset, preprocessor, device):
    """Predicted vs ground-truth actions on the overfit dataset, in normalized space.

    No SO-101 simulator exists in this repo, so this is the cheapest "did the
    policy actually memorize the trajectory?" check. Comparing in normalized
    space (preprocessed `batch["action"]` vs `policy.predict_action_chunk(...)`
    output) avoids round-tripping through the postprocessor — both are already
    in the same units. The diagnostic value (pred ≈ GT?) is identical to a
    physical-space plot.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: must be before pyplot import
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    pred_first, gt_first = [], []
    policy.eval()
    with torch.no_grad():
        for raw_batch in eval_loader:
            batch = preprocessor(raw_batch)
            # SmolVLA / lerobot v0.5.x exposes predict_action_chunk; fall back
            # to select_action if the API ever changes.
            if hasattr(policy, "predict_action_chunk"):
                pred = policy.predict_action_chunk(batch)
            else:
                pred = policy.select_action(batch)

            # Normalize shapes — predict_action_chunk returns (B, T, A) for ACT/SmolVLA,
            # select_action may return (B, A). Take the first action either way.
            pred_t = pred[:, 0, :] if pred.dim() == 3 else pred
            gt = batch["action"]
            gt_t = gt[:, 0, :] if gt.dim() == 3 else gt

            pred_first.append(pred_t.detach().cpu().float().numpy()[0])
            gt_first.append(gt_t.detach().cpu().float().numpy()[0])
    policy.train()

    pred_arr = np.stack(pred_first, axis=0)  # (T, A)
    gt_arr = np.stack(gt_first, axis=0)
    action_dim = pred_arr.shape[-1]

    # Try to pull human-readable per-dim labels from the dataset; fall back to indices.
    dim_names = [f"action[{i}]" for i in range(action_dim)]
    try:
        feats = getattr(dataset, "features", None) or {}
        action_feat = feats.get("action") if isinstance(feats, dict) else None
        names = getattr(action_feat, "names", None) if action_feat is not None else None
        if names and len(names) == action_dim:
            dim_names = list(names)
    except Exception:
        pass

    fig, axes = plt.subplots(action_dim, 1, sharex=True, figsize=(10, 1.6 * action_dim))
    if action_dim == 1:
        axes = [axes]
    timesteps = np.arange(pred_arr.shape[0])
    for i, ax in enumerate(axes):
        ax.plot(timesteps, gt_arr[:, i], label="ground truth", linewidth=1.5)
        ax.plot(timesteps, pred_arr[:, i], label="predicted", linestyle="--", linewidth=1.5)
        ax.set_ylabel(dim_names[i])
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("timestep")
    fig.suptitle("Overfit action overlay (normalized space)")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[overfit] config: {args.config}")
    print(f"[overfit] experiment: {cfg['experiment_name']}")

    # Each run gets its own timestamped subdir under cfg.train.output_dir, so
    # repeated overfits with the same yaml don't clobber each other's
    # checkpoints. Slurm job id is appended when available — easier to
    # cross-reference with logs/slurm-<jobid>.{out,err}.
    run_id = time.strftime("%Y%m%d-%H%M%S")
    if "SLURM_JOB_ID" in os.environ:
        run_id = f"{run_id}_job{os.environ['SLURM_JOB_ID']}"
    base_out = Path(cfg["train"]["output_dir"])
    cfg["train"]["output_dir"] = str(base_out / run_id)
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[overfit] run_id: {run_id}")
    print(f"[overfit] output_dir: {out_dir}")

    # Wandb init — after run_id/output_dir are finalized, so the snapshot
    # captures the real resolved paths. Conditional import keeps disabled
    # runs free of wandb side effects.
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
        print(f"[overfit] wandb: {wandb_run.url if wandb_run.url else '(offline)'}")

    # Lazy imports — keep argparse responsive even when torch is slow to import.
    import torch
    from torch.utils.data import DataLoader

    # lerobot v0.5.x — no `common/` namespace.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    torch.manual_seed(cfg["seed"])

    dcfg = cfg["data"]
    tcfg = cfg["train"]
    mcfg = cfg["model"]

    # ---- Dataset metadata (stats, features, fps) ----
    # Two modes:
    #   * Local fixture: `repo_id` is bookkeeping, `root` points at an
    #     existing on-disk LeRobotDataset (e.g. test_data/).
    #   * HF Hub: `repo_id` is the real `<user>/<dataset>` and `root` is
    #     None — lerobot downloads & caches under HF_LEROBOT_HOME (which
    #     train.slurm pins to $SCRATCH/Lerobot/hf_cache via HF_HOME).
    # If the dataset is private, the runner must have run
    # `hf auth login` first (inside the lerobot conda env).
    repo_id = dcfg.get("repo_id") or "local/test_episode"
    root = dcfg.get("root")
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)

    # ---- Policy config (must be built BEFORE the dataset, so we can use
    # its delta-indices to pull action chunks).
    policy_cfg = SmolVLAConfig(
        pretrained_path=mcfg["base"],
        chunk_size=mcfg["chunk_size"],
        n_action_steps=mcfg["chunk_size"],
        device=tcfg["device"],
        freeze_vision_encoder=mcfg["freeze_vision_encoder"],
    )

    # ---- Dataset (with delta_timestamps so each sample contains a full
    # action chunk, not just a single step).
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=dcfg["episodes"],
        delta_timestamps=delta_timestamps,
    )
    print(f"[overfit] dataset frames: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=0,            # keep simple for debugging
        drop_last=True,
    )

    # ---- Policy + pre/post processors ----
    # `make_policy` reads features from ds_meta and loads the pretrained
    # weights from `mcfg["base"]`. `make_pre_post_processors` loads the
    # tokenizer + normalizer state from the same checkpoint and overrides
    # normalization stats with this dataset's.
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.train()
    print(f"[overfit] params: {sum(p.numel() for p in policy.parameters()):,}")

    preprocessor, _postprocessor = make_pre_post_processors(
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

    # ---- Optim ----
    optim = torch.optim.AdamW(
        policy.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )

    # ---- Loop ----
    step = 0
    last_loss = float("inf")
    # out_dir already created above (before wandb.init)

    while step < tcfg["num_steps"]:
        for batch in loader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            optim.zero_grad()
            loss.backward()
            optim.step()

            last_loss = float(loss.detach().cpu())
            if step % tcfg["log_every"] == 0:
                print(f"[overfit] step={step:5d}  loss={last_loss:.4f}")
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": last_loss,
                            "train/lr": optim.param_groups[0]["lr"],
                        },
                        step=step,
                    )
            if step > 0 and step % tcfg["save_every"] == 0:
                policy.save_pretrained(out_dir / f"step_{step}")

            step += 1
            if step >= tcfg["num_steps"]:
                break

    # ---- Final save + pass/fail ----
    # try/finally so wandb.finish() runs even on exceptions, and so sys.exit
    # is deferred until after wandb has flushed.
    threshold = cfg["pass_criteria"]["final_loss_below"]
    passed = last_loss <= threshold
    try:
        final_dir = out_dir / "final"
        policy.save_pretrained(final_dir)
        # Also persist the pre/post processors so downstream loaders (the
        # lerobot v0.5 migrator, run_inference.py) get the normalization
        # stats baked in. Without these, state/action go through the
        # policy un-normalized and you get jittery / nonsense actions on
        # the real robot — same issue regardless of model dtype.
        preprocessor.save_pretrained(final_dir)
        _postprocessor.save_pretrained(final_dir)

        # Action-overlay plot: best-effort. A plotting/API issue must not
        # mask the training pass/fail signal — wrap in try/except.
        if use_wandb:
            try:
                fig = plot_action_overlay(policy, dataset, preprocessor, tcfg["device"])
                fig.savefig(out_dir / "action_overlay.png", dpi=120, bbox_inches="tight")
                wandb.log({"eval/action_overlay": wandb.Image(fig)})
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                print(f"[overfit] action overlay skipped: {e!r}")

        if use_wandb:
            wandb.summary["final_loss"] = last_loss
            wandb.summary["pass"] = bool(passed)
    finally:
        if use_wandb and wandb_run is not None:
            wandb.finish()

    if passed:
        print(f"[overfit] PASS  final_loss={last_loss:.4f}  <=  {threshold}")
        sys.exit(0)
    else:
        print(f"[overfit] FAIL  final_loss={last_loss:.4f}  >  {threshold}")
        print("[overfit] DO NOT scale up. Debug data loading / norm / loss first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
