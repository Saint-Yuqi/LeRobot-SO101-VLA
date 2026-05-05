"""Offline validation harness — confirm a trained checkpoint predicts
sensible actions before risking the real SO-101.

The wandb train/val loss curves can only tell you whether the model fits
the recorded distribution; they cannot tell you whether the model actually
*understands* the prompt. This script answers three orthogonal questions:

1. **Test A — Open-loop replay**: from N frames per held-out episode,
   predict the action chunk and compare with the ground-truth chunk in
   the dataset. Reports per-joint MAE and per-frame chunk MSE.

2. **Test B — Prompt-equivalence consistency** (the eval-2 acid test):
   for each frame, vary the prompt across all phrasings that point at
   the SAME target color (direct + ordinal + relational + negation, sourced
   from `src.data.prompt_aug.build_prompt_pool`) and compare the predicted
   chunks. We expect within-target cosine similarity > 0.9 and
   between-target cosine similarity < 0.7. The ratio (within / between)
   is the key metric: high → augmentation actually taught the model that
   these prompts are equivalent.

3. **Test C — OOD sanity**: feed nonsense / OOD prompts (a colour that
   doesn't exist, an irrelevant question, an empty string) and verify
   the prediction has no NaN/Inf and stays inside the training joint
   range. Catches catastrophic blow-ups before they reach the robot.

Usage
-----
    # eval-2 — full battery (uses arrangements for test B)
    python scripts/eval_offline.py \\
        --checkpoint PrajnaYang/so101-eval2-smolvla-v1 \\
        --dataset ethrl2026/so101_pickup_20260503_165245_task2 \\
        --arrangements configs/data/arrangements.json \\
        --frames-per-episode 4 \\
        --out reports/eval2_offline.json

    # eval-1 — only A + C (no compositional pool needed)
    python scripts/eval_offline.py \\
        --checkpoint PrajnaYang/so101-eval1-smolvla-v2 \\
        --dataset ethrl2026/so101_pickup_20260503_153511_task1 \\
        --frames-per-episode 4 \\
        --out reports/eval1_offline.json

Pass criteria (printed at the end):
  Test A: per-joint MAE < 5° (gripper < 0.5cm) on majority of frames
  Test B: within/between cos-sim ratio > 1.4
  Test C: zero NaN, zero out-of-range actions
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- OOD prompts (test C). Intentionally drawn from outside the training
# distribution: a non-existent colour, an irrelevant sentence, an empty
# string, and gibberish. Picked to stress the LM tokenizer + the policy
# head; not meant to be solvable. ----
OOD_PROMPTS = [
    "Put the banana in the purple colored bowl.",
    "What is the meaning of life?",
    "",
    "asdf qwerty zxcv banana bowl table",
]


# ---------------------------------------------------------------- helpers


def _resolve_checkpoint(arg: str) -> str:
    """Same lookup as run_inference.resolve_checkpoint — local dir or HF repo."""
    p = Path(arg)
    if p.is_dir() and (p / "config.json").exists():
        return str(p)
    if p.exists():
        raise SystemExit(
            f"--checkpoint '{arg}' exists but isn't a checkpoint dir (no config.json)."
        )
    if "/" not in arg or arg.count("/") > 1:
        raise SystemExit(
            f"--checkpoint '{arg}' is neither a local dir nor a HF repo id."
        )
    print(f"[eval] '{arg}' is not a local path; pulling from HuggingFace Hub…")
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=arg, repo_type="model")
    print(f"[eval] cached at {local}")
    return local


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays."""
    af, bf = a.flatten(), b.flatten()
    n = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return float(np.dot(af, bf) / n) if n > 0 else 0.0


def _summary(arr: list[float]) -> dict:
    if not arr:
        return {"n": 0}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
    }


# ---------------------------------------------------------------- data


def load_test_samples(repo_id: str, episodes: list[int], frames_per_ep: int,
                      chunk_size: int):
    """Pull `frames_per_ep` evenly-spaced frames from each requested episode.

    Returns a list of dicts with keys:
        image_uint8      : (H, W, 3) uint8 RGB
        state            : (action_dim,) float32
        task             : str (the recorded prompt)
        action_gt        : (chunk_size, action_dim) float32 ground-truth chunk
        episode_index    : int
        frame_index      : int (within-episode 0-based)
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id=repo_id)
    fps = float(meta.fps)
    delta_ts = {"action": [i / fps for i in range(chunk_size)]}

    samples: list[dict] = []
    for ep in episodes:
        ds = LeRobotDataset(repo_id=repo_id, episodes=[ep], delta_timestamps=delta_ts)
        n = len(ds)
        # Drop the final chunk_size-1 frames so action_gt isn't truncated.
        last = max(0, n - chunk_size)
        if last <= 0:
            print(f"[eval]   episode {ep}: only {n} frames < chunk_size={chunk_size}, skipping")
            continue
        idxs = np.linspace(0, last, frames_per_ep, dtype=int)
        for fi in idxs:
            s = ds[int(fi)]
            img_t = s["observation.images.main"]  # (3, H, W) float [0,1]
            img_uint8 = (img_t.permute(1, 2, 0).cpu().numpy() * 255.0
                         ).clip(0, 255).astype(np.uint8)
            samples.append({
                "image_uint8": img_uint8,
                "state": s["observation.state"].cpu().numpy().astype(np.float32),
                "task": s["task"],
                "action_gt": s["action"].cpu().numpy().astype(np.float32),
                "episode_index": int(s["episode_index"].item()
                                     if hasattr(s["episode_index"], "item")
                                     else s["episode_index"]),
                "frame_index": int(fi),
            })
    print(f"[eval] collected {len(samples)} test frames "
          f"({frames_per_ep}/episode × {len(episodes)} episodes)")
    return samples, meta


# ---------------------------------------------------------------- inference


def predict(wrapper, sample: dict, prompt: str) -> np.ndarray:
    """Single forward pass returning (chunk_size, action_dim) numpy chunk."""
    from src.models.base_vla import Observation
    obs = Observation(
        images={"main": sample["image_uint8"]},
        state=sample["state"],
        prompt=prompt,
    )
    return wrapper.predict(obs).actions


# ---------------------------------------------------------------- tests


def test_a_replay(wrapper, samples, action_lo, action_hi):
    """Predict each sample's chunk, compare with ground truth."""
    print("\n=== Test A: open-loop replay ===")
    per_joint_mae: list[np.ndarray] = []
    chunk_l2: list[float] = []
    nan_count = 0
    out_of_range = 0
    rng = action_hi - action_lo
    for s in samples:
        pred = predict(wrapper, s, s["task"])
        if not np.isfinite(pred).all():
            nan_count += 1
            continue
        diff = np.abs(pred - s["action_gt"])
        per_joint_mae.append(diff.mean(axis=0))
        chunk_l2.append(float(np.linalg.norm(diff) / np.sqrt(diff.size)))
        # Out of training range = beyond [lo - 0.05*rng, hi + 0.05*rng]
        margin = 0.05 * np.maximum(rng, 1e-6)
        if ((pred < action_lo - margin) | (pred > action_hi + margin)).any():
            out_of_range += 1
    if not per_joint_mae:
        return {"error": "no finite predictions"}
    pjm = np.stack(per_joint_mae)  # (N, action_dim)
    out = {
        "n_samples": len(samples),
        "n_nan": nan_count,
        "n_out_of_training_range": out_of_range,
        "per_joint_mae_mean": pjm.mean(axis=0).tolist(),
        "per_joint_mae_max": pjm.max(axis=0).tolist(),
        "chunk_rms": _summary(chunk_l2),
    }
    print(f"  samples              : {out['n_samples']}")
    print(f"  NaN predictions      : {out['n_nan']}")
    print(f"  out-of-train-range   : {out['n_out_of_training_range']}")
    print(f"  per-joint MAE (mean) : {[f'{v:.4f}' for v in out['per_joint_mae_mean']]}")
    print(f"  per-joint MAE (max)  : {[f'{v:.4f}' for v in out['per_joint_mae_max']]}")
    print(f"  chunk RMS  mean={out['chunk_rms']['mean']:.4f}  "
          f"median={out['chunk_rms']['median']:.4f}  "
          f"max={out['chunk_rms']['max']:.4f}")
    return out


def test_b_consistency(wrapper, samples, arrangements: dict[int, list[str]]):
    """For each sample, vary prompt across the equivalent pool. Compute
    within-target and between-target cosine similarity over predicted chunks."""
    print("\n=== Test B: prompt-equivalence consistency ===")
    from src.data.prompt_aug import _target_from_task, build_prompt_pool

    within_sims: list[float] = []   # cos sim among same-target prompts
    between_sims: list[float] = []  # cos sim across different-target prompts
    skipped = 0
    per_sample_rows: list[dict] = []

    for s in samples:
        ep = s["episode_index"]
        if ep not in arrangements:
            skipped += 1
            continue
        arr = arrangements[ep]
        gt_target = _target_from_task(s["task"])
        if gt_target is None or gt_target not in arr:
            skipped += 1
            continue

        # Predict for every (target, phrasing) cell.
        # rows[target] = list of (chunk_size, action_dim) numpy
        rows: dict[str, list[np.ndarray]] = {}
        for tgt in arr:
            pool = build_prompt_pool(arr, tgt)
            rows[tgt] = [predict(wrapper, s, p) for p in pool]

        # WITHIN-target cosine similarity (across phrasings).
        # Reference is the canonical direct-color phrase (index 0 of pool).
        canon = rows[gt_target][0]
        for chunk in rows[gt_target][1:]:
            within_sims.append(_cos(canon, chunk))

        # BETWEEN-target: canonical-of-target vs canonical-of-other-target
        for other in arr:
            if other == gt_target:
                continue
            between_sims.append(_cos(canon, rows[other][0]))

        # Per-sample summary
        within_for_sample = [
            _cos(canon, c) for c in rows[gt_target][1:]
        ]
        between_for_sample = [
            _cos(canon, rows[o][0]) for o in arr if o != gt_target
        ]
        per_sample_rows.append({
            "episode_index": ep,
            "frame_index": s["frame_index"],
            "arrangement": arr,
            "gt_target": gt_target,
            "within_mean": float(np.mean(within_for_sample)) if within_for_sample else None,
            "between_mean": float(np.mean(between_for_sample)) if between_for_sample else None,
        })

    if not within_sims:
        return {"error": "no usable samples (arrangements not provided or no target match)"}

    out = {
        "n_samples": len(samples) - skipped,
        "n_skipped": skipped,
        "within_target_cos": _summary(within_sims),
        "between_target_cos": _summary(between_sims) if between_sims else {"n": 0},
        "ratio_within_over_between": (
            (sum(within_sims) / len(within_sims)) /
            (sum(between_sims) / len(between_sims))
        ) if between_sims else None,
        "per_sample": per_sample_rows[:30],   # cap to keep report compact
    }
    print(f"  usable / skipped      : {out['n_samples']} / {out['n_skipped']}")
    print(f"  within-target  cos-sim mean={out['within_target_cos']['mean']:.4f}  "
          f"min={out['within_target_cos']['min']:.4f}  "
          f"median={out['within_target_cos']['median']:.4f}")
    if between_sims:
        print(f"  between-target cos-sim mean={out['between_target_cos']['mean']:.4f}  "
              f"max={out['between_target_cos']['max']:.4f}")
        print(f"  ratio (within/between): {out['ratio_within_over_between']:.3f}  "
              "(>1.4 ≈ augmentation working)")
    return out


def test_c_sanity(wrapper, samples, action_lo, action_hi, ood_prompts=OOD_PROMPTS):
    """Feed OOD prompts; verify no NaN/Inf and predictions stay within
    the training action range (with 10% margin)."""
    print("\n=== Test C: OOD sanity ===")
    rng = action_hi - action_lo
    margin = 0.10 * np.maximum(rng, 1e-6)
    nan_total = 0
    out_of_range_total = 0
    deviations: list[float] = []
    n_trials = 0
    for s in samples:
        for p in ood_prompts:
            n_trials += 1
            chunk = predict(wrapper, s, p)
            if not np.isfinite(chunk).all():
                nan_total += 1
                continue
            oor = ((chunk < action_lo - margin) | (chunk > action_hi + margin))
            if oor.any():
                out_of_range_total += 1
            # deviation from the centre of the training range, normalised
            centre = 0.5 * (action_lo + action_hi)
            dev = float(np.max(np.abs(chunk - centre) / np.maximum(rng, 1e-6)))
            deviations.append(dev)
    out = {
        "n_trials": n_trials,
        "n_nan": nan_total,
        "n_out_of_range": out_of_range_total,
        "max_normalized_deviation": _summary(deviations),
        "ood_prompts": ood_prompts,
    }
    print(f"  trials               : {out['n_trials']}")
    print(f"  NaN/Inf              : {out['n_nan']}")
    print(f"  out-of-train-range   : {out['n_out_of_range']}")
    print(f"  normalized deviation : "
          f"mean={out['max_normalized_deviation']['mean']:.3f}  "
          f"max={out['max_normalized_deviation']['max']:.3f}")
    return out


def test_d_action_overlay(wrapper, repo_id: str, episodes: list[int],
                          chunk_size: int, out_dir: Path,
                          wandb_run=None) -> dict:
    """Per-episode predicted-vs-ground-truth action trajectory overlay.

    For each episode, walk every frame, predict the action chunk, take the
    FIRST action of each chunk and stitch into a trajectory. Plot this
    against the recorded ground-truth first-action sequence per joint. This
    mirrors the closed-loop MPC execution pattern (`run_inference.py`
    refills its action queue from chunk[0:N] each time it's empty), so the
    overlay reflects what the policy would actually drive on the robot.

    Reveals failures the cosine / MAE metrics hide:
      * gripper open/close phase shift (failure mode #1 on real demos)
      * single-joint drift while others are fine
      * mode collapse to a mean trajectory (flat predicted line)
    """
    print("\n=== Test D: action trajectory overlay ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id=repo_id)
    fps = float(meta.fps)
    delta_ts = {"action": [i / fps for i in range(chunk_size)]}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pull human-readable joint labels if the dataset provides them.
    dim_names = None
    try:
        action_feat = meta.features.get("action")
        names = getattr(action_feat, "names", None) if action_feat else None
        if names:
            dim_names = list(names)
    except Exception:
        pass

    overlays_written: list[str] = []
    per_episode_mae: list[dict] = []
    for ep in episodes:
        ds = LeRobotDataset(repo_id=repo_id, episodes=[ep],
                            delta_timestamps=delta_ts, force_cache_sync=True)
        n = len(ds)
        if n < 2:
            print(f"  ep{ep}: too short ({n} frames), skipping")
            continue
        pred_first, gt_first = [], []
        prompt = None
        for fi in range(n):
            s = ds[fi]
            img_t = s["observation.images.main"]
            img_uint8 = (img_t.permute(1, 2, 0).cpu().numpy() * 255.0
                         ).clip(0, 255).astype(np.uint8)
            sample = {
                "image_uint8": img_uint8,
                "state": s["observation.state"].cpu().numpy().astype(np.float32),
            }
            if prompt is None:
                prompt = s["task"]
            chunk = predict(wrapper, sample, prompt)  # (chunk_size, action_dim)
            pred_first.append(chunk[0])
            gt_first.append(s["action"].cpu().numpy().astype(np.float32)[0])
        pred_arr = np.stack(pred_first, axis=0)        # (T, A)
        gt_arr = np.stack(gt_first, axis=0)            # (T, A)
        action_dim = pred_arr.shape[-1]
        labels = (dim_names if dim_names and len(dim_names) == action_dim
                  else [f"action[{i}]" for i in range(action_dim)])

        # Per-joint MAE for the verdict.
        per_joint_mae = np.abs(pred_arr - gt_arr).mean(axis=0)
        per_episode_mae.append({
            "episode": ep,
            "n_frames": int(n),
            "prompt": prompt,
            "per_joint_mae": per_joint_mae.tolist(),
            "overall_mae": float(per_joint_mae.mean()),
        })

        fig, axes = plt.subplots(action_dim, 1, sharex=True,
                                 figsize=(10, 1.5 * action_dim))
        if action_dim == 1:
            axes = [axes]
        ts = np.arange(pred_arr.shape[0]) / fps
        for i, ax in enumerate(axes):
            ax.plot(ts, gt_arr[:, i], label="ground truth", linewidth=1.4)
            ax.plot(ts, pred_arr[:, i], label="predicted", linestyle="--", linewidth=1.4)
            ax.set_ylabel(labels[i])
            ax.grid(alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"ep{ep}  prompt: {prompt}  ({repo_id})")
        fig.tight_layout()
        png = out_dir / f"action_overlay_ep{ep:03d}.png"
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        overlays_written.append(str(png))
        print(f"  ep{ep:3d}  frames={n}  overall_mae={float(per_joint_mae.mean()):.4f}  -> {png.name}")

        if wandb_run is not None:
            try:
                import wandb
                wandb_run.log({f"eval/action_overlay/ep{ep}": wandb.Image(str(png))})
            except Exception as e:
                print(f"  ep{ep}: wandb upload skipped: {e!r}")

    return {
        "n_episodes": len(per_episode_mae),
        "overlay_pngs": overlays_written,
        "per_episode": per_episode_mae,
    }


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Local checkpoint dir OR HuggingFace repo id.")
    parser.add_argument("--dataset", required=True,
                        help="HF dataset repo_id to pull test frames from.")
    parser.add_argument("--episodes", default=None,
                        help="Comma-separated episode indices. Default: 5 evenly-spaced "
                             "from across the dataset.")
    parser.add_argument("--frames-per-episode", type=int, default=4)
    parser.add_argument("--arrangements", default=None,
                        help="Path to configs/data/arrangements.json. "
                             "Required for Test B; without it Test B is skipped.")
    parser.add_argument("--out", default=None,
                        help="Optional JSON report path.")
    parser.add_argument("--overlay-dir", default=None,
                        help="If set, run Test D (per-episode action overlay PNGs) "
                             "and write them under this directory.")
    parser.add_argument("--overlay-episodes", default=None,
                        help="Comma-separated episodes for the overlay (defaults to "
                             "--episodes if omitted; falls back to 3 evenly-spaced).")
    parser.add_argument("--wandb-run", default=None,
                        help="Optional wandb run id (e.g. \"<entity>/Lerobot/<run_id>\") "
                             "to log eval/action_overlay images alongside training metrics.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    t0 = time.time()
    ckpt = _resolve_checkpoint(args.checkpoint)

    # Load wrapper + processors.
    from src.models.smolvla_wrapper import SmolVLAWrapper
    wrapper = SmolVLAWrapper.from_checkpoint(ckpt, camera_keys=("main",))
    wrapper = wrapper.to(args.device).eval()
    print(f"[eval] checkpoint loaded ({wrapper.active_param_count:,} params)")

    # Pick episodes if not given. 5 evenly spaced across the dataset.
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(repo_id=args.dataset)
    n_eps_total = meta.total_episodes
    if args.episodes:
        episodes = [int(x) for x in args.episodes.split(",")]
    else:
        episodes = np.linspace(0, n_eps_total - 1, 5, dtype=int).tolist()
    print(f"[eval] testing on {len(episodes)} episodes from {args.dataset}: {episodes}")

    # Pull samples.
    chunk_size = 50  # SmolVLA default
    samples, _ = load_test_samples(args.dataset, episodes, args.frames_per_episode, chunk_size)
    if not samples:
        raise SystemExit("[eval] no test samples collected")

    # Compute action range from dataset stats (for Tests A + C bound checks).
    action_stats = meta.stats.get("action", {})
    if "min" in action_stats and "max" in action_stats:
        action_lo = np.asarray(action_stats["min"], dtype=np.float32).reshape(-1)
        action_hi = np.asarray(action_stats["max"], dtype=np.float32).reshape(-1)
    else:
        # Fallback: derive from collected ground-truth chunks (worse, but workable).
        all_gt = np.concatenate([s["action_gt"] for s in samples], axis=0)
        action_lo, action_hi = all_gt.min(axis=0), all_gt.max(axis=0)
    print(f"[eval] action range per joint:")
    for j, (lo, hi) in enumerate(zip(action_lo, action_hi)):
        print(f"          joint {j}: [{lo:.3f}, {hi:.3f}]")

    report = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "episodes": episodes,
        "frames_per_episode": args.frames_per_episode,
        "n_samples": len(samples),
        "elapsed_s_load": time.time() - t0,
    }
    report["test_a"] = test_a_replay(wrapper, samples, action_lo, action_hi)

    if args.arrangements:
        from src.data.prompt_aug import load_arrangements
        arrangements = load_arrangements(args.arrangements, args.dataset)
        if not arrangements:
            print(f"[eval] WARNING: arrangements file has no entry for "
                  f"{args.dataset!r}; skipping Test B")
        else:
            report["test_b"] = test_b_consistency(wrapper, samples, arrangements)
    else:
        print("\n[eval] --arrangements not given; skipping Test B")

    report["test_c"] = test_c_sanity(wrapper, samples, action_lo, action_hi)

    # Test D: action trajectory overlay PNGs (opt-in via --overlay-dir).
    if args.overlay_dir:
        overlay_eps = (
            [int(x) for x in args.overlay_episodes.split(",")]
            if args.overlay_episodes
            else (episodes[:3] if len(episodes) >= 3
                  else np.linspace(0, n_eps_total - 1, 3, dtype=int).tolist())
        )
        wandb_run = None
        if args.wandb_run:
            try:
                import wandb
                ent_proj_run = args.wandb_run.split("/")
                wandb_run = wandb.init(
                    project=ent_proj_run[1] if len(ent_proj_run) >= 2 else "Lerobot",
                    entity=ent_proj_run[0] if len(ent_proj_run) >= 2 else None,
                    id=ent_proj_run[-1],
                    resume="must",
                )
                print(f"[eval] resumed wandb run {args.wandb_run}")
            except Exception as e:
                print(f"[eval] WARNING: wandb resume failed ({e!r}); writing PNGs only")
        report["test_d"] = test_d_action_overlay(
            wrapper, args.dataset, overlay_eps, chunk_size,
            Path(args.overlay_dir), wandb_run=wandb_run,
        )
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
    report["elapsed_s_total"] = time.time() - t0

    # ---- Pass / fail verdict ----
    # MAE thresholds are RELATIVE to each joint's action range so the same
    # criterion works in normalized space (range≈[-1,1]) and in raw degrees
    # (SO-101 ranges 70–185°). 10% of range is the cutoff for replay (Test A,
    # full chunk prediction is harder), 5% for the closed-loop overlay
    # (Test D, only chunk[0] each step — easier).
    print("\n=== Verdict ===")
    joint_range = np.maximum(action_hi - action_lo, 1e-6)
    a_pj_mae = np.asarray(report["test_a"]["per_joint_mae_mean"])
    a_rel = float(np.mean(a_pj_mae / joint_range))
    a_ok = report["test_a"].get("n_nan", 0) == 0 and a_rel < 0.10
    print(f"  Test A (replay)       : {'PASS' if a_ok else 'FAIL'}  "
          f"(mean MAE / range = {a_rel*100:.2f}%)")
    if "test_b" in report:
        b = report["test_b"]
        ratio = b.get("ratio_within_over_between") or 0.0
        b_ok = (b["within_target_cos"]["mean"] > 0.85 and ratio > 1.4)
        print(f"  Test B (consistency)  : {'PASS' if b_ok else 'FAIL'}  "
              f"(within={b['within_target_cos']['mean']:.3f}, ratio={ratio:.2f})")
    c_ok = (report["test_c"]["n_nan"] == 0 and
            report["test_c"]["n_out_of_range"] == 0)
    print(f"  Test C (OOD sanity)   : {'PASS' if c_ok else 'FAIL'}")
    if "test_d" in report:
        per_ep = report["test_d"]["per_episode"]
        if per_ep:
            mean_overall_mae = float(np.mean([p["overall_mae"] for p in per_ep]))
            mean_joint_range = float(np.mean(joint_range))
            d_rel = mean_overall_mae / max(mean_joint_range, 1e-6)
            d_ok = d_rel < 0.05
            print(f"  Test D (overlay)      : {'PASS' if d_ok else 'FAIL'}  "
                  f"(mean MAE={mean_overall_mae:.3f}°, "
                  f"= {d_rel*100:.2f}% of range, "
                  f"PNGs in {Path(args.overlay_dir).resolve()})")
        else:
            print(f"  Test D (overlay)      : SKIP (no episodes overlaid)")
    print(f"  total elapsed         : {report['elapsed_s_total']:.1f}s")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"[eval] report → {out}")


if __name__ == "__main__":
    main()
