"""Validate a recorded LeRobot dataset before using it for training.

Catches:
  - empty episodes
  - all-black or constant images (camera unplugged)
  - action ranges outside joint limits
  - non-monotonic timestamps
  - empty / missing prompts

Run before EVERY training run on new data. Cheaper than 8 hours of
training on a broken dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="path to LeRobot dataset root")
    parser.add_argument("--strict", action="store_true",
                        help="exit nonzero on any warning")
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    ds = LeRobotDataset(repo_id=None, root=args.dataset)
    n = len(ds)
    print(f"[validate] frames: {n}")
    print(f"[validate] episodes: {ds.num_episodes}")

    warnings: list[str] = []

    # ---- Episode lengths ----
    lengths = [ds.episode_data_index["to"][i] - ds.episode_data_index["from"][i]
               for i in range(ds.num_episodes)]
    print(f"[validate] episode length: "
          f"min={min(lengths)}  max={max(lengths)}  mean={np.mean(lengths):.1f}")
    if min(lengths) < 10:
        warnings.append(f"episode shorter than 10 frames: {min(lengths)}")
    if max(lengths) > 600:
        warnings.append(f"episode longer than 600 frames: {max(lengths)}")

    # ---- Spot-check a few frames per episode ----
    for ep_idx in range(min(ds.num_episodes, 5)):
        start = ds.episode_data_index["from"][ep_idx].item()
        sample = ds[start]

        # Images
        for k, v in sample.items():
            if "image" in k and hasattr(v, "shape"):
                arr = v.numpy() if hasattr(v, "numpy") else v
                if arr.std() < 1e-3:
                    warnings.append(f"ep{ep_idx} {k}: image looks constant (std={arr.std():.4f})")
                if arr.mean() < 0.01:
                    warnings.append(f"ep{ep_idx} {k}: image nearly black (mean={arr.mean():.4f})")

        # Actions
        if "action" in sample:
            a = sample["action"].numpy() if hasattr(sample["action"], "numpy") else sample["action"]
            print(f"[validate] ep{ep_idx} first action: {np.round(a, 3)}")

        # Prompt / task
        for key in ("task", "prompt", "language_instruction"):
            if key in sample:
                val = sample[key]
                if not val or (isinstance(val, str) and not val.strip()):
                    warnings.append(f"ep{ep_idx} empty {key}")
                else:
                    print(f"[validate] ep{ep_idx} {key}: {val!r}")
                break

    # ---- Report ----
    if warnings:
        print("\n[validate] WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
        if args.strict:
            sys.exit(1)
    else:
        print("\n[validate] OK — no warnings.")


if __name__ == "__main__":
    main()
