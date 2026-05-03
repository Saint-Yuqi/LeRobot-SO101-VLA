"""Merge multiple LeRobotDataset v3.0 sessions into a single combined dataset.

Wraps `lerobot.datasets.aggregate.aggregate_datasets`, which handles
re-indexing episode_index/frame_index/index, copying videos with renumbered
filenames, and recomputing per-feature stats (mean/std/min/max/quantiles)
across the union of all source frames.

Usage:
    python scripts/merge_datasets.py \\
        --src data/raw/eval1_session1 data/raw/eval1_session2 data/raw/eval1_session3 \\
        --dst data/raw/eval1_merged \\
        --repo-id local/eval1_merged
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument(
        "--repo-id",
        default="local/eval1_merged",
        help="repo_id for the aggregated dataset (bookkeeping for local roots).",
    )
    args = ap.parse_args()

    # aggregate_datasets calls LeRobotDatasetMetadata.create which does
    # mkdir(exist_ok=False) on the destination — must NOT pre-create.
    if args.dst.exists():
        shutil.rmtree(args.dst)
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    src_repo_ids = [f"local/{p.name}" for p in args.src]

    aggregate_datasets(
        repo_ids=src_repo_ids,
        aggr_repo_id=args.repo_id,
        roots=list(args.src),
        aggr_root=args.dst,
    )

    print(f"merged into {args.dst}")
    for p in sorted(args.dst.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
