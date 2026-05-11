"""One-shot probe to validate the open->closed gripper detector defaults.

Reads action[:, 5] (gripper) per episode for the three SO-101 task datasets and
prints a small sparkline + the detected close-frame index t* + per-episode
pregrasp_frac. The defaults (close_threshold=10.0, open_threshold=22.0,
post_close_margin=3) are grounded in eval1_merged stats.json but need a quick
real-trajectory sanity check before they ship into src/data/phase_labels.py.

Run:
    python scripts/spikes/probe_phase_detector.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


import argparse

DATASETS = [
    ("eval1", "ethrl2026/task1_20260509_prompt_lighting_augmented_360"),
    ("eval2", "ethrl2026/task2_20260509_stage2_random_lighting_augmented_2160"),
    ("eval3", "ethrl2026/so101_pickup_20260509_185350_task3"),
]
SAMPLE_EPISODES_PER_DATASET = 6

POST_CLOSE_MARGIN = 3
# Adaptive thresholds: use per-episode min/max so the detector tracks the
# actual gripper amplitude in *this* demo, not a global absolute.
OPEN_FRAC = 0.6   # fraction of (max-min) above min to count as "open"
CLOSE_FRAC = 0.4  # below this fraction is "closed"
MIN_AMPLITUDE = 5.0  # gripper must move at least this much across episode

SPARK = "▁▂▃▄▅▆▇█"


def find_close_frame_adaptive(g, post_close_margin=POST_CLOSE_MARGIN,
                              open_frac=OPEN_FRAC, close_frac=CLOSE_FRAC,
                              min_amplitude=MIN_AMPLITUDE):
    g_min, g_max = float(g.min()), float(g.max())
    amplitude = g_max - g_min
    if amplitude < min_amplitude:
        return None  # gripper barely moved (degenerate / static demo)
    open_threshold = g_min + open_frac * amplitude
    close_threshold = g_min + close_frac * amplitude
    open_idxs = np.where(g > open_threshold)[0]
    if len(open_idxs) == 0:
        return None
    t_open = int(open_idxs[0])
    close_idxs = np.where(g[t_open:] < close_threshold)[0]
    if len(close_idxs) == 0:
        return None
    t_star = t_open + int(close_idxs[0])
    end = min(t_star + post_close_margin, len(g) - 1)
    if g[t_star : end + 1].max() >= close_threshold:
        # transient dip — retry past it
        t_search = t_star + 1
        while True:
            close_idxs2 = np.where(g[t_search:] < close_threshold)[0]
            if len(close_idxs2) == 0:
                return None
            t_star2 = t_search + int(close_idxs2[0])
            end2 = min(t_star2 + post_close_margin, len(g) - 1)
            if g[t_star2 : end2 + 1].max() < close_threshold:
                return t_star2
            t_search = t_star2 + 1
    return t_star


find_close_frame = find_close_frame_adaptive


def sparkline(arr, width=80):
    if len(arr) == 0:
        return ""
    arr = np.asarray(arr, dtype=np.float32)
    n = len(arr)
    if n > width:
        bin_edges = np.linspace(0, n, width + 1, dtype=int)
        ds = np.array([arr[bin_edges[i]:bin_edges[i+1]].mean() if bin_edges[i+1] > bin_edges[i] else arr[bin_edges[i]]
                       for i in range(width)])
    else:
        ds = arr
    lo, hi = ds.min(), ds.max()
    rng = max(hi - lo, 1e-6)
    norm = ((ds - lo) / rng * (len(SPARK) - 1)).astype(int)
    return "".join(SPARK[i] for i in norm)


def load_dataset_root(repo_id: str) -> Path:
    # Prefer reading the local HF snapshot if it already exists (avoids API
    # quota: snapshot_download still hits the API even when files are cached).
    cache_root = Path.home() / ".cache/huggingface/hub" / f"datasets--{repo_id.replace('/', '--')}/snapshots"
    if cache_root.exists():
        snaps = sorted(cache_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snaps:
            if (snap / "meta/info.json").exists():
                return snap
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub not available; activate flower env")
    return Path(snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision="v3.0",
        allow_patterns=["meta/*", "data/**"],
    ))


def load_episode_table(root: Path):
    ep_files = sorted((root / "meta/episodes").rglob("file-*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"no episode meta under {root}")
    tables = [pq.read_table(f).to_pandas() for f in ep_files]
    import pandas as pd
    return pd.concat(tables, ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def gripper_for_episode(root: Path, ep_row) -> np.ndarray | None:
    data_path = root / f"data/chunk-{int(ep_row['data/chunk_index']):03d}/file-{int(ep_row['data/file_index']):03d}.parquet"
    if not data_path.exists():
        return None
    tbl = pq.read_table(data_path, columns=["episode_index", "action", "frame_index"])
    df = tbl.to_pandas()
    df = df[df["episode_index"] == int(ep_row["episode_index"])].sort_values("frame_index")
    actions = np.stack([np.asarray(a, dtype=np.float32) for a in df["action"].values])
    return actions[:, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Tags to probe (eval1 / eval2 / eval3). Default: all.")
    args = ap.parse_args()
    targets = DATASETS if args.only is None else [d for d in DATASETS if d[0] in set(args.only)]
    for tag, repo_id in targets:
        print(f"\n========== {tag} :: {repo_id} ==========")
        try:
            root = load_dataset_root(repo_id)
        except Exception as e:
            print(f"  skip — could not load: {e}")
            continue
        ep_df = load_episode_table(root)
        n_eps = len(ep_df)
        print(f"  {n_eps} episodes; probing first {SAMPLE_EPISODES_PER_DATASET}")

        ds_total_frames = 0
        ds_pregrasp_frames = 0
        ds_n_failed = 0
        n_probed = 0

        for _, ep_row in ep_df.iterrows():
            g = gripper_for_episode(root, ep_row)
            if g is None:
                continue
            n_probed += 1
            ep_len = len(g)
            t_star = find_close_frame(g)
            if t_star is None:
                ds_n_failed += 1
                pre_frames = ep_len  # all-pregrasp fallback
            else:
                pre_frames = min(t_star + POST_CLOSE_MARGIN + 1, ep_len)
            ds_total_frames += ep_len
            ds_pregrasp_frames += pre_frames

        ds_pregrasp_frac = ds_pregrasp_frames / max(ds_total_frames, 1)
        print(f"  probed {n_probed}/{n_eps} episodes (rest had missing parquets)")
        print(f"  dataset pregrasp_frac = {ds_pregrasp_frac:.3f}  "
              f"(n_failed={ds_n_failed}/{n_probed} = {ds_n_failed/max(n_probed,1):.1%})")

        # Show per-episode detail for first SAMPLE_EPISODES_PER_DATASET
        shown = 0
        for _, ep_row in ep_df.iterrows():
            if shown >= SAMPLE_EPISODES_PER_DATASET:
                break
            g = gripper_for_episode(root, ep_row)
            if g is None:
                continue
            shown += 1
            t_star = find_close_frame(g)
            ep_idx = int(ep_row["episode_index"])
            ep_len = len(g)
            spark = sparkline(g, width=60)
            gmin, gmax, gmean = g.min(), g.max(), g.mean()
            if t_star is None:
                marker = "  t*=None (FAIL)"
                pre = ep_len
            else:
                marker = f"  t*={t_star:4d} ({t_star/ep_len:.0%})"
                pre = min(t_star + POST_CLOSE_MARGIN + 1, ep_len)
            print(f"  ep{ep_idx:03d} N={ep_len:4d} g=[{gmin:5.1f},{gmax:5.1f}] mean={gmean:5.1f}  "
                  f"|{spark}|{marker}  pre_frac={pre/ep_len:.2f}")


if __name__ == "__main__":
    main()
