"""Aggregate per-batch JSONs from `scripts/benchmark_batch.py` into a markdown table.

Usage:
    python scripts/aggregate_bench.py logs/bench/<tag> [> summary.md]

Highlights the row matching the Stage-A selection rule:
    largest batch with samples_per_s within 2% of the max AND max_vram_gb <= 72.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

VRAM_HEADROOM_GB = 72.0   # 80 GB - 8 GB safety buffer
SAMPLES_TOLERANCE_PCT = 2.0


def fmt(x, fmt_spec=".2f"):
    if x is None:
        return "—"
    return format(x, fmt_spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_dir", help="logs/bench/<tag> directory")
    ap.add_argument("--vram-cap-gb", type=float, default=VRAM_HEADROOM_GB)
    ap.add_argument("--samples-tol-pct", type=float, default=SAMPLES_TOLERANCE_PCT)
    args = ap.parse_args()

    tag_dir = Path(args.tag_dir)
    files = sorted(glob.glob(str(tag_dir / "run_*.json")))
    if not files:
        print(f"no run_*.json under {tag_dir}", file=sys.stderr)
        sys.exit(1)

    rows = [json.load(open(f)) for f in files]
    rows.sort(key=lambda r: (r.get("batch_size", 0), r.get("compile", False), r.get("freeze_vision_encoder", False)))

    ok = [r for r in rows if r.get("status") == "ok" and r.get("max_vram_gb", 1e9) <= args.vram_cap_gb]
    if ok:
        max_samples = max(r["samples_per_s"] for r in ok)
        threshold = max_samples * (1.0 - args.samples_tol_pct / 100.0)
        eligible = [r for r in ok if r["samples_per_s"] >= threshold]
        winner = max(eligible, key=lambda r: r["batch_size"]) if eligible else None
    else:
        winner = None

    headers = ["batch", "compile", "freeze_ve", "status", "vram_gb",
               "steps/s", "samples/s", "stdev%", "util_avg", "util_min"]
    print(f"# Benchmark summary — `{tag_dir.name}`")
    print()
    print(f"VRAM cap: {args.vram_cap_gb:.0f} GB · samples/s tolerance: {args.samples_tol_pct:.1f}%")
    if winner:
        print(f"\n**Pick:** batch_size = **{winner['batch_size']}** "
              f"(compile={winner['compile']}, freeze_ve={winner['freeze_vision_encoder']}, "
              f"{winner['samples_per_s']:.1f} samples/s, {winner['max_vram_gb']:.1f} GB VRAM)")
    else:
        print("\n**No row passed the VRAM cap — nothing to pick.**")
    print()
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        marker = " ⬅" if winner and r is winner else ""
        print("| " + " | ".join([
            f"{r.get('batch_size','?')}{marker}",
            "✓" if r.get("compile") else "·",
            "✓" if r.get("freeze_vision_encoder") else "·",
            r.get("status", "?"),
            fmt(r.get("max_vram_gb"), ".1f"),
            fmt(r.get("steps_per_s"), ".3f"),
            fmt(r.get("samples_per_s"), ".1f"),
            fmt(r.get("stdev_step_pct"), ".1f"),
            fmt(r.get("gpu_util_avg"), ".0f"),
            fmt(r.get("gpu_util_min"), ".0f"),
        ]) + " |")


if __name__ == "__main__":
    main()
