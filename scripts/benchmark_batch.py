"""Single-batch throughput probe for the A100 sweep.

One process measures one (batch_size, compile, freeze_vision_encoder)
combo and writes a small JSON to `logs/bench/<tag>/run_<batch>.json`.
This is the unit invoked by `scripts/bench_array.slurm` — running each
combo in its own process means an OOM kills only that array task, not
the whole sweep.

Stages from the plan:
  A: batch sweep at fixed (compile=false, freeze_ve=false)
  B: compile/freeze cross at fixed B*
  C: batch re-sweep on top of stage-B winner

Usage:
    python scripts/benchmark_batch.py --config configs/train/bench.yaml \\
        --batch-size 192 --tag stageA_2026XXXX [--compile] [--freeze-ve]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/bench.yaml")
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--num-steps", type=int, default=None,
                    help="overrides train.num_steps; warmup+measure must sum to this")
    ap.add_argument("--warmup", type=int, default=None,
                    help="overrides bench.warmup_steps")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.set_defaults(compile=None)
    ap.add_argument("--freeze-ve", action="store_true", dest="freeze_ve")
    ap.add_argument("--no-freeze-ve", action="store_false", dest="freeze_ve")
    ap.set_defaults(freeze_ve=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--tag", default=None,
                    help="subdir under bench.out_dir; defaults to slurm job id or timestamp")
    ap.add_argument("--out", default=None,
                    help="explicit output JSON path; overrides tag-based naming")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bcfg = cfg.get("bench") or {}
    tcfg = cfg["train"]
    mcfg = cfg["model"]
    dcfg = cfg["data"]

    # Apply CLI overrides
    tcfg["batch_size"] = args.batch_size
    if args.num_steps is not None:
        tcfg["num_steps"] = args.num_steps
    if args.num_workers is not None:
        tcfg["num_workers"] = args.num_workers
    if args.compile is not None:
        bcfg["compile"] = args.compile
    if args.freeze_ve is not None:
        mcfg["freeze_vision_encoder"] = args.freeze_ve

    warmup_steps = int(args.warmup if args.warmup is not None else bcfg.get("warmup_steps", 30))
    total_steps = int(tcfg["num_steps"])
    measure_steps = total_steps - warmup_steps
    if measure_steps <= 0:
        raise ValueError(f"warmup ({warmup_steps}) >= num_steps ({total_steps})")

    tag = args.tag or os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") \
        or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(bcfg.get("out_dir", "logs/bench")) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        out_dir / f"run_b{args.batch_size}"
        f"_c{int(bcfg.get('compile', False))}"
        f"_f{int(mcfg['freeze_vision_encoder'])}.json"
    )

    print(f"[bench] tag={tag}  batch={args.batch_size}  compile={bcfg.get('compile', False)}  "
          f"freeze_ve={mcfg['freeze_vision_encoder']}  warmup={warmup_steps}  measure={measure_steps}")

    record: dict = {
        "batch_size": args.batch_size,
        "compile": bool(bcfg.get("compile", False)),
        "freeze_vision_encoder": bool(mcfg["freeze_vision_encoder"]),
        "num_workers": int(tcfg["num_workers"]),
        "warmup_steps": warmup_steps,
        "measure_steps_target": measure_steps,
        "tag": tag,
        "config": args.config,
    }

    # ---- Lazy heavy imports ----
    import torch
    from torch.utils.data import DataLoader

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    from src.utils.gpu_metrics import GpuSampler

    torch.manual_seed(int(cfg.get("seed", 42)))

    repo_id = dcfg.get("repo_id") or "local/eval1_merged"
    root = dcfg.get("root")
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)

    policy_cfg = SmolVLAConfig(
        pretrained_path=mcfg["base"],
        chunk_size=mcfg["chunk_size"],
        n_action_steps=mcfg["chunk_size"],
        device=tcfg["device"],
        freeze_vision_encoder=mcfg["freeze_vision_encoder"],
    )
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(
        repo_id=repo_id, root=root,
        episodes=dcfg.get("episodes"),
        delta_timestamps=delta_timestamps,
    )
    record["dataset_frames"] = len(dataset)
    record["dataset_episodes"] = ds_meta.total_episodes

    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=True,
        persistent_workers=tcfg.get("num_workers", 4) > 0,
    )

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.train()
    record["params"] = sum(p.numel() for p in policy.parameters())

    preprocessor, _ = make_pre_post_processors(
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

    if bcfg.get("compile", False):
        # `reduce-overhead` cuts CUDA graph capture overhead; first step is
        # slow (trace+compile), so warmup_steps should be ≥60 for compile runs.
        policy = torch.compile(policy, mode="reduce-overhead")

    optim = torch.optim.AdamW(
        policy.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        fused=torch.cuda.is_available(),
    )

    sampler = GpuSampler()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step_times: list[float] = []
    util_samples: list[float] = []
    nan_seen = False

    try:
        step = 0
        while step < total_steps:
            for batch in loader:
                t_start = time.perf_counter()
                batch = preprocessor(batch)
                loss, _ = policy.forward(batch)
                if not math.isfinite(float(loss.detach().cpu())):
                    nan_seen = True
                    break
                loss.backward()
                optim.step()
                optim.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t_start

                if step >= warmup_steps:
                    step_times.append(dt)
                    s = sampler.sample()
                    if "system/gpu_util_pct" in s:
                        util_samples.append(s["system/gpu_util_pct"])

                if step % 10 == 0:
                    print(f"[bench] step={step:4d}  dt={dt*1000:.1f}ms  "
                          f"util={s.get('system/gpu_util_pct', float('nan')):.0f}%"
                          if step >= warmup_steps else
                          f"[bench] step={step:4d} (warmup)  dt={dt*1000:.1f}ms")

                step += 1
                if step >= total_steps:
                    break
            if nan_seen:
                break
    except torch.cuda.OutOfMemoryError as e:
        record["status"] = "oom"
        record["error"] = str(e)
        print(f"[bench] OOM at batch {args.batch_size}: {e}")
        if torch.cuda.is_available():
            record["max_vram_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        sampler.shutdown()
        sys.exit(0)

    sampler.shutdown()

    if nan_seen:
        record["status"] = "nan"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[bench] NaN/Inf loss at batch {args.batch_size}; wrote {out_path}")
        sys.exit(2)

    if not step_times:
        record["status"] = "no_measurements"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[bench] no measured steps (likely loader exhausted); wrote {out_path}")
        sys.exit(2)

    mean_dt = statistics.mean(step_times)
    stdev_dt = statistics.stdev(step_times) if len(step_times) > 1 else 0.0
    steps_per_s = 1.0 / mean_dt
    samples_per_s = steps_per_s * args.batch_size
    record.update({
        "status": "ok",
        "measured_steps": len(step_times),
        "mean_step_ms": mean_dt * 1000,
        "stdev_step_ms": stdev_dt * 1000,
        "stdev_step_pct": (stdev_dt / mean_dt) * 100 if mean_dt else 0.0,
        "steps_per_s": steps_per_s,
        "samples_per_s": samples_per_s,
        "gpu_util_avg": statistics.mean(util_samples) if util_samples else None,
        "gpu_util_min": min(util_samples) if util_samples else None,
        "gpu_util_max": max(util_samples) if util_samples else None,
        "max_vram_gb": (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else None,
    })
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[bench] OK b={args.batch_size}  {steps_per_s:.3f} steps/s  "
          f"{samples_per_s:.1f} samples/s  vram={record['max_vram_gb']:.1f} GB  "
          f"util={record['gpu_util_avg']!r}  -> {out_path}")


if __name__ == "__main__":
    main()
