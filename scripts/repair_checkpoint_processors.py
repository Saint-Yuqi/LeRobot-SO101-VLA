"""Reconstruct the missing preprocessor/postprocessor for an old overfit
checkpoint that was saved with `policy.save_pretrained()` only.

Rebuilds the same pipelines that `overfit_test.py` builds at training time
(same dataset stats, same policy config) and saves them next to the model,
so the v0.5 migrator and run_inference.py see proper normalization stats.

Usage:
    python scripts/repair_checkpoint_processors.py \\
        --checkpoint <ckpt_dir>          # has config.json + model.safetensors
        --dataset-root <ds_root>         # e.g. test_data
        --base lerobot/smolvla_base
        --chunk-size 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--repo-id", default="local/test_episode")
    ap.add_argument("--base", default="lerobot/smolvla_base")
    ap.add_argument("--chunk-size", type=int, default=50)
    args = ap.parse_args()

    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=str(args.dataset_root))

    policy_cfg = SmolVLAConfig(
        pretrained_path=args.base,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        device="cpu",
        freeze_vision_encoder=False,
    )
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.base,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
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

    preprocessor.save_pretrained(str(args.checkpoint))
    postprocessor.save_pretrained(str(args.checkpoint))
    print(f"wrote pre/post processors into {args.checkpoint}")
    for p in sorted(args.checkpoint.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
