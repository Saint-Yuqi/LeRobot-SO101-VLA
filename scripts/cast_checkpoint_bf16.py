"""Cast a saved SmolVLA checkpoint down to all-BF16 to halve on-disk size
for sharing. Loss is negligible for inference / overfit replay.

Usage:
    python scripts/cast_checkpoint_bf16.py <src_dir> <dst_dir>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    state = load_file(str(args.src / "model.safetensors"))
    cast = {}
    n_cast = n_kept = 0
    for k, v in state.items():
        if v.dtype in (torch.float32, torch.float16):
            cast[k] = v.to(torch.bfloat16)
            n_cast += 1
        else:
            cast[k] = v
            n_kept += 1
    print(f"cast {n_cast} tensors -> bf16, kept {n_kept} as-is")

    save_file(cast, str(args.dst / "model.safetensors"))

    # Copy every sibling file (config.json, policy_preprocessor.json,
    # policy_postprocessor.json, and the *_normalizer_processor.safetensors
    # stat tensors). Without these the colleague gets un-normalized state/
    # action and the policy outputs nonsense — exact bug we fixed last round.
    for src_file in args.src.iterdir():
        if src_file.name == "model.safetensors":
            continue
        if src_file.is_file():
            shutil.copy(src_file, args.dst / src_file.name)
            print(f"  copied: {src_file.name}")

    src_size = (args.src / "model.safetensors").stat().st_size
    dst_size = (args.dst / "model.safetensors").stat().st_size
    print(f"size: {src_size/1e9:.2f} GB -> {dst_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
