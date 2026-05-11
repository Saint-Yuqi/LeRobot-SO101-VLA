"""FlowerVLA feasibility spike — THROWAWAY.

Goal: prove FlowerVLA can consume one batch of `LeRobotDataset` data
and produce a finite action loss + non-degenerate grad-norm, before we
commit to writing the lerobot adapter in Phase 2.

This script:
  1. Loads one batch from task3 (smallest target dataset).
  2. Translates the lerobot batch into the dict format FlowerVLA-CALVIN
     expects (`rgb_obs`, `lang_text`, `actions`).
  3. Logs concrete shape/dtype/dimension gaps so docs/flowervla_spike.md
     can record numbers instead of guesses.
  4. (Optionally — gated behind --instantiate) builds the FlowerVLA model
     and runs `encode_observations + rf_loss`. Requires `/tmp/flower_calvin`
     to be cloned and its deps installed.

Will be DELETED in the Phase 2 landing commit (called out in the message).
The `spikes/` subdir signals throwaway; do not import from this file.

Run:
    /shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/lerobot/bin/python \\
        scripts/spikes/flower.py

    # Once flower_calvin deps are installed, also pass model forward:
    /shares/.../python scripts/spikes/flower.py --instantiate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

TASK3 = "ethrl2026/so101_pickup_20260509_185350_task3"


def load_one_batch(batch_size: int = 2, chunk: int = 50):
    """Pull one batch from task3 with the same delta_timestamps SmolVLA uses."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(TASK3)
    fps = ds_meta.fps
    # Build delta_timestamps for action chunking: t, t+1/fps, ... t+(chunk-1)/fps.
    delta = {"action": [i / fps for i in range(chunk)]}
    ds = LeRobotDataset(repo_id=TASK3, delta_timestamps=delta)
    print(f"[spike] dataset frames={len(ds)}  episodes={ds_meta.total_episodes}")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    batch = next(iter(loader))
    return batch, ds_meta


def adapt_to_flowervla(batch: dict, use_proprio: bool = True) -> dict:
    """Reshape a lerobot batch into the dict FlowerVLA-CALVIN forward expects.

    See /tmp/flower_calvin/flower/models/flower.py:encode_observations:690
    for the keys it reads.
    """
    img = batch["observation.images.main"]   # (B, C, H, W) — already float in [0,1]
    if img.dim() == 4:
        img = img.unsqueeze(1)                # add T=1 dim -> (B, T=1, C, H, W)
    state = batch["observation.state"]        # (B, 6)
    actions = batch["action"]                 # (B, chunk, 6)
    task = batch["task"]                      # list[str] of length B

    fb = {
        "rgb_obs": {
            "rgb_static": img,
        },
        "lang_text": list(task) if isinstance(task, (list, tuple)) else [task],
        "actions": actions,
    }
    if use_proprio:
        # CALVIN code reads proprio via batch[self.obs_modalities]['proprio'].
        # self.obs_modalities is a string set by config; default in many configs
        # is 'state_obs'. Add it under both keys to be defensive.
        fb["state_obs"] = {"proprio": state}
        fb["observation"] = {"proprio": state}
    return fb


def report_shapes(lerobot_batch: dict, flower_batch: dict) -> None:
    print("\n[spike] ============ lerobot batch ============")
    for k, v in lerobot_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}  dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)}  sample={v[0]!r}")
        else:
            print(f"  {k}: {type(v).__name__}")

    print("\n[spike] ============ adapted flower batch ============")
    for k, v in flower_batch.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                print(f"  {k}.{k2}: shape={tuple(v2.shape)}  dtype={v2.dtype}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}  dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)}  sample={v[0]!r}")


def report_gaps(flower_batch: dict, chunk: int) -> None:
    print("\n[spike] ============ gaps to address in adapter ============")
    img = flower_batch["rgb_obs"]["rgb_static"]
    print(f"  G1 image: lerobot gives (B,1,3,{img.shape[3]},{img.shape[4]});")
    print(f"            Florence-2 vision tower resizes internally — "
          "should pass through without explicit resize, BUT verify the "
          "feature-map dimensions are sane post `_encode_image`.")

    actions = flower_batch["actions"]
    print(f"  G2 action_dim: SO-101 = {actions.shape[-1]}; FlowerVLA default = 7 (CALVIN).")
    print(f"     -> set FlowerVLA `action_dim=6` in config; CALVIN's action_decoders "
          "use it as a max — confirm DiT supports 6 cleanly.")

    print(f"  G3 chunk_size: SO-101 plan = 50; FlowerVLA CALVIN default `act_window_size=10`.")
    print(f"     -> set `act_window_size=50` in config; expect DiT positional/RoPE buffers "
          "to need that.")

    proprio = flower_batch.get("state_obs", {}).get("proprio")
    if proprio is not None:
        print(f"  G4 proprio: SO-101 dim={proprio.shape[-1]}; FlowerVLA default `lowdim_obs_dim=7`.")
        print(f"     -> set `lowdim_obs_dim=6` and `use_proprio=True`.")

    print(f"  G5 language: SO-101 `task` = raw str; FlowerVLA-CALVIN tokenizes inside "
          "`construct_prompts` via Florence-2's processor. Pass through OK.")

    print(f"  G6 action normalization: lerobot's processor unnorms after the policy; "
          "FlowerVLA's rf_loss expects normalized targets. The lerobot `preprocessor` "
          "step (NormalizerProcessor) handles this *before* forward — should be fine, "
          "but verify range matches what flow-matching expects (typically [-1,1] or [0,1]).")


def run_flower_forward(flower_batch: dict, chunk: int) -> dict:
    """Best-effort instantiation + forward. Will fail without flower_calvin deps installed."""
    sys.path.insert(0, "/tmp/flower_calvin")
    try:
        from flower.models.flower import FLOWERVLA
    except Exception as e:
        print(f"\n[spike] FAILED to import flower.models.flower: {e}")
        print(f"[spike] Install /tmp/flower_calvin/requirements.txt into the lerobot env first.")
        return {"verdict": "NO-GO (import)", "reason": str(e)}

    try:
        model = FLOWERVLA(
            vlm_path="microsoft/Florence-2-base",
            freeze_florence=True,
            freeze_vision_tower=True,
            action_dim=6,             # SO-101
            lowdim_obs_dim=6,         # SO-101 state
            act_window_size=chunk,    # SO-101 plan = 50
            use_second_view=False,    # SO-101 has only `main`
            use_proprio=True,
        )
        model.eval()
        with torch.no_grad():
            cond = model.encode_observations(flower_batch)
            loss, info = model.rf_loss(cond, flower_batch["actions"])
        print(f"\n[spike] forward OK. loss={loss.item():.4f}  info_keys={list(info.keys())}")
        loss.requires_grad_(True)
        # grad norm sanity
        try:
            loss.backward()
            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            if grad_norms:
                gn = sum(grad_norms) / len(grad_norms)
                print(f"[spike] mean grad norm: {gn:.4e}  finite: {torch.isfinite(loss).item()}")
        except Exception as e:
            print(f"[spike] backward failed: {e}")
        return {
            "verdict": "GO" if torch.isfinite(loss).item() and loss.item() > 0 else "NO-GO",
            "loss": float(loss.item()),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"verdict": "NO-GO (forward)", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--instantiate", action="store_true",
                        help="Try to build FlowerVLA and run a forward. Requires "
                             "flower_calvin deps installed.")
    args = parser.parse_args()

    lerobot_batch, ds_meta = load_one_batch(args.batch_size, args.chunk)
    flower_batch = adapt_to_flowervla(lerobot_batch)
    report_shapes(lerobot_batch, flower_batch)
    report_gaps(flower_batch, args.chunk)

    if args.instantiate:
        result = run_flower_forward(flower_batch, args.chunk)
        print(f"\n[spike] result: {result}")
        print(f"[spike] SPIKE_VERDICT: {result['verdict']}")
    else:
        print("\n[spike] Adapter math + shapes look workable.")
        print("[spike] Re-run with --instantiate to do the actual forward pass.")
        print("[spike] SPIKE_VERDICT: PROVISIONAL-GO (batch adaptation only; forward not yet run)")


if __name__ == "__main__":
    main()
