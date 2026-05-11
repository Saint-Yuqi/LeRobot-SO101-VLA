# FlowerVLA feasibility spike

**Branch:** `spike/flowervla-feasibility`
**Date:** 2026-05-11
**Status:** PROVISIONAL-GO (batch adaptation verified; model forward not yet executed)

## Verdict

**SPIKE_VERDICT: PROVISIONAL-GO**

Batch-shape adaptation between `LeRobotDataset` and FlowerVLA-CALVIN's
expected dict format is straightforward — all six concrete gaps are
addressable with config knobs and a thin batch-shaping wrapper.

The remaining unknown is whether the resulting forward actually produces
finite loss + non-zero grad — this requires installing flower_calvin's
deps into the lerobot env, which is the gating step for upgrading
PROVISIONAL-GO → GO. The Phase 2 plan should not begin until that runs.

## Upstream choice: flower_calvin (not flower_pret)

| Aspect | flower_pret (OXE) | flower_calvin (CALVIN/LIBERO) | Pick |
|--------|-------------------|-------------------------------|------|
| Purpose | pretraining on OXE-RLDS | finetune on a single embodiment | flower_calvin (closer to what we want) |
| Framework | plain `nn.Module` + `accelerate` | PyTorch Lightning | flower_calvin (cleaner forward) |
| Language input | pre-tokenized `input_ids` / `attention_mask` | raw `lang_text: List[str]` (tokenized inside `construct_prompts`) | flower_calvin (lerobot gives us raw strings) |
| Multi-action-space | `ActionIndex` for OXE diversity | single action space | flower_calvin (simpler for SO-101) |
| VLM default | Florence-2-large | Florence-2-base | flower_calvin (smaller; less VRAM) |
| Cameras | primary + optional wrist | primary `rgb_static` + optional `rgb_gripper` | tied (we use primary only) |

## Pre-spike sanity (verified)

- `lerobot==0.5.1` installed; `from lerobot.optim.optimizers import AdamConfig` imports cleanly.
- All three target HF datasets exist and are accessible:
  - `ethrl2026/task1_20260509_prompt_lighting_augmented_360` (private, 360 eps, 165828 frames, natural-language prompts)
  - `ethrl2026/task2_20260509_stage2_random_lighting_augmented_2160` (private, 2160 eps, 870828 frames, natural-language prompts)
  - `ethrl2026/so101_pickup_20260509_185350_task3` (private, 45 eps, 18959 frames, **encoded labels** `Y+L`, `T+M`, etc.)
- HF whoami: `PrajnaYang`, member of `ethrl2026` org → can push to the eval{1,2,3}-flower-v1 repos.
- **Hub-tag caveat:** none of the new datasets have a `v3.0` git tag, which
  `LeRobotDataset.__init__` calls `get_safe_version` for. Workaround:
  `snapshot_download(..., revision="main", local_dir=$HF_LEROBOT_HOME/<repo>)`,
  then `rm -rf <local_dir>/.cache` (lerobot detects the snapshot's marker
  dir and forces a re-fetch otherwise). All three datasets now load cleanly
  from cache. A proper fix is to ask the dataset owner to run
  `HfApi().create_tag(repo, tag="v3.0", repo_type="dataset")`.

## Batch-shape gaps and adapter contract

Spike output (`scripts/spikes/flower.py`) confirms the following with concrete numbers:

| # | Aspect | LeRobotDataset (SO-101) | FlowerVLA-CALVIN expects | Adapter fix |
|---|--------|-------------------------|--------------------------|-------------|
| G1 | image | `observation.images.main` shape `(B, 3, 480, 640)` float32 in [0,1] | `batch["rgb_obs"]["rgb_static"]` shape `(B, T, 3, H, W)` | unsqueeze T=1; Florence-2's `_encode_image` does its own internal resize/normalize |
| G2 | action_dim | `action` shape `(B, 50, 6)` (6-DoF joints) | default 7 (CALVIN delta-EE) | set `FlowerVLAConfig.action_dim=6` |
| G3 | chunk_size | 50 | default `act_window_size=10` | set `act_window_size=50`; verify DiT positional buffers grow |
| G4 | proprio | `observation.state` shape `(B, 6)` (joint positions, deg) | `batch[obs_modalities]["proprio"]` if `use_proprio=True`; default `lowdim_obs_dim=7` | set `lowdim_obs_dim=6`, `use_proprio=True`; pack under `obs_modalities` (CALVIN config uses `"state_obs"`) |
| G5 | language | `task` = `List[str]`, raw natural language (or encoded `'Y+L'` on task3) | `batch["lang_text"]` = `List[str]`, tokenized inside `construct_prompts` via Florence-2 processor | pass through; no work needed |
| G6 | action norm | lerobot's `NormalizerProcessor` normalizes before policy, unnorms after | flow-matching expects targets in a stable range (usually [-1,1] or [0,1]) | run lerobot's normalizer; verify the resulting range is in the band flow-matching trains on (decide during forward-pass spike) |

`observation_delta_indices` value for the FlowerVLAConfig (Phase 2.2):
**None** (single-frame obs, like SmolVLA). FlowerVLA-CALVIN treats `T` as
1 unless `use_second_view=True` with separate temporal windows — for SO-101's
single `main` camera and our current setup, `T=1` is correct.

## Concrete `FlowerVLAConfig` initial knobs (filled from spike, no guessing)

```python
@PreTrainedConfig.register_subclass("flowervla")
@dataclass
class FlowerVLAConfig(PreTrainedConfig):
    pretrained_path: str | None = None       # upstream HF id for a pretrained ckpt (TBD)
    chunk_size: int = 50
    n_action_steps: int = 50
    n_obs_steps: int = 1                     # CALVIN default; single-frame
    device: str = "cuda"
    use_amp: bool = False

    # FlowerVLA-specific (verified from /tmp/flower_calvin/flower/models/flower.py):
    vlm_path: str = "microsoft/Florence-2-base"
    action_dim: int = 6                      # SO-101
    lowdim_obs_dim: int = 6                  # SO-101 joint state
    act_window_size: int = 50                # matches chunk_size
    use_second_view: bool = False            # SO-101 single camera
    use_proprio: bool = True                 # we have observation.state
    freeze_florence: bool = False            # spike default; consider True for cheap finetune
    freeze_vision_tower: bool = False

    num_sampling_steps: int = 4              # FlowerVLA published default for inference
    sampling_type: str = "ln"                # rectified-flow default

    def get_optimizer_preset(self):
        # lerobot.optim.optimizers.AdamConfig imports OK in v0.5.1.
        # Either return AdamConfig(lr=1e-4) or None (train.py builds AdamW directly).
        return None

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        required = {"observation.state", "action"}
        present = set(self.input_features) | set(self.output_features)
        missing = required - present
        if missing:
            raise ValueError(f"FlowerVLAConfig: missing features {missing}")

    @property
    def observation_delta_indices(self):
        return None   # single-frame obs; matches SmolVLA, matches FlowerVLA T=1

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None
```

## What's NOT yet done (Phase 0 punch list)

1. **Run actual model forward.** Requires installing `/tmp/flower_calvin/requirements.txt`
   (note: it pins `torch==2.2.2`, `pytorch-lightning==2.0.8`, `hydra-core==1.1.1`).
   Likely conflicts with lerobot's current torch (verify before installing).
   Once installed:
   ```
   /shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/lerobot/bin/python \
       scripts/spikes/flower.py --instantiate
   ```
   That will print `loss=<float>` + grad-norm. **Only then** does the
   verdict become a full GO and Phase 2 starts.

2. **Decide on pretrained init.** FlowerVLA's value vs SmolVLA depends on
   the pretrained checkpoint being a strong prior. Need to point
   `pretrained_path` at one of:
   - the OXE pretrain (from `flower_vla_pret`)
   - a CALVIN-finetuned ckpt (from `flower_vla_calvin`)
   - random init (likely much worse — only worth it as a control)
   Browse https://huggingface.co/intuitive-robots for an actual HF weights repo.

3. **Action normalization sanity.** SO-101 actions are joint angles in degrees,
   range roughly ±90°. lerobot's per-dim normalizer maps them to a learned range
   from dataset stats. FlowerVLA's `rf_loss` interpolates between targets and
   `randn_like` noise (mean 0, std 1) — targets that aren't roughly unit-scale
   will train poorly. Confirm at first forward.

## Files this spike leaves on the branch

- `scripts/spikes/flower.py` — the throwaway batch-shape script (to be deleted in the Phase 2 landing commit, explicit in the commit message)
- `docs/flowervla_spike.md` — this file (kept; basis for Phase 2 adapter code)

`third_party/flower_vla/` (the submodule) is added in Phase 2.1, not here —
the spike clones to `/tmp/` to avoid polluting the working tree until we
commit to integration.
