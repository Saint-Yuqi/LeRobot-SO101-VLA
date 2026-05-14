# VLA Project — SO-101 Pick & Place

Vision-language-conditioned manipulation policy for the SO-101 robot.
Three eval setups: color-conditioned pick & place, compositional
instructions, and celebrity image targeting.

## Status

- [x] Robot calibration verified
- [x] Teleop recording working
- [x] Overfit test passes (model memorizes one episode)
- [x] Inference loop runs on robot from checkpoint
- [x] Eval 1 dataset collected — 19 episodes / 5 926 frames (3 sessions merged)
- [x] Eval 1 model trained — 20 000 steps, run `20260502-174455_job2668259`, avg-50 loss ≈ 0.027
- [x] **Phase-weighted sampling** — pre-grasp frames upweighted at train time (see below)
- [ ] Eval 1 checkpoint pushed to HuggingFace Hub
- [ ] Eval 2 strategy decided (end-to-end vs decoupled)
- [ ] Eval 3 strategy decided
- [ ] Final eval rehearsal in HG

## Quickstart (after `git pull`)

**Environment.** Requires **Python 3.12+** with lerobot v0.5.1 installed
(pinned in `pyproject.toml`). Older 3.11 envs will fail on lerobot
imports. Set up an env however you prefer (conda, uv, venv) and install
the project:

```bash
# example with conda
conda create -n lerobot python=3.12
conda activate lerobot
pip install -e .          # pulls lerobot[smolvla] @ v0.5.1 + the rest
python --version          # should print Python 3.12.x
```

### Run inference (downloads the checkpoint from HuggingFace)

The trained checkpoint is **not** in git (too large, ~1.2 GB).
`scripts/run_inference.py` pulls it for you the first time you run —
just pass the public HuggingFace repo id as `--checkpoint`. Subsequent
runs hit the local cache.

```bash
# Dry run (no robot connected) — sanity-check that the checkpoint
# downloads, the policy loads, and actions come out.
python scripts/run_inference.py \
    --checkpoint PrajnaYang/so101-eval1-smolvla-v1 \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 5 \
    --dry-run

# Real-robot rollout (20 s budget per the eval brief).
python scripts/run_inference.py \
    --checkpoint PrajnaYang/so101-eval1-smolvla-v1 \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 20
```

The first run prints `[infer] cached at <path>` so you can see exactly
where the ~1.2 GB landed. By default that's `~/.cache/huggingface/hub/`.
Set `HF_HOME=<somewhere>` before running if you need it elsewhere.

### Common flags

| flag | default | what it does |
|---|---|---|
| `--checkpoint` | (required) | Local checkpoint dir, or `<user>/<repo>` HF id. |
| `--prompt` | (required) | Natural-language task. Wrap in quotes. |
| `--max-seconds` | `20.0` | Hard time limit on the rollout. |
| `--camera-key` | `main` | Camera name. Eval 1 dataset uses `main`; only change if you re-record with a different camera. |
| `--control-hz` | `30.0` | Robot control rate. |
| `--dry-run` | off | Don't connect to the robot, just print actions. |
| `--policy-type` | `smolvla` | `smolvla` (current) or `decoupled` (future). |

### Troubleshooting

- **"Failed to load preprocessor/postprocessor"** — your checkpoint
  folder is missing `policy_preprocessor.json` /
  `policy_postprocessor.json` or their `*_normalizer_processor.safetensors`
  stat tensors. Without them the policy sees un-normalized state +
  image and the robot does nonsense. Delete the cache and re-run, or
  re-pull the whole `final/` folder.
- **HF download is slow / fills up disk** — the cache lives at
  `~/.cache/huggingface/hub/` by default. `export HF_HOME=<bigger-disk>`
  before running to redirect it.
- **`pip install -e .` fails on `lerobot`** — make sure you're on
  Python 3.12+ and have `git` available (lerobot is pulled from a
  GitHub tag, not PyPI).

## Eval 1 — current trained checkpoint

| field | value |
|---|---|
| run dir | `checkpoints/eval1/20260502-174455_job2668259/` |
| final checkpoint | `…/final/` (6 files, ~1.2 GB) |
| intermediate ckpts | `step_2000` … `step_18000` (every 2 000 steps) |
| training config | `configs/train/full_eval1.yaml` |
| dataset | `data/raw/eval1_merged` (`local/eval1_merged`) |
| HF repo (public, after upload) | `PrajnaYang/so101-eval1-smolvla-v1` |

A `final/` directory contains everything `run_inference.py` needs:
`config.json`, `model.safetensors`, `policy_preprocessor.json`,
`policy_postprocessor.json`, plus the two `*_normalizer_processor.safetensors`
stat tensors. Never share a partial copy — without the processors the
policy outputs un-normalized actions.

## Phase-weighted sampling (pre-grasp upweighting)

Real-robot rollouts on the three SO-101 pickup tasks consistently fail in the
**pre-grasp segment** — the model approaches the bowl but the moment of
contact / first successful gripper closure is unreliable. Once the object is
in the gripper, lift / place / release usually succeeds.

[scripts/train.py](scripts/train.py) now supports a `WeightedRandomSampler`
driven by a binary per-frame label (pre_grasp / post_grasp) computed from
the gripper signal in the action column. Single config knob, no
loss-function edits.

```yaml
# configs/train/full_eval*.yaml
train:
  phase_sampling:
    enabled: true              # ← the only switch
    weight_pregrasp: 2.0       # 1.0 = uniform-with-replacement (still NOT bit-equivalent
                               # to today's shuffle=True — sampling is with replacement)
    open_frac: 0.6             # per-episode adaptive: g_min + 0.6 * (g_max - g_min)
    close_frac: 0.4            # per-episode adaptive: g_min + 0.4 * (g_max - g_min)
    min_amplitude: 5.0         # skip episodes whose gripper barely moves (raw units)
    post_close_margin: 3       # frames after first close to confirm stability (~100ms @30fps)
```

**Default in all 3 production configs: `enabled: true, weight_pregrasp: 2.0`.**
Setting `enabled: false` reverts to uniform shuffle (bit-identical to before
this change, modulo `replacement=True` on the with-replacement control case).

Implementation:
- [src/data/phase_labels.py](src/data/phase_labels.py) — per-episode adaptive
  open→close detector + NPZ cache + `PhaseLabelResult` dataclass with
  alignment fields.
- [src/data/sampler.py](src/data/sampler.py) — `make_phase_weighted_sampler()`
  factory, `concat_phase_labels()` for ConcatDataset multi-source, runtime
  `assert_dataset_alignment()` / `assert_concat_alignment()` that catch any
  iteration-order divergence at startup.

Detector defaults were validated on all three task datasets in
[scripts/spikes/probe_phase_detector.py](scripts/spikes/probe_phase_detector.py).
Absolute gripper thresholds break on eval3 (78% miss); per-episode adaptive
thresholds work on all three (eval1 0% / eval2 0% / eval3 11% failed-close,
with `pregrasp_frac` ≈ 0.63–0.66 across tasks).

Per-phase MAE breakdown is also reported by
[scripts/eval_offline.py](scripts/eval_offline.py):
`mae_pregrasp_anchor`, `mae_postgrasp_anchor`, and per-joint variants.
This is the metric to track A/B against — does upweighting pre-grasp narrow
`mae_pregrasp_anchor` without inflating `mae_postgrasp_anchor`?

## Re-training Eval 1 from scratch

The merged dataset has to exist first (one-time, then sits on the
shared filesystem):

```bash
python scripts/merge_datasets.py \
    --src data/raw/eval1_session1 data/raw/eval1_session2 data/raw/eval1_session3 \
    --dst data/raw/eval1_merged
```

Then launch training via slurm. Output goes under
`checkpoints/eval1/<run-id>/`:

```bash
export HF_USER=PrajnaYang     # so cfg.hf.repo_id expands correctly
sbatch scripts/train.slurm configs/train/full_eval1.yaml scripts/train.py
```

## pi0 fine-tune (task1 / task2)

Two recipes for the `lerobot/pi0` base (PaliGemma VLM + Gemma 300M action expert):

- **task1** — full fine-tune of the action expert only (PaliGemma frozen).
  Config: `configs/train/pi0_task1.yaml`. Saves full policy weights.
- **task2** — openpi-aligned LoRA on both LLMs (PaliGemma + action expert),
  SigLIP frozen, projection MLPs full-FT via `modules_to_save`.
  Config: `configs/train/pi0_task2.yaml`. Saves a PEFT adapter only.

### Extra install (pi0)

`peft` is **not** pulled by `lerobot[smolvla]`. Install it explicitly before
training task2 or running inference on a task2 checkpoint:

```bash
pip install "peft>=0.18,<1.0"
# or, equivalently, pull lerobot's peft extra:
# pip install "lerobot[peft] @ git+https://github.com/huggingface/lerobot.git@v0.5.1"
```

Task1 needs nothing beyond the current `pip install -e .` — `lerobot[smolvla]`
already pulls a transformers new enough for PaliGemma / pi0.

### Train

```bash
sbatch scripts/train.slurm configs/train/pi0_task1.yaml scripts/train.py
sbatch scripts/train.slurm configs/train/pi0_task2.yaml scripts/train.py
```

`scripts/train.py` dispatches on `model.family` / `model.base`: pi0 → `PI0Config`,
otherwise → `SmolVLAConfig`. When `model.use_peft: true` (task2) the trainer
wraps the policy with a fresh LoRA adapter after `make_policy` (openpi recipe;
see comment in `train.py` near `wrap_with_peft`).

### Inference — `scripts/run_inference_real.py`

`run_inference_real.py` is the closed-loop runner for the modern lerobot API
(`lerobot.robots.so101_follower`). It works for **SmolVLA, pi0 task1, and pi0
task2** — the script auto-detects the checkpoint layout:

- If the checkpoint dir contains `adapter_config.json` → load `lerobot/pi0`
  base, then attach the LoRA adapter (task2 path).
- Otherwise → load directly via `policy_cls.from_pretrained` (SmolVLA + pi0
  task1 path; behaviour unchanged from before).

```bash
# SmolVLA eval1 (unchanged behaviour)
python scripts/run_inference_real.py \
    --checkpoint ethrl2026/so101-eval1-smolvla-v1 \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 20

# pi0 task1 — full action-expert fine-tune (loads like a normal checkpoint)
python scripts/run_inference_real.py \
    --checkpoint ethrl2026/so101-pi0-task1 \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 20

# pi0 task2 — LoRA adapter (base + adapter merged at load time)
python scripts/run_inference_real.py \
    --checkpoint ethrl2026/so101-eval2-pi0-lora-v1 \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 20

# Dry run (no robot) — useful for verifying a checkpoint downloads + loads.
python scripts/run_inference_real.py \
    --checkpoint ethrl2026/so101-eval2-pi0-lora-v1 \
    --prompt "..." --max-seconds 5 --dry-run
```

Flags worth knowing (full list: `--help`):

| flag | default | what it does |
|---|---|---|
| `--checkpoint` | required | Local ckpt dir, HF repo id, or PEFT adapter dir. |
| `--prompt` | required | Task instruction. |
| `--max-seconds` | `20.0` | Hard rollout time cap. |
| `--device` | `mps` | `cuda` for the cluster, `mps` for Apple, `cpu` fallback. |
| `--control-hz` | `30.0` | Control loop rate. |
| `--robot-port` / `--robot-id` | — | SO-101 serial port + calibration id. |
| `--camera-key` / `--camera-index` | `main` / `0` | OpenCV camera selection. |
| `--dry-run` | off | Synthetic robot — no hardware needed, prints actions. |
| `--display-data` | off | Stream obs + actions to a rerun viewer. |
| `--log-dir` / `--no-log` | `logs/inference` | Per-rollout dossier (meta.json, steps.csv, optional frames/). |

Each rollout writes `logs/inference/<timestamp>_<ckpt>/` with `meta.json`,
`steps.csv`, `outcome.json`, and (every `--frame-every` steps) JPEGs under
`frames/` — enough to debug a failed rollout after the fact.

## Repository layout

- `configs/` — YAML configs for every experiment. **Never hardcode
  hyperparams in scripts.**
- `src/` — All logic. Importable, testable.
  - `models/base_vla.py` — the `BaseVLA` interface every policy implements.
  - `models/smolvla_wrapper.py` — end-to-end SmolVLA implementation.
  - `data/phase_labels.py` + `data/sampler.py` — phase-weighted sampling
    (pre-grasp upweighting, used by `train.py`).
- `scripts/` — Thin entry points. Parse args, call into `src/`.
  - `train.py` — full fine-tune (eval 1+).
  - `overfit_test.py` — single-episode sanity check.
  - `run_inference.py` — closed-loop on the real robot.
  - `eval_offline.py` — offline action-MAE eval (also reports per-phase
    MAE: `mae_pregrasp_anchor`, `mae_postgrasp_anchor`).
  - `merge_datasets.py` — concatenate teleop sessions into one LeRobot dataset.
  - `cast_checkpoint_bf16.py` — halve checkpoint size for sharing.
  - `repair_checkpoint_processors.py` — rebuild missing pre/postprocessor
    files for old checkpoints saved before we started persisting them.
- `data/raw/` — Untouched teleop recordings (gitignored).
- `data/processed/` — LeRobot-format datasets ready for training (gitignored).
- `checkpoints/` — All training outputs (gitignored).
- `logs/` — Slurm + training logs (gitignored).
- `docs/` — Decision log, eval setup notes, data collection protocol.

## Key design decision

`src/models/base_vla.py` defines a `BaseVLA` interface with a single
`predict(images, prompt, state) -> action_chunk` method. Both the
end-to-end SmolVLA wrapper and the decoupled (VLM + policy) approach
implement this interface. **Inference, eval, and the robot runner
never need to know which is in use.** This lets the team try both
strategies in parallel for Eval 2/3 without forking the codebase.

## Per-eval model selection

The brief allows different checkpoints/models per eval. We use:

- **Eval 1** — SmolVLA fine-tuned end-to-end (target: smallest possible for bonus pts).
- **Eval 2** — TBD with team. Likely end-to-end with rich teleop prompts, or decoupled.
- **Eval 3** — TBD with team. Decoupled likely required (small VLA can't recognize OOD celebrities zero-shot).

See `docs/decisions.md` for the full reasoning.

## Cluster shortcut (maintainer only)

When running on the UZH cluster where the training itself happened,
skip the HuggingFace download and point `--checkpoint` at the shared
run dir directly — same files, no network, no quota usage:

```bash
python scripts/run_inference.py \
    --checkpoint /shares/feldmann.ics.mnf.uzh/Yuqi/Lerobot/checkpoints/eval1/20260502-174455_job2668259/final \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 20
```
