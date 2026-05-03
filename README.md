# VLA Project — SO-101 Pick & Place

Vision-language-conditioned manipulation policy for the SO-101 robot.
Three eval setups: color-conditioned pick & place, compositional instructions,
and celebrity image targeting.

## Status

- [x] Robot calibration verified
- [x] Teleop recording working
- [x] First episode recorded & validated
- [x] Overfit test passes (model memorizes one episode)
- [x] Inference loop runs on robot from checkpoint
- [x] Eval 1 dataset collected — 19 episodes / 5926 frames (merged from 3 sessions)
- [x] Eval 1 model trained — 20 000 steps, run `20260502-174455_job2668259`, avg50 loss ≈ 0.027
- [ ] Eval 1 checkpoint pushed to HuggingFace Hub — **upload failed (401), see below**
- [ ] Eval 2 strategy decided (end-to-end vs decoupled)
- [ ] Eval 3 strategy decided
- [ ] Final eval rehearsal in HG

## Eval 1 — current trained checkpoint

| field | value |
|---|---|
| run dir | `checkpoints/eval1/20260502-174455_job2668259/` |
| final checkpoint | `checkpoints/eval1/20260502-174455_job2668259/final/` |
| intermediate ckpts | `step_2000` … `step_18000` (every 2 000 steps) |
| training config | `configs/train/full_eval1.yaml` |
| dataset | `data/raw/eval1_merged` (`local/eval1_merged`) |
| target HF repo | `PrajnaYang/so101-eval1-smolvla-v1` (public) |
| HF push status | **FAILED — 401 Unauthorized at end of run** |

A `final/` directory contains everything `run_inference.py` needs:
`config.json`, `model.safetensors`, `policy_preprocessor.json`,
`policy_postprocessor.json`, plus the two `*_normalizer_processor.safetensors`
stat tensors. Without the preprocessor/postprocessor pair the policy
outputs nonsense (un-normalized state/action) — never share a partial
copy.

## Verifying training + HF upload yourself

Confirm the training really finished (look for `saved final checkpoint`
near the end of the slurm log — the upload error sits right after it):

```bash
# 1. Last lines of the slurm log for this run
tail -n 30 logs/slurm-2668259.out

# 2. Final checkpoint exists and is complete
ls -lh checkpoints/eval1/20260502-174455_job2668259/final/
```

Confirm whether the HF Hub repo actually has the checkpoint:

```bash
# Important: the HF CLI lives in the `lerobot` conda env.
# `huggingface-cli` is deprecated — the new binary is `hf`.
conda activate lerobot

# Browser (public — no login needed once it's uploaded):
#   https://huggingface.co/PrajnaYang/so101-eval1-smolvla-v1
# Right now you get 404 because the repo was never created (the 401 at
# upload time also blocked `create_repo`).

hf auth whoami        # only needed to confirm you can push — not to read
hf download PrajnaYang/so101-eval1-smolvla-v1 --revision main \
    --include "config.json" --local-dir /tmp/hf_check
# 404 / RepositoryNotFoundError → repo doesn't exist yet, see below.
```

If the repo is missing or empty, re-create + re-upload manually. Note
the `hf upload` argument order is `<repo> <local_path> <path_in_repo>`:

```bash
conda activate lerobot
hf auth login                              # paste a PrajnaYang token with WRITE scope
hf repos create PrajnaYang/so101-eval1-smolvla-v1 --type model   # one-time, public
hf upload PrajnaYang/so101-eval1-smolvla-v1 \
    checkpoints/eval1/20260502-174455_job2668259/final  .  \
    --repo-type=model
```

## Quickstart for teammates (after `git pull`)

**Environment.** Tested with Python **3.12.13** inside the cluster's
`lerobot` conda env. `pyproject.toml` requires `python >= 3.12`. Older
3.11 envs will fail on lerobot v0.5.1 imports. Always work inside the
env so `hf`, `lerobot`, and the pinned torch are on PATH:

```bash
conda activate lerobot
python --version       # should print Python 3.12.13
```

The repo only contains code. The trained checkpoint is **not** in git
(too large) — pull it from one of two places:

```bash
# Option A — directly use the shared cluster path (preferred, no download).
export CKPT=/shares/feldmann.ics.mnf.uzh/Yuqi/Lerobot/checkpoints/eval1/20260502-174455_job2668259/final

# Option B — once HF upload is fixed, pull from the Hub. Repo is public,
#   so no `hf auth login` needed. `hf` lives in the `lerobot` conda env.
conda activate lerobot
hf download PrajnaYang/so101-eval1-smolvla-v1 \
    --local-dir checkpoints/eval1/hf_final
export CKPT=$PWD/checkpoints/eval1/hf_final
```

Then install and run inference:

```bash
# 1. Setup (uses the pinned lerobot v0.5.1 in pyproject.toml)
pip install -e .

# 2. Dry run — prints actions, no robot connected. Sanity check that the
#    checkpoint loads and the preprocessor stats are present.
python scripts/run_inference.py \
    --checkpoint "$CKPT" \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 5 \
    --dry-run

# 3. Real-robot rollout (20 s budget per the eval brief)
python scripts/run_inference.py \
    --checkpoint "$CKPT" \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 20
```

If the dry run errors with a missing `policy_preprocessor.json` or
`*_normalizer_processor.safetensors`, you grabbed an incomplete copy —
re-pull the whole `final/` directory.

## Re-training Eval 1 from scratch

The merged dataset has to exist first (one-time, then committed to the
shared filesystem):

```bash
python scripts/merge_datasets.py \
    --src data/raw/eval1_session1 data/raw/eval1_session2 data/raw/eval1_session3 \
    --dst data/raw/eval1_merged
```

Then launch training (slurm) — output goes under `checkpoints/eval1/<run-id>/`:

```bash
export HF_USER=PrajnaYang                 # so cfg.hf.repo_id expands correctly
sbatch scripts/train.slurm configs/train/full_eval1.yaml scripts/train.py
```

## Repository Layout

- `configs/`  — YAML configs for every experiment. **Never hardcode hyperparams in scripts.**
- `src/`      — All logic. Importable, testable.
- `scripts/`  — Thin entry points. Parse args, call into `src/`.
  - `train.py` — full fine-tune (eval 1+).
  - `overfit_test.py` — single-episode sanity check.
  - `run_inference.py` — closed-loop on the real robot.
  - `merge_datasets.py` — concatenate teleop sessions into one LeRobot dataset.
  - `cast_checkpoint_bf16.py` — halve checkpoint size for sharing.
  - `repair_checkpoint_processors.py` — rebuild missing pre/postprocessor files
    for old checkpoints saved before we started persisting them.
- `data/raw/` — Untouched teleop recordings.
- `data/processed/` — LeRobot-format datasets ready for training.
- `checkpoints/` — All training outputs (gitignored).
- `docs/`    — Decision log, eval setup notes, data collection protocol.

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
