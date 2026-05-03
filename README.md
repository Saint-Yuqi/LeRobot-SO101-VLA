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
- [ ] Eval 1 checkpoint pushed to HuggingFace Hub (see "Maintainer notes")
- [ ] Eval 2 strategy decided (end-to-end vs decoupled)
- [ ] Eval 3 strategy decided
- [ ] Final eval rehearsal in HG

## Quickstart (after `git pull`)

**Environment.** Tested with **Python 3.12.13** inside the cluster's
`lerobot` conda env. `pyproject.toml` requires `python >= 3.12`. Older
3.11 envs will fail on lerobot v0.5.1 imports. Always work inside the
env so `hf`, `lerobot`, and the pinned torch are on `PATH`.

```bash
conda activate lerobot
python --version          # should print Python 3.12.13
pip install -e .          # one-time
```

Run inference. The trained checkpoint is **not** in git (too large):
`scripts/run_inference.py` pulls it for you. `--checkpoint` accepts
either a local directory or a HuggingFace repo id, and auto-downloads
the latter into the HF cache the first time it's used.

```bash
# Dry run (no robot connected) — sanity-check that the checkpoint loads
# and the policy emits actions. Useful before plugging in the SO-101.
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

On the shared cluster you can skip the HF download entirely by pointing
at the run dir directly:

```bash
python scripts/run_inference.py \
    --checkpoint /shares/feldmann.ics.mnf.uzh/Yuqi/Lerobot/checkpoints/eval1/20260502-174455_job2668259/final \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 20
```

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
  image and the robot does nonsense. Re-pull the whole `final/` folder.
- **HF download is slow / runs out of disk** — set `HF_HOME` to a path
  on scratch (this is what `train.slurm` does:
  `export HF_HOME=/home/yuqyan/scratch/Lerobot/hf_cache`).
- **`hf` not found** — you're not in the `lerobot` conda env.

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

## Maintainer notes — pushing the checkpoint to HuggingFace

The first training run finished cleanly but the final HF upload failed
with a 401 (expired token). The local checkpoint is intact; the repo on
the Hub does not yet exist. To finish the upload (only the maintainer
with the `PrajnaYang` HF account can do this):

```bash
conda activate lerobot
hf auth login                                   # paste a WRITE-scope token
hf repos create PrajnaYang/so101-eval1-smolvla-v1 --type model    # public, one-time
hf upload PrajnaYang/so101-eval1-smolvla-v1 \
    checkpoints/eval1/20260502-174455_job2668259/final  .  \
    --repo-type=model
```

Note `hf upload` argument order: `<repo> <local_path> <path_in_repo>`.
The trailing `.` puts the 6 files at the repo root. After this, the
quickstart commands above work for everyone with no HF login.

To verify a future run finished cleanly:

```bash
tail -n 30 logs/slurm-<JOBID>.out                # look for "saved final checkpoint"
ls -lh checkpoints/eval1/<run-id>/final/         # 6 files, ~1.2 GB
hf download PrajnaYang/so101-eval1-smolvla-v1 \
    --include "config.json" --local-dir /tmp/hf_check    # 404 → not uploaded yet
```

## Repository layout

- `configs/` — YAML configs for every experiment. **Never hardcode
  hyperparams in scripts.**
- `src/` — All logic. Importable, testable.
  - `models/base_vla.py` — the `BaseVLA` interface every policy implements.
  - `models/smolvla_wrapper.py` — end-to-end SmolVLA implementation.
- `scripts/` — Thin entry points. Parse args, call into `src/`.
  - `train.py` — full fine-tune (eval 1+).
  - `overfit_test.py` — single-episode sanity check.
  - `run_inference.py` — closed-loop on the real robot.
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
