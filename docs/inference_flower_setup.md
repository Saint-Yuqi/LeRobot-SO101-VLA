# Inferring FlowerVLA checkpoints from this repo

This repo (SmolVLA, Python 3.12, `lerobot` conda env) **cannot** load
FlowerVLA checkpoints directly. FlowerVLA is a separate stack — different
backbone (Florence-2), different Python, different `torch`, different
`transformers`. The two are wired up to live in sibling repos:

- **SmolVLA** (this repo): `Lerobot/` + `conda activate lerobot` (Python 3.12)
- **FlowerVLA**: `Lerobot_flower/` + `conda activate flower` (Python 3.10)

The new `scripts/run_inference_real_flower.py` script in this repo lets you
launch a FlowerVLA rollout with the same CLI shape as
`scripts/run_inference_real.py`, but it has to be run **inside the `flower`
conda env** and it imports the policy code from the sibling
`Lerobot_flower` checkout.

## Why the `lerobot` env can't run FlowerVLA

| Axis | `lerobot` env | `flower` env | Conflict |
|---|---|---|---|
| Python | 3.12 | 3.10 | Some FlowerVLA pinned deps have no 3.12 wheels. |
| `torch` | 2.10.0 + cu128 | 2.2.2 + cu121 | `torchdiffeq` / `torchsde` flow-matching solvers behave differently across this gap. |
| `transformers` | ≥4.50 (modern, pulled by `lerobot[smolvla]`) | 4.46.3 | **Hard blocker.** Florence-2 has an upstream `forced_bos_token_id` bug on transformers ≥5 that we cannot patch. |
| Framework | `lerobot[smolvla]` 0.5.x | vendored `third_party/flower_vla/` + `pytorch-lightning 2.0.8` + `hydra 1.1.1` | Two independent config + data pipelines; installing both into one env would clobber each other. |

Net: trying to bolt FlowerVLA onto the `lerobot` env requires downgrading
`transformers` and `torch`, which breaks SmolVLA inference. Keeping two
envs is the cheapest stable option.

## One-time setup for the `flower` env

The `flower` conda env is shared filesystem-wide on the lab box and may
already exist. Check first:

```bash
conda env list | grep -E '^flower\b'
```

### Option A — env already exists (lab box, default case)

Just verify it's healthy:

```bash
conda activate flower
python --version                # 3.10.x
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expect: 2.2.2+cu121  True  (False is fine on a Mac; just confirms the env loads)
```

If `import torch` fails, fall through to Option B.

### Option B — create it from scratch (new machine, e.g. teammate's Mac)

You need the sibling `Lerobot_flower` repo cloned locally. The
`environment.yml` lives there.

```bash
# 1. Clone Lerobot_flower next to this repo.
cd <parent of this repo>
git clone <Lerobot_flower remote> Lerobot_flower    # ask Yuqi for the URL

# 2. Create the env (~5–10 min the first time).
cd Lerobot_flower
conda env create -f environment.yml                 # creates env "flower"
conda activate flower
python --version                                    # 3.10.x

# 3. Smoke-test that FlowerVLAPolicy imports.
python -c "import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'third_party'); \
           from src.flower.policy import FlowerVLAPolicy; print('OK')"
```

If you keep the env in sync later (Lerobot_flower added a dep): `conda env
update -f environment.yml --prune`.

## Running an inference rollout from this repo

Once `flower` is active and `Lerobot_flower` is cloned somewhere, run the
script in this repo:

```bash
conda activate flower

# Task 1 — banana in the blue bowl, real arm.
python scripts/run_inference_real_flower.py \
    --checkpoint ethrl2026/so101-eval1-flower-v100x8-all \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 20

# Task 2 — same script, different checkpoint + prompt.
python scripts/run_inference_real_flower.py \
    --checkpoint ethrl2026/so101-eval2-flower-<...> \
    --prompt "<task-2 instruction>" \
    --max-seconds 20

# Dry-run smoke test (no hardware, no libGL needed).
python scripts/run_inference_real_flower.py \
    --checkpoint ethrl2026/so101-eval1-flower-v100x8-all \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 5 --dry-run
```

The script auto-discovers `Lerobot_flower` from the sibling directory of
this repo. If yours lives elsewhere, point at it explicitly:

```bash
python scripts/run_inference_real_flower.py \
    --flower-repo /path/to/Lerobot_flower \
    --checkpoint ... --prompt "..."

# or, persist it for the shell session:
export LEROBOT_FLOWER_ROOT=/path/to/Lerobot_flower
```

## Output layout

Identical to `run_inference_real.py`: each rollout writes a dossier under
`logs/inference/<timestamp>_<ckpt-short>_flower/` containing `meta.json`
(includes `policy_family: flowervla`), `steps.csv`, `outcome.json`, and
`frames/` (if `--frame-every > 0`). The `_flower` suffix in the run dir
name distinguishes flower rollouts from SmolVLA rollouts in the same log
tree.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not locate the Lerobot_flower repo.` | Pass `--flower-repo <path>` or `export LEROBOT_FLOWER_ROOT=<path>`. The script prints all paths it tried. |
| `ImportError: libGL.so.1: cannot open shared object file` | You're on a headless server and have `--frame-every > 0`. Either install `libgl1` (Debian/Ubuntu: `apt install libgl1`) or pass `--frame-every 0` / `--no-log`. |
| `forced_bos_token_id` error on policy load | You're not in the `flower` env (transformers version is wrong). `conda activate flower` and retry. |
| `RuntimeError: CUDA ... mismatch` | The `flower` env is cu121. If your driver is older, run on CPU (`--device cpu`) or rebuild the env against your CUDA. |
