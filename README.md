# VLA Project — SO-101 Pick & Place

Vision-language-conditioned manipulation policy for the SO-101 robot.
Three eval setups: color-conditioned pick & place, compositional instructions,
and celebrity image targeting.

## Status

- [ ] Robot calibration verified
- [ ] Teleop recording working
- [ ] First episode recorded & validated
- [ ] Overfit test passes (model memorizes one episode)
- [ ] Inference loop runs on robot from checkpoint
- [ ] Eval 1 dataset collected (~50+ episodes target)
- [ ] Eval 1 model trained
- [ ] Eval 2 strategy decided (end-to-end vs decoupled)
- [ ] Eval 3 strategy decided
- [ ] Final eval rehearsal in HG

## Quickstart

```bash
# 1. Setup
pip install -e .

# 2. Record one episode (with the existing lerobot CLI)
bash scripts/teleop_record.sh data/raw/test_episode

# 3. Validate the recorded episode
python scripts/teleop_validate.py --dataset data/raw/test_episode

# 4. Run overfit test (CRITICAL sanity check before scaling up)
python scripts/overfit_test.py --config configs/train/overfit.yaml

# 5. Replay overfit checkpoint on the real robot
python scripts/run_inference.py --checkpoint checkpoints/overfit/final.pt
```

## Repository Layout

- `configs/`  — YAML configs for every experiment. **Never hardcode hyperparams in scripts.**
- `src/`      — All logic. Importable, testable.
- `scripts/`  — Thin entry points. Parse args, call into `src/`.
- `data/raw/` — Untouched teleop recordings.
- `data/processed/` — LeRobot-format datasets ready for training.
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
