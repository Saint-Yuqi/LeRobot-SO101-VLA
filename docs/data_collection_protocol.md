# Data Collection Protocol

Inconsistent teleop data is the #1 cause of poor VLA performance. Follow this protocol every time.

## Before each session

1. **Robot calibration.** Run `lerobot-calibrate` if the arm has been moved or unmounted.
2. **Camera mounted same way as eval.** If we plan to use shoulder cam at eval, record with shoulder cam.
3. **Lighting.** Record under multiple lighting conditions across sessions. *Within* one session keep it constant.
4. **Table surface.** Most sessions on white table; at least 20% on slightly off-white / textured table (HG isn't perfectly white per the brief).
5. **Verify framing.** All three bowls + banana visible in camera frame, gripper ≥15 cm from banana at start.

## Episode recording

For each episode:
1. Set up scene with object positions varying within the ±5 cm allowed range.
2. **Vary the prompt** if relevant for that eval (Eval 2 especially).
3. Smooth, deliberate teleop — no jerky corrections. If you mess up, throw the episode away, do not "save and clean later."
4. Episode ends with banana clearly inside the bowl OR can placed clearly on the celebrity image.
5. Return arm to home pose between episodes (do not record the return).

## Per-eval recording targets (rough)

| Eval | Episodes | Prompt variety | Notes |
|------|----------|----------------|-------|
| 1    | 60–100   | 3 prompts × ~25 each | "Put banana in [blue/red/green] bowl" |
| 2    | 80–150   | 8–12 prompt templates | mixing colors, ordinal, negation |
| 3    | 50–80    | per-celebrity coverage | only if going end-to-end |

## Validation (run before training)

```bash
python scripts/teleop_validate.py --dataset data/raw/<your_session>
```

Checks: action range sanity, image not all-black, episode lengths
reasonable, prompts non-empty, timestamps monotonic.

## Bad data signals — throw the episode away

- Gripper collides with bowl
- Banana ends up between two bowls
- Operator hesitation > 2 s mid-trajectory
- Camera bumped during recording
- Wrong prompt logged for the action performed
