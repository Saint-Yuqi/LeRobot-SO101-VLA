# Architecture & Strategy Decisions

Append-only log. Each decision: date, what, why, what we considered instead.

---

## 2026-04-28 — Project structure: configs + base interface

**Decided:** Configs in YAML under `configs/`, models implement a `BaseVLA`
interface, scripts are thin wrappers over `src/`.

**Why:** Multi-week project with 3 evals → dozens of training runs. Hardcoded
hyperparams = lost reproducibility. Common interface lets us swap end-to-end
vs decoupled approaches without forking.

**Considered:** Hydra (overkill for a 3-month project), pure CLI flags
(rejected — too brittle).

---

## 2026-04-28 — Base model: SmolVLA

**Decided:** Start with SmolVLA as the action-policy backbone for all 3 evals.

**Why:** Smallest LeRobot-native VLA (~450M params), maximizes bonus points,
ecosystem support, recommended in the project brief.

**Considered:** Pi0 (larger, harder to win bonus), TinyVLA/FlowerVLA (less
mature tooling).

---

## TBD — Eval 2 strategy: end-to-end vs decoupled

**Status:** Open. Team discussion needed.

**Options:**
1. End-to-end SmolVLA fine-tuned on compositional prompts. Needs varied
   teleop prompts ("the bowl mixing red and blue", "second from left", etc).
2. Decoupled: frozen VLM resolves prompt + image → target bowl coordinate;
   small policy executes "place banana at coord."

**Decision criteria:** How much teleop time can we afford for prompt-varied
data? How small must the inference-time model be (bonus competition)?

---

## TBD — Eval 3 strategy

**Status:** Open. Strong prior: decoupled is correct.

**Reason:** SmolVLA's vision encoder won't recognize OOD celebrities
(Federer, Merkel) from a few hundred fine-tuning episodes. A frozen VLM
with strong world knowledge will. Bonus point cost depends on whether
"active inference params" includes the VLM call.

**Action item:** Ask in `#project-1-vla` Slack how active params are counted
when a VLM is invoked once per episode for target selection.

---

## TBD — Action representation

**Default:** LeRobot's standard for SO-101 — joint-space, absolute targets,
30 Hz control, action chunk size 50 (SmolVLA default).

**To revisit if:** Overfit test fails or actions look jittery on replay.

---

## TBD — Camera placement (wrist vs shoulder)

**Default:** Keep wrist camera for Eval 1 (close-up of bowl is helpful).

**To revisit:** Eval 3 may benefit from shoulder camera (needs to see all
celebrity images at once, wide field of view).
