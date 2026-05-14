"""Run a trained FlowerVLA policy on the real SO-101 robot.

Florence-2-backed counterpart of `scripts/run_inference_real.py`. The CLI,
defaults, and per-rollout dossier (meta.json, steps.csv, outcome.json,
optional frames/) are kept identical so muscle memory carries over from the
SmolVLA workflow. Only the policy backend differs: FlowerVLAPolicy is loaded
from the sibling `Lerobot_flower` repo, and the closed-loop control logic
lives in `src.flower.runner.run_rollout`.

Task1 and Task2 are both handled the same way — the policy is
task-conditioned through the language prompt, so just point `--checkpoint` at
the right Hub repo and pass the matching `--prompt`. No code change needed
between tasks.

Requirements:
    - The sibling `Lerobot_flower` repo (auto-discovered, or pass
      `--flower-repo /path/to/Lerobot_flower`, or set $LEROBOT_FLOWER_ROOT).
    - The `flower` conda env (Python 3.10 with the FlowerVLA stack).

Usage:
    # HF Hub checkpoint, dry-run (no hardware):
    python scripts/run_inference_real_flower.py \\
        --checkpoint ethrl2026/so101-eval1-flower-v100x8-all \\
        --prompt "Pick up the banana." \\
        --max-seconds 20 --dry-run

    # Live arm:
    python scripts/run_inference_real_flower.py \\
        --checkpoint ethrl2026/so101-eval1-flower-v100x8-all \\
        --prompt "Pick up the banana." \\
        --max-seconds 20

    # Task2 checkpoint, same script:
    python scripts/run_inference_real_flower.py \\
        --checkpoint ethrl2026/so101-eval2-flower-... \\
        --prompt "<task2 instruction>" \\
        --max-seconds 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

# cv2 is only needed for optional frame-saving; lazy-import in _on_step to
# avoid a libGL hard-dependency on headless boxes (dry-run smoke tests).
cv2 = None  # type: ignore[assignment]


# --------------------------------------------------------------------- imports
# FlowerVLA lives in the sibling repo. Locate it before importing src.flower.*

def _locate_flower_repo(cli_value: str | None) -> Path:
    """Return the Lerobot_flower repo root, or exit with a helpful error."""
    candidates: list[Path] = []
    if cli_value:
        candidates.append(Path(cli_value).expanduser().resolve())
    env_value = os.environ.get("LEROBOT_FLOWER_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser().resolve())

    here = Path(__file__).resolve()
    # Sibling layout: .../Lerobot/scripts/run_inference_real_flower.py
    #              -> .../Lerobot_flower
    for parent in (here.parent.parent.parent, here.parent.parent):
        candidates.append((parent / "Lerobot_flower").resolve())

    # Common absolute paths in the user's setup.
    candidates.extend([
        Path("/home/yuqyan/Yuqi/Lerobot_flower"),
        Path("/shares/feldmann.ics.mnf.uzh/Yuqi/Lerobot_flower"),
    ])

    seen: set[str] = set()
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        if (c / "src" / "flower" / "policy.py").is_file():
            return c

    sys.exit(
        "Could not locate the Lerobot_flower repo.\n"
        "Tried:\n  " + "\n  ".join(sorted(seen)) + "\n"
        "Pass --flower-repo /path/to/Lerobot_flower or set "
        "LEROBOT_FLOWER_ROOT."
    )


def _bootstrap_flower_imports(flower_repo: Path) -> None:
    """Prepend Lerobot_flower (and its third_party dir) to sys.path."""
    for p in (str(flower_repo), str(flower_repo / "third_party")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------- helpers

def resolve_checkpoint(arg: str) -> tuple[str, str]:
    """Accept either a local checkpoint dir or an HF Hub repo id.

    Returns (local_path, source) where source is 'local' or 'hf'.
    """
    p = Path(arg)
    if p.is_dir() and (p / "config.json").exists():
        return str(p), "local"
    if p.exists():
        sys.exit(
            f"--checkpoint '{arg}' exists but is not a checkpoint dir "
            "(missing config.json)."
        )
    if "/" not in arg or arg.count("/") > 1:
        sys.exit(
            f"--checkpoint '{arg}' is neither a local checkpoint dir nor an "
            "HF repo id of the form '<user>/<repo>'."
        )
    print(f"[infer] '{arg}' is not a local path; downloading from HF Hub...")
    local_path = snapshot_download(repo_id=arg, repo_type="model")
    print(f"[infer] cached at {local_path}")
    return local_path, "hf"


# ----------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Local FlowerVLA checkpoint dir OR HF repo id "
                             "(e.g. ethrl2026/so101-eval1-flower-v100x8-all). "
                             "Works for any task — the policy is "
                             "prompt-conditioned.")
    parser.add_argument("--prompt", required=True, help="Task instruction.")
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--device", default="cuda",
                        help="Torch device: cuda | mps | cpu "
                             "(FlowerVLA is heavier than SmolVLA — prefer cuda).")
    parser.add_argument("--robot-port", default="/dev/tty.usbmodem5B141136551")
    parser.add_argument("--robot-id", default="follower_111")
    parser.add_argument("--camera-key", default="main")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--log-dir", default="logs/inference")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--frame-every", type=int, default=10,
                        help="Save a JPEG every K steps; 0 disables.")
    parser.add_argument("--dry-run", action="store_true",
                        help="No hardware; print actions instead of sending.")
    parser.add_argument("--flower-repo", default=None,
                        help="Path to the Lerobot_flower checkout. Auto-detected "
                             "if omitted (sibling dir or $LEROBOT_FLOWER_ROOT).")
    args = parser.parse_args()

    # ---- Locate sibling repo and import flower stack ----
    flower_repo = _locate_flower_repo(args.flower_repo)
    _bootstrap_flower_imports(flower_repo)
    print(f"[infer] using Lerobot_flower at {flower_repo}")

    from src.flower.policy import FlowerVLAPolicy
    from src.flower.runner import (
        DryRunRobot,
        JOINT_KEYS,
        make_live_robot,
        run_rollout,
    )

    # ---- Resolve checkpoint ----
    ckpt_path, ckpt_source = resolve_checkpoint(args.checkpoint)

    # ---- Per-rollout log dir ----
    ckpt_short = Path(args.checkpoint).name.replace("/", "-")[:48]
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{ckpt_short}_flower"
    log_dir = Path(args.log_dir) / run_id
    if not args.no_log:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[infer] log_dir: {log_dir}")

    meta = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "policy_family": "flowervla",
        "checkpoint": args.checkpoint,
        "checkpoint_source": ckpt_source,
        "checkpoint_local_path": ckpt_path,
        "flower_repo": str(flower_repo),
        "prompt": args.prompt,
        "max_seconds": args.max_seconds,
        "control_hz": args.control_hz,
        "device": args.device,
        "robot_port": args.robot_port,
        "robot_id": args.robot_id,
        "camera_key": args.camera_key,
        "camera_index": args.camera_index,
        "dry_run": args.dry_run,
    }
    if not args.no_log:
        (log_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---- Load policy ----
    print(f"[infer] loading FlowerVLA policy from {ckpt_path} on {args.device}...")
    policy = FlowerVLAPolicy.from_pretrained(ckpt_path, device=args.device)
    policy.eval()
    chunk_size = int(policy.config.chunk_size)
    action_dim = int(policy.config.action_dim)
    image_hw = int(policy.config.image_hw)
    video_key = policy.config.video_key
    print(f"[infer] policy: chunk_size={chunk_size} "
          f"action_dim={action_dim} image_hw={image_hw}")

    # ---- Robot ----
    if args.dry_run:
        robot = DryRunRobot(camera_key=args.camera_key)
        print("[infer] DRY RUN — synthetic robot, no hardware.")
    else:
        robot = make_live_robot(
            port=args.robot_port,
            robot_id=args.robot_id,
            camera_key=args.camera_key,
            camera_index=args.camera_index,
        )
    robot.connect()

    # ---- Logging plumbing (mirrors run_inference_real.py's csv layout) ----
    steps_writer = None
    steps_fh = None
    frames_dir = (
        log_dir / "frames"
        if (not args.no_log and args.frame_every > 0)
        else None
    )
    if frames_dir is not None:
        frames_dir.mkdir(exist_ok=True)

    t_start = time.perf_counter()
    last_step_count = {"step": 0}

    def _on_step(payload: dict) -> None:
        nonlocal steps_writer, steps_fh
        step = int(payload["step"])
        state = payload["state"]  # np.ndarray (action_dim,)
        action_sent = payload["action_sent"]  # np.ndarray (action_dim,)
        period_ms = payload.get("period_actual_ms")

        if not args.no_log:
            if steps_writer is None:
                action_keys = list(JOINT_KEYS)  # joint ordering of the robot
                state_keys = list(JOINT_KEYS)
                steps_fh = open(log_dir / "steps.csv", "w", newline="")
                steps_writer = csv.writer(steps_fh)
                steps_writer.writerow(
                    ["step", "t_elapsed_s", "period_ms",
                     "chunk_idx", "chunk_step", "inferred_this_step"]
                    + [f"obs.{k}" for k in state_keys]
                    + [f"act.{k}" for k in action_keys]
                )
            row = [
                step,
                time.perf_counter() - t_start,
                "" if period_ms is None else round(float(period_ms), 3),
                int(payload["chunk_idx"]),
                int(payload["chunk_step"]),
                int(bool(payload["inferred_this_step"])),
            ]
            row += [float(v) for v in state.tolist()]
            row += [float(v) for v in action_sent.tolist()]
            steps_writer.writerow(row)

            if frames_dir is not None and step % args.frame_every == 0:
                img = payload.get("image")
                if isinstance(img, np.ndarray) and img.ndim == 3:
                    global cv2
                    if cv2 is None:
                        import cv2 as _cv2
                        cv2 = _cv2
                    cv2.imwrite(
                        str(frames_dir / f"frame_{step:05d}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    )
        last_step_count["step"] = step + 1

    termination = "completed_max_seconds"
    summary: dict = {}
    try:
        summary = run_rollout(
            policy=policy,
            robot=robot,
            prompt=args.prompt,
            max_seconds=float(args.max_seconds),
            control_hz=float(args.control_hz),
            camera_key=args.camera_key,
            image_hw=image_hw,
            video_key=video_key,
            on_step=_on_step,
            on_chunk=None,
        )
        # run_rollout's reasons: "timeout" | "user_abort" | "policy_nan"
        # | "policy_fault" | "robot_fault". Map to dossier vocabulary.
        reason = summary.get("termination_reason", "timeout")
        termination = {
            "timeout": "completed_max_seconds",
            "user_abort": "user_abort",
            "policy_nan": "policy_nan",
            "policy_fault": "policy_fault",
            "robot_fault": "robot_fault",
        }.get(reason, reason)
    except KeyboardInterrupt:
        termination = "user_abort"
        print("\n[infer] Ctrl-C — aborting rollout.")
    except Exception as e:
        termination = f"exception:{type(e).__name__}"
        print(f"[infer] EXCEPTION: {e!r}")
        traceback.print_exc()
    finally:
        if steps_fh is not None:
            steps_fh.close()
        try:
            robot.disconnect()
        except Exception as e:
            print(f"[infer] WARN robot.disconnect failed: {e!r}")

        steps = int(summary.get("steps", last_step_count["step"]))
        wall = float(summary.get("wall_seconds", time.perf_counter() - t_start))
        outcome = {
            "termination": termination,
            "steps": steps,
            "wall_seconds": wall,
            "final_chunk_idx": int(summary.get("final_chunk_idx", -1)),
            "ended_at": datetime.now().isoformat(),
        }
        if not args.no_log:
            (log_dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
        print(f"[infer] done. steps={steps} termination={termination} "
              f"wall={wall:.2f}s")
        if not args.no_log:
            print(f"[infer] logs at {log_dir}")

    sys.exit(0 if termination in ("completed_max_seconds", "user_abort") else 2)


if __name__ == "__main__":
    main()
