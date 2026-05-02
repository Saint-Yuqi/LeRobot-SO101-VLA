"""Run a trained policy on the real SO-101 robot.

Closed-loop inference: read camera + joints, query the policy, send
actions to the robot, repeat. Includes a hard time limit (per the eval
brief, you have 20 s per rollout).

Usage:
    python scripts/run_inference.py \\
        --checkpoint checkpoints/eval1/best \\
        --prompt "Put the banana in the blue colored bowl." \\
        --max-seconds 20

Implementation detail: this file deliberately depends only on the BaseVLA
interface, NOT on SmolVLA specifically. Swapping in a DecoupledPolicy
later requires zero changes here.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--policy-type", choices=["smolvla", "decoupled"],
                        default="smolvla")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="print actions without sending to robot")
    args = parser.parse_args()

    # Lazy imports
    from src.models.base_vla import Observation
    if args.policy_type == "smolvla":
        from src.models.smolvla_wrapper import SmolVLAWrapper
        policy = SmolVLAWrapper.from_checkpoint(args.checkpoint)
    else:
        from src.models.decoupled_policy import DecoupledPolicy
        policy = DecoupledPolicy.from_checkpoint(args.checkpoint)

    policy = policy.to("cuda").eval()
    policy.reset()
    print(f"[infer] policy loaded. active params: {policy.active_param_count:,}")

    # ---- Robot + camera setup ----
    # NOTE: replace with the actual lerobot robot/camera classes you use.
    # Common pattern:
    #   from lerobot.common.robot_devices.robots.factory import make_robot
    #   robot = make_robot("so101")
    #   robot.connect()
    if args.dry_run:
        robot = None
        print("[infer] DRY RUN — no robot connected.")
    else:
        from lerobot.common.robot_devices.robots.factory import make_robot  # type: ignore
        robot = make_robot("so101")
        robot.connect()

    # ---- Control loop ----
    period = 1.0 / args.control_hz
    deadline = time.time() + args.max_seconds
    action_queue: list[np.ndarray] = []

    while time.time() < deadline:
        loop_start = time.time()

        # Read observation
        if robot is not None:
            cam_frames = robot.capture_observation()  # adapt to your robot API
            images = {"wrist": cam_frames["observation.images.wrist"]}
            state = cam_frames["observation.state"]
        else:
            # Dry-run synthetic obs for offline testing
            images = {"wrist": np.zeros((480, 640, 3), dtype=np.uint8)}
            state = np.zeros(6, dtype=np.float32)

        # Refill action queue if empty
        if not action_queue:
            obs = Observation(images=images, state=state, prompt=args.prompt)
            chunk = policy.predict(obs)
            action_queue = list(chunk.actions)

        action = action_queue.pop(0)

        if robot is not None:
            robot.send_action(action)
        else:
            print(f"[infer] action: {np.round(action, 3)}")

        # Maintain control rate
        elapsed = time.time() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)

    print("[infer] time limit reached.")
    if robot is not None:
        robot.disconnect()


if __name__ == "__main__":
    main()
