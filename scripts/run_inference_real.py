"""Run a trained policy on the real SO-101 robot (modern lerobot API).

Closed-loop inference: read camera + joints, query the policy, send
actions to the robot, repeat. Writes a per-rollout dossier (meta.json,
steps.csv, outcome.json, optional frames/) so failed attempts can be
debugged after the fact.

Standalone — no project-internal deps. Works with the modern
`lerobot.robots.so101_follower` API and the standard `make_pre_post_processors`
loader. Replaces the older `run_inference.py` which used
`lerobot.common.robot_devices` (removed in current lerobot).

Usage:
    python scripts/run_inference_real.py \\
        --checkpoint ethrl2026/so101-eval2-smolvla-v1 \\
        --prompt "Pick up the banana." \\
        --max-seconds 20

    # Local checkpoint dir works too:
    python scripts/run_inference_real.py \\
        --checkpoint /path/to/checkpoint/dir \\
        --prompt "Pick up the banana." \\
        --max-seconds 20

    # Dry run (no hardware), prints actions only:
    python scripts/run_inference_real.py \\
        --checkpoint <path-or-repo-id> --prompt "..." --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor.factory import make_default_processors
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


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


class _DryRunRobot:
    """Synthetic robot for --dry-run; mirrors SO101Follower's interface."""

    robot_type = "so101_follower_dryrun"
    name = "so101_follower_dryrun"

    def __init__(self, camera_key: str = "main"):
        self._camera_key = camera_key
        self._joints = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in self._joints}

    @property
    def observation_features(self) -> dict:
        out = {f"{j}.pos": float for j in self._joints}
        out[self._camera_key] = (480, 640, 3)
        return out

    @property
    def cameras(self) -> dict:
        return {self._camera_key: None}

    def connect(self) -> None:
        return

    def disconnect(self) -> None:
        return

    def get_observation(self) -> dict:
        out = {f"{j}.pos": 0.0 for j in self._joints}
        out[self._camera_key] = np.zeros((480, 640, 3), dtype=np.uint8)
        return out

    def send_action(self, action: dict) -> dict:
        print(f"[infer] dry-run action: "
              f"{ {k: round(v, 3) for k, v in action.items()} }")
        return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Local checkpoint dir OR HF repo id (user/repo).")
    parser.add_argument("--prompt", required=True, help="Task instruction.")
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--device", default="mps",
                        help="Torch device: mps | cuda | cpu")
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
    parser.add_argument("--display-data", action="store_true",
                        help="Stream observations + actions to a rerun viewer.")
    parser.add_argument("--display-ip", default=None,
                        help="Connect to a remote rerun viewer at this IP (port required).")
    parser.add_argument("--display-port", type=int, default=None,
                        help="Port for the remote rerun viewer.")
    args = parser.parse_args()

    # ---- Resolve checkpoint ----
    ckpt_path, ckpt_source = resolve_checkpoint(args.checkpoint)

    # ---- Per-rollout log dir ----
    ckpt_short = Path(args.checkpoint).name.replace("/", "-")[:48]
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{ckpt_short}"
    log_dir = Path(args.log_dir) / run_id
    if not args.no_log:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[infer] log_dir: {log_dir}")

    meta = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "checkpoint_source": ckpt_source,
        "checkpoint_local_path": ckpt_path,
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

    # ---- Load policy + processors directly (bypass make_policy, which
    # demands ds_meta or env_cfg). Two layouts supported:
    #   1) Full save (SmolVLA, pi0 task1 train_expert_only): ckpt dir has
    #      config.json + model.safetensors → load directly.
    #   2) PEFT adapter save (pi0 task2 LoRA): ckpt dir has adapter_config.json
    #      pointing at a base model (lerobot/pi0). Load the base, then wrap
    #      with PeftModel to attach the adapter weights. ----
    adapter_cfg_path = Path(ckpt_path) / "adapter_config.json"
    is_peft_adapter = adapter_cfg_path.exists()

    if is_peft_adapter:
        from peft import PeftConfig, PeftModel

        peft_cfg = PeftConfig.from_pretrained(ckpt_path)
        base_path = peft_cfg.base_model_name_or_path
        if not base_path:
            sys.exit(f"adapter_config.json at {ckpt_path} has no base_model_name_or_path")
        print(f"[infer] PEFT adapter detected; loading base policy from {base_path} "
              f"then attaching LoRA from {ckpt_path}")
        policy_cfg = PreTrainedConfig.from_pretrained(base_path)
        policy_cfg.device = args.device
        policy_cfg.pretrained_path = base_path
        policy_cls = get_policy_class(policy_cfg.type)
        base_policy = policy_cls.from_pretrained(base_path, config=policy_cfg)
        policy = PeftModel.from_pretrained(base_policy, ckpt_path, config=peft_cfg)
        policy.to(args.device)
    else:
        print(f"[infer] loading policy from {ckpt_path} on {args.device}...")
        policy_cfg = PreTrainedConfig.from_pretrained(ckpt_path)
        policy_cfg.device = args.device
        policy_cfg.pretrained_path = ckpt_path
        policy_cls = get_policy_class(policy_cfg.type)
        policy = policy_cls.from_pretrained(ckpt_path, config=policy_cfg)
    policy.eval()

    # Processors are saved alongside the policy in the ckpt dir for both
    # layouts (train.py calls preprocessor.save_pretrained(ckpt_dir) regardless
    # of PEFT). Pass the active policy_cfg so the factory can derive shapes.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=ckpt_path,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
        },
    )
    print(f"[infer] policy type={policy_cfg.type}, "
          f"chunk_size={getattr(policy_cfg, 'chunk_size', '?')}")

    # ---- Optional rerun viewer ----
    if args.display_data:
        init_rerun(session_name=run_id, ip=args.display_ip, port=args.display_port)
        print("[infer] rerun viewer initialized (spawned local viewer or "
              f"connected to {args.display_ip}:{args.display_port})")

    # ---- Robot ----
    if args.dry_run:
        robot = _DryRunRobot(camera_key=args.camera_key)
        print("[infer] DRY RUN — synthetic robot, no hardware.")
    else:
        robot_cfg = SO101FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            cameras={args.camera_key: OpenCVCameraConfig(
                index_or_path=args.camera_index, width=640, height=480, fps=30
            )},
        )
        robot = SO101Follower(robot_cfg)
    robot.connect()

    # ---- Build dataset_features (needed by build_dataset_frame +
    # make_robot_action). We don't have a real dataset here, but the
    # default processors expose the same feature derivation lerobot-record
    # uses internally. ----
    teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()
    dataset_features = {
        **aggregate_pipeline_dataset_features(
            pipeline=teleop_action_proc,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        **aggregate_pipeline_dataset_features(
            pipeline=robot_obs_proc,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    }

    # ---- Loop ----
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    period = 1.0 / float(args.control_hz)
    deadline = time.perf_counter() + float(args.max_seconds)
    device_t = get_safe_torch_device(args.device)

    # Open steps.csv lazily; need first observation to know joint names.
    steps_writer = None
    steps_fh = None
    frames_dir = log_dir / "frames" if (not args.no_log and args.frame_every > 0) else None
    if frames_dir is not None:
        frames_dir.mkdir(exist_ok=True)

    step = 0
    termination = "completed_max_seconds"
    t_start = time.perf_counter()

    try:
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            obs_processed = robot_obs_proc(obs)
            observation_frame = build_dataset_frame(
                dataset_features, obs_processed, prefix=OBS_STR
            )

            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device_t,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
                task=args.prompt,
                robot_type=getattr(robot, "robot_type", None),
            )

            act_processed = make_robot_action(action_values, dataset_features)
            robot_action_to_send = robot_action_proc((act_processed, obs))

            # NaN guard
            for k, v in robot_action_to_send.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    termination = "policy_nan"
                    raise RuntimeError(f"policy emitted non-finite action on key {k}: {v}")

            sent = robot.send_action(robot_action_to_send)

            if args.display_data:
                log_rerun_data(observation=obs_processed, action=robot_action_to_send)

            # Logging
            if not args.no_log:
                if steps_writer is None:
                    action_keys = sorted(robot.action_features)
                    state_keys = [k for k in obs.keys() if k.endswith(".pos")]
                    state_keys.sort()
                    steps_fh = open(log_dir / "steps.csv", "w", newline="")
                    steps_writer = csv.writer(steps_fh)
                    steps_writer.writerow(
                        ["step", "t_elapsed_s", "loop_dt_s"]
                        + [f"obs.{k}" for k in state_keys]
                        + [f"act.{k}" for k in action_keys]
                    )
                row = [step, t0 - t_start, time.perf_counter() - t0]
                row += [obs.get(k, "") for k in state_keys]
                row += [sent.get(k, "") for k in action_keys]
                steps_writer.writerow(row)

                if frames_dir is not None and step % args.frame_every == 0:
                    img = obs.get(args.camera_key)
                    if isinstance(img, np.ndarray) and img.ndim == 3:
                        cv2.imwrite(
                            str(frames_dir / f"frame_{step:05d}.jpg"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        )

            step += 1
            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
            else:
                # Slow loop — visible warning so users know.
                if step % 30 == 1:
                    print(f"[infer] WARN slow loop: {1.0/elapsed:.1f}Hz (target {args.control_hz}Hz)")

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

        outcome = {
            "termination": termination,
            "steps": step,
            "wall_seconds": time.perf_counter() - t_start,
            "ended_at": datetime.now().isoformat(),
        }
        if not args.no_log:
            (log_dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
        print(f"[infer] done. steps={step} termination={termination} "
              f"wall={outcome['wall_seconds']:.2f}s")
        if not args.no_log:
            print(f"[infer] logs at {log_dir}")

    sys.exit(0 if termination in ("completed_max_seconds", "user_abort") else 2)


if __name__ == "__main__":
    main()