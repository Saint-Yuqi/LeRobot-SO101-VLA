"""Run a trained policy on the real SO-101 robot, with full telemetry.

Closed-loop inference: read camera + joints, query the policy, send actions
to the robot, repeat. Hard time limit (per the eval brief, 20 s per rollout)
and a per-rollout dossier (`meta.json`, `steps.csv`, `chunks.jsonl`,
`chunks.npz`, `episode.json`, `outcome.json`, optional `frames/`) so failed
attempts can be debugged after the fact.

Built on the modern `lerobot.robots.so_follower` + `predict_action` pipeline
(the team verified this end-to-end on the physical arm). The previous
`run_inference_real.py` has been folded in here.

Usage:
    # HF Hub repo id — auto-downloads on first use.
    python scripts/run_inference.py \\
        --checkpoint ethrl2026/so101-eval1-smolvla-v2 \\
        --prompt "Put the banana in the blue colored bowl." \\
        --max-seconds 20 --device mps \\
        --robot-port /dev/tty.usbmodem5B141136551 --robot-id follower_111

    # Local checkpoint dir works too.
    # Dry-run (no hardware), prints actions:
    python scripts/run_inference.py \\
        --checkpoint <path-or-repo-id> --prompt "..." --dry-run --no-wandb
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.rollout_logger import NoopLogger, RolloutLogger
from src.utils.checkpoint_meta import checkpoint_short_id, resolve_training_run
from src.utils.run_metadata import capture_runtime_metadata


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_checkpoint(arg: str) -> tuple[str, str, str]:
    """Accept either a local checkpoint dir or a HF Hub repo id.

    Returns (local_path, source, origin) where source is 'local' or 'hf'
    and origin preserves the original argument (the HF repo id when used).
    """
    p = Path(arg)
    if p.is_dir() and (p / "config.json").exists():
        return str(p), "local", str(p)
    if p.exists():
        raise SystemExit(
            f"--checkpoint '{arg}' exists but is not a checkpoint dir "
            "(missing config.json)."
        )
    if "/" not in arg or arg.count("/") > 1:
        raise SystemExit(
            f"--checkpoint '{arg}' is neither a local checkpoint dir nor "
            "a HuggingFace repo id of the form '<user>/<repo>'."
        )

    print(f"[infer] '{arg}' is not a local path; pulling from HuggingFace Hub...")
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(repo_id=arg, repo_type="model")
    print(f"[infer] cached at {local_path}")
    return local_path, "hf", arg


def _sanitize_tag(prefix: str, body: str, max_body: int = 32) -> str:
    """Wandb tags must match a restricted charset and stay <= 64 chars."""
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", body)[:max_body]
    return f"{prefix}_{safe}"


class _DryRunRobot:
    """Synthetic robot for `--dry-run`; mirrors `SO101Follower`'s interface."""

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
        out: dict = {f"{j}.pos": float for j in self._joints}
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
        out: dict = {f"{j}.pos": 0.0 for j in self._joints}
        out[self._camera_key] = np.zeros((480, 640, 3), dtype=np.uint8)
        return out

    def send_action(self, action: dict) -> dict:
        print(f"[infer] dry-run action: "
              f"{ {k: round(float(v), 3) for k, v in action.items()} }")
        return action


def _build_logger(args, meta_dict: dict, action_dim: int, chunk_size: int):
    """Construct the logger; cheap (does NOT open wandb)."""
    if args.no_log:
        return NoopLogger()
    return RolloutLogger(
        log_dir=args.log_dir,
        inference_run_id=meta_dict["inference_run_id"],
        meta=meta_dict,
        action_dim=int(action_dim),
        state_dim=int(action_dim),
        control_hz=float(args.control_hz),
        chunk_size=int(chunk_size),
        frame_every=int(args.frame_every),
        video=bool(args.video),
    )


def _install_chunk_capture(policy):
    """Wrap the policy's chunk-generating method so each call lands in a side slot.

    `select_action` calls `_get_action_chunk` exactly once per queue refill
    (and `predict_action_chunk` on the RTC code path) — patching both means
    we fire once per chunk no matter which entry point dequeue uses, with
    no extra forward pass. Returns the slot dict the loop polls after each
    `predict_action(...)` call.
    """
    slot: dict = {"count": 0, "actions": None, "t0": None, "t1": None}

    # `_get_action_chunk` is the lowest-level chunk producer; both
    # `select_action` (used by predict_action) and the public
    # `predict_action_chunk` route through it, so a single hook here
    # captures every chunk exactly once.
    target_name = "_get_action_chunk" if hasattr(policy, "_get_action_chunk") \
        else ("predict_action_chunk" if hasattr(policy, "predict_action_chunk") else None)
    if target_name is None:
        return slot
    orig = getattr(policy, target_name)

    def _wrapped(batch, *posargs, **kwargs):
        t0 = time.perf_counter()
        out = orig(batch, *posargs, **kwargs)
        try:
            arr = out.detach().to("cpu").float().numpy()
            if arr.ndim == 3:
                arr = arr[0]
        except Exception:
            arr = None
        slot["actions"] = arr
        slot["t0"] = t0
        slot["t1"] = time.perf_counter()
        slot["count"] += 1
        return out

    setattr(policy, target_name, _wrapped)
    return slot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True,
        help="Local checkpoint dir OR HuggingFace repo id (user/repo).",
    )
    parser.add_argument("--prompt", required=True, help="Task instruction.")
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--device", default="cuda",
                        help="Torch device: cuda | mps | cpu")

    parser.add_argument("--robot-port", default="/dev/tty.usbmodem5B141136551")
    parser.add_argument("--robot-id", default="follower_111")
    parser.add_argument("--camera-key", default="main",
                        help="Camera name used during teleop recording. "
                             "Eval1/Eval2 datasets use 'main'.")
    parser.add_argument("--camera-index", type=int, default=0)

    parser.add_argument("--dry-run", action="store_true",
                        help="No hardware; print actions instead of sending.")
    parser.add_argument("--log-dir", default="logs/inference",
                        help="parent dir for the per-run subdir")
    parser.add_argument("--no-log", action="store_true",
                        help="swap RolloutLogger for the noop logger (smoke test)")
    parser.add_argument("--frame-every", type=int, default=10,
                        help="save a JPEG every K steps; 0 disables")
    parser.add_argument("--video", action="store_true",
                        help="stitch episode.mp4 from saved frames")

    parser.add_argument("--no-wandb", action="store_true",
                        help="skip wandb (useful when lab box is offline)")
    parser.add_argument("--wandb-project", default="Lerobot-rollouts",
                        help="rollout-side wandb project (NOT inherited from sidecar)")
    parser.add_argument("--wandb-mode", choices=["online", "offline"],
                        default="online")

    parser.add_argument("--display-data", action="store_true",
                        help="Stream observations + actions to a rerun viewer.")
    parser.add_argument("--display-ip", default=None,
                        help="Connect to a remote rerun viewer at this IP.")
    parser.add_argument("--display-port", type=int, default=None,
                        help="Port for the remote rerun viewer.")

    args = parser.parse_args()

    ckpt_path, ckpt_source, ckpt_origin = resolve_checkpoint(args.checkpoint)

    # ---- Identity / lineage ----
    runtime = capture_runtime_metadata()
    ckpt_short = checkpoint_short_id(args.checkpoint)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    inference_run_id = (
        f"{time.strftime('%Y%m%d-%H%M%S')}_job{job_id}_{ckpt_short}"
    )
    training = resolve_training_run(ckpt_path)

    # ---- Modern lerobot imports (gated so dry-run still works on a
    # workstation that has no robot hardware drivers installed). ----
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.policies.factory import (
        get_policy_class, make_pre_post_processors,
    )
    from lerobot.policies.utils import make_robot_action
    from lerobot.processor.factory import make_default_processors
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    meta_dict: dict = {
        "inference_run_id": inference_run_id,
        **runtime,
        "cli_args": vars(args),
        "checkpoint_source": ckpt_source,
        "checkpoint_path": ckpt_path,
        "checkpoint_origin": ckpt_origin,
        "training_run": training,
        "started_at": datetime.now().isoformat(),
        "robot": {
            "robot_type": "so101_follower_dryrun" if args.dry_run else "so101_follower",
            "port": args.robot_port,
            "id": args.robot_id,
            "camera_key": args.camera_key,
            "camera_index": args.camera_index,
        },
    }

    if args.video and args.no_log:
        print("[infer] WARNING: --video is a no-op with --no-log "
              "(NoopLogger.maybe_save_frame returns ''); skipping mp4.")

    # Logger with provisional dims; refresh after policy load.
    logger = _build_logger(args, meta_dict, action_dim=6, chunk_size=50)
    if isinstance(logger, RolloutLogger):
        print(f"[infer] inference_run_id: {inference_run_id}")
        print(f"[infer] log_dir: {logger.run_dir}")

    termination_reason = "init_fault"
    wandb_run = None
    robot = None
    policy = None
    t_start = time.perf_counter()
    step = 0

    try:
        # ---- Policy + pre/post-processors (modern lerobot) ----
        print(f"[infer] loading policy from {ckpt_path} on {args.device}...")
        policy_cfg = PreTrainedConfig.from_pretrained(ckpt_path)
        policy_cfg.device = args.device
        policy_cfg.pretrained_path = ckpt_path
        policy_cls = get_policy_class(policy_cfg.type)
        policy = policy_cls.from_pretrained(ckpt_path, config=policy_cfg)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=ckpt_path,
            preprocessor_overrides={
                "device_processor": {"device": args.device},
            },
        )
        chunk_size = int(getattr(policy_cfg, "chunk_size", 50) or 50)
        print(f"[infer] policy type={policy_cfg.type}, chunk_size={chunk_size}")

        try:
            param_count = sum(int(p.numel()) for p in policy.parameters() if p.requires_grad)
            meta_dict["policy_active_param_count"] = param_count
            meta_dict["policy_type"] = str(policy_cfg.type)
            meta_dict["policy_chunk_size"] = chunk_size
            print(f"[infer] policy loaded. active params: {param_count:,}")
        except Exception:
            pass

        # Install chunk-capture hook so RolloutLogger can still log per-chunk.
        chunk_slot = _install_chunk_capture(policy)

        # ---- Optional rerun viewer ----
        if args.display_data:
            init_rerun(session_name=inference_run_id,
                       ip=args.display_ip, port=args.display_port)
            print("[infer] rerun viewer initialized "
                  f"(spawned local viewer or connected to "
                  f"{args.display_ip}:{args.display_port})")

        # ---- Robot ----
        if args.dry_run:
            robot = _DryRunRobot(camera_key=args.camera_key)
            print("[infer] DRY RUN — synthetic robot, no hardware.")
        else:
            from lerobot.cameras.opencv import OpenCVCameraConfig
            from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
            from lerobot.robots.so_follower.so_follower import SO101Follower
            robot_cfg = SO101FollowerConfig(
                port=args.robot_port,
                id=args.robot_id,
                cameras={args.camera_key: OpenCVCameraConfig(
                    index_or_path=args.camera_index,
                    width=640, height=480, fps=30,
                )},
            )
            robot = SO101Follower(robot_cfg)
        robot.connect()

        # Refresh logger dims now that we know the actual action_features.
        action_keys = sorted(robot.action_features)
        action_dim = len(action_keys)
        if isinstance(logger, RolloutLogger):
            logger.action_dim = action_dim
            logger.state_dim = action_dim
            logger.chunk_size = chunk_size
            # Re-write meta.json with the now-complete metadata.
            (logger.run_dir / "meta.json").write_text(
                json.dumps(meta_dict, indent=2, default=str)
            )

        # ---- Dataset features for build_dataset_frame + make_robot_action ----
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

        # ---- Wandb (best-effort) ----
        if not args.no_wandb:
            try:
                import wandb
                tags = [ckpt_short]
                tags.append(_sanitize_tag(
                    "robot", str(getattr(robot, "robot_type", "unknown"))
                ))
                tags.append(_sanitize_tag("prompt", str(args.prompt)))
                training_group = (training or {}).get("wandb_run_id") or "no-sidecar"
                training_entity = (training or {}).get("wandb_entity")
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=training_entity,
                    id=inference_run_id,
                    name=inference_run_id,
                    group=training_group,
                    tags=tags,
                    config=meta_dict,
                    mode=args.wandb_mode,
                    dir=str(Path(args.log_dir) / inference_run_id),
                )
                print(f"[infer] wandb: "
                      f"{getattr(wandb_run, 'url', None) or '(offline)'}")
            except Exception as e:
                print(f"[infer] WARNING: wandb.init failed ({e!r}); continuing offline")
                wandb_run = None
        logger.attach_wandb(wandb_run)

        # ---- Reset policy + processors ----
        try:
            policy.reset()
        except Exception:
            pass
        try:
            preprocessor.reset()
            postprocessor.reset()
        except Exception:
            pass

        # ---- Control loop ----
        period = 1.0 / max(float(args.control_hz), 1e-6)
        deadline = time.perf_counter() + float(args.max_seconds)
        device_t = get_safe_torch_device(args.device)

        chunk_idx = -1
        chunk_step = 0
        prev_loop_start: float | None = None
        last_chunk_count = chunk_slot["count"]
        termination_reason = "timeout"

        while time.perf_counter() < deadline:
            loop_start = time.perf_counter()
            period_actual_ms = (
                None if prev_loop_start is None
                else (loop_start - prev_loop_start) * 1000.0
            )

            # Read observation.
            try:
                obs = robot.get_observation()
            except Exception as e:
                termination_reason = "robot_fault"
                logger.note_event("error.robot", repr(e))
                print(f"[infer] EXCEPTION during get_observation: {e!r}")
                traceback.print_exc()
                break

            obs_processed = robot_obs_proc(obs)
            observation_frame = build_dataset_frame(
                dataset_features, obs_processed, prefix=OBS_STR
            )

            # Predict (queues a chunk on first/empty step; pops one action otherwise).
            try:
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
            except Exception as e:
                termination_reason = "policy_fault"
                logger.note_event("error.policy", repr(e))
                print(f"[infer] EXCEPTION during predict_action: {e!r}")
                traceback.print_exc()
                break

            # If a new chunk was generated this step, log it before logging the step.
            inferred_this_step = chunk_slot["count"] > last_chunk_count
            if inferred_this_step:
                chunk_idx += 1
                chunk_step = 0
                last_chunk_count = chunk_slot["count"]
                actions_arr = chunk_slot["actions"]
                if actions_arr is not None:
                    chunk_obj = SimpleNamespace(
                        actions=actions_arr,
                        chunk_size=int(actions_arr.shape[0]),
                    )
                    state_keys = sorted(k for k in obs.keys() if k.endswith(".pos"))
                    state_arr = np.array(
                        [float(obs[k]) for k in state_keys], dtype=np.float32
                    )
                    images_for_log = {}
                    img = obs.get(args.camera_key)
                    if isinstance(img, np.ndarray) and img.ndim == 3:
                        images_for_log[args.camera_key] = img
                    try:
                        logger.log_chunk(
                            chunk_idx=chunk_idx,
                            t0=chunk_slot["t0"],
                            t1=chunk_slot["t1"],
                            state=state_arr,
                            prompt=args.prompt,
                            images=images_for_log,
                            chunk=chunk_obj,
                        )
                    except Exception as e:
                        logger.note_event("warn.log_chunk", repr(e))
            else:
                chunk_step += 1

            act_processed = make_robot_action(action_values, dataset_features)
            robot_action_to_send = robot_action_proc((act_processed, obs))

            # NaN/Inf guard — abort BEFORE sending to the arm.
            nan_dim = None
            for k, v in robot_action_to_send.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                if not np.isfinite(fv):
                    nan_dim = k
                    break
            if nan_dim is not None:
                termination_reason = "policy_nan"
                logger.note_event(
                    "nan",
                    f"step={step} key={nan_dim} val={robot_action_to_send[nan_dim]!r}",
                )
                print(f"[infer] EXCEPTION: policy emitted non-finite action on {nan_dim}")
                break

            # Send to robot.
            try:
                sent = robot.send_action(robot_action_to_send)
            except Exception as e:
                termination_reason = "robot_fault"
                logger.note_event("error.robot", repr(e))
                print(f"[infer] EXCEPTION during send_action: {e!r}")
                traceback.print_exc()
                break

            # Optional rerun stream.
            if args.display_data:
                try:
                    log_rerun_data(observation=obs_processed, action=robot_action_to_send)
                except Exception as e:
                    logger.note_event("warn.rerun", repr(e))

            # Telemetry.
            try:
                state_keys = sorted(k for k in obs.keys() if k.endswith(".pos"))
                state_arr = np.array(
                    [float(obs[k]) for k in state_keys], dtype=np.float32
                )
                action_raw_arr = np.asarray(action_values, dtype=np.float32).reshape(-1)
                action_sent_arr = np.array(
                    [float(sent.get(k, robot_action_to_send.get(k, np.nan)))
                     for k in action_keys],
                    dtype=np.float32,
                )
                images_for_log = {}
                img = obs.get(args.camera_key)
                if isinstance(img, np.ndarray) and img.ndim == 3:
                    images_for_log[args.camera_key] = img
                queue_depth_after = 0
                try:
                    queue_depth_after = int(len(policy._queues[ACTION]))
                except Exception:
                    pass
                logger.log_step(
                    step=step,
                    chunk_idx=max(chunk_idx, 0),
                    chunk_step=chunk_step,
                    inferred_this_step=inferred_this_step,
                    queue_depth_after=queue_depth_after,
                    state=state_arr,
                    action_raw=action_raw_arr,
                    action_sent=action_sent_arr,
                    clamped_mask=np.zeros(action_dim, dtype=np.uint8),
                    period_actual_ms=period_actual_ms,
                    frame_path=logger.maybe_save_frame(images_for_log, step),
                )
            except Exception as e:
                logger.note_event("warn.log_step", repr(e))

            step += 1
            prev_loop_start = loop_start
            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
            else:
                # Surface slow loops periodically — critical when running on the arm.
                if step % 30 == 1:
                    print(f"[infer] WARN slow loop: {1.0/elapsed:.1f}Hz "
                          f"(target {args.control_hz}Hz)")

        # while-loop ran to its deadline without break → timeout.
        if termination_reason == "init_fault":
            termination_reason = "timeout"
        print(f"[infer] termination_reason: {termination_reason}")

        # Reverse pointer on the training run (best-effort, online only).
        if (training is not None
                and wandb_run is not None
                and args.wandb_mode == "online"):
            _push_reverse_pointer(
                training=training,
                rollout_id=inference_run_id,
                rollout_url=getattr(wandb_run, "url", "") or "",
                verdict="unset",
            )

    except KeyboardInterrupt:
        termination_reason = "user_abort"
        print("\n[infer] Ctrl-C — aborting rollout.")
    except Exception as e:
        termination_reason = f"exception:{type(e).__name__}"
        print(f"[infer] EXCEPTION: {e!r}")
        traceback.print_exc()
    finally:
        try:
            logger.close(verdict="unset", notes="", reason=termination_reason)
        except Exception as e:
            print(f"[infer] WARNING: logger.close failed: {e!r}")
        if wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as e:
                print(f"[infer] WARN robot.disconnect failed: {e!r}")
        wall = time.perf_counter() - t_start
        print(f"[infer] done. steps={step} termination={termination_reason} "
              f"wall={wall:.2f}s")
        if isinstance(logger, RolloutLogger):
            print(f"[infer] logs at {logger.run_dir}")

    sys.exit(0 if termination_reason in ("success", "timeout", "user_abort") else 2)


def _push_reverse_pointer(
    *,
    training: dict,
    rollout_id: str,
    rollout_url: str,
    verdict: str,
) -> None:
    """Best-effort: append a {id, url, verdict} entry to the TRAINING run.

    Wrapped in try/except — no failure path blocks the rollout. Skipped
    automatically when training is None or wandb is offline (the caller
    enforces that).

    CONCURRENCY CAVEAT: read-modify-write is not race-safe. If two
    rollouts against the same training run finish within seconds of each
    other, the later .summary.update overwrites the earlier append.
    Acceptable under the current operating model (one operator, one robot
    at a time).
    """
    try:
        import wandb
        api = wandb.Api()
        entity = training.get("wandb_entity")
        project = training.get("wandb_project")
        rid = training.get("wandb_run_id")
        if not (entity and project and rid):
            return
        tr = api.run(f"{entity}/{project}/{rid}")
        prior = list(tr.summary.get("rollouts") or [])
        prior.append({"id": rollout_id, "url": rollout_url, "verdict": verdict})
        tr.summary.update({
            "rollouts": prior[-20:],
            "rollouts_total": len(prior),
            "last_rollout_url": rollout_url,
        })
    except Exception as e:
        print(f"[infer] reverse-pointer write skipped ({e!r})")


if __name__ == "__main__":
    main()
