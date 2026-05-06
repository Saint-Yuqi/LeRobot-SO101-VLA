"""Run a trained policy on the real SO-101 robot, with full telemetry.

Closed-loop inference: read camera + joints, query the policy, send
actions to the robot, repeat. Includes a hard time limit (per the eval
brief, you have 20 s per rollout) and writes a per-rollout dossier
(meta.json, steps.csv, chunks.jsonl, chunks.npz, episode.json,
outcome.json, optional frames/ + episode.mp4) so failed attempts can be
debugged after the fact.

Usage:
    # Local checkpoint dir
    python scripts/run_inference.py \\
        --checkpoint checkpoints/eval1/<run-id>/final \\
        --prompt "Put the banana in the blue colored bowl." \\
        --max-seconds 20

    # HF Hub repo id — auto-downloads all checkpoint files into the
    # HuggingFace cache the first time, then re-uses the cache.
    python scripts/run_inference.py \\
        --checkpoint PrajnaYang/so101-eval1-smolvla-v1 \\
        --prompt "Put the banana in the blue colored bowl." \\
        --max-seconds 20

Implementation detail: this file deliberately depends only on the
BaseVLA interface, NOT on SmolVLA specifically. Swapping in a
DecoupledPolicy later requires zero changes here.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.rollout_logger import NoopLogger, RolloutLogger
from src.inference.runner import run_rollout
from src.inference.safety import clamp_action
from src.utils.checkpoint_meta import checkpoint_short_id, resolve_training_run
from src.utils.run_metadata import capture_runtime_metadata


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_checkpoint(arg: str) -> tuple[str, str, str]:
    """Accept either a local checkpoint dir or a HF Hub repo id.

    Returns (local_path, source, origin) where:
      source is 'local' or 'hf'
      origin is the original argument (preserves HF repo id if used)
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


def _load_robot_yaml() -> dict:
    path = REPO_ROOT / "configs" / "robot" / "so101.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _sanitize_tag(prefix: str, body: str, max_body: int = 32) -> str:
    """Wandb tags must match a restricted charset and stay <= 64 chars.

    The colon in `prompt:` / `robot:` is NOT in the allowed set, so we
    use an underscore separator. Only the suffix is sanitized; the
    prefix stays intact for grep-prefix purposes.
    """
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", body)[:max_body]
    return f"{prefix}_{safe}"


def _identity_clamp(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """No-op clamp; mask shape MUST match clamp_action's: (action_dim,) uint8."""
    a32 = np.asarray(a, dtype=np.float32)
    return a32, np.zeros(a32.shape[-1], dtype=np.uint8)


def _resolve_clamp(
    args, robot_cfg: dict
) -> tuple[bool, str | None, callable, list[float], list[str]]:
    """Apply the CLI-vs-YAML precedence rule and build the clamp_fn.

    Returns:
        (clamp_effective, clamp_disabled_by, clamp_fn, joint_limits_flat, joint_names)
    `clamp_disabled_by` is `None` when clamp ran, else "yaml" / "cli" /
    "yaml+cli".
    """
    yaml_clamp = bool(robot_cfg.get("clamp_actions", True))
    cli_disabled = bool(getattr(args, "no_clamp", False))

    if yaml_clamp and not cli_disabled:
        effective = True
        disabled_by = None
    elif yaml_clamp and cli_disabled:
        effective = False
        disabled_by = "cli"
        print("[infer] WARNING: --no-clamp set; safety clamp DISABLED.")
    elif not yaml_clamp and not cli_disabled:
        effective = False
        disabled_by = "yaml"
        print("[infer] WARNING: clamp_actions=false in YAML; safety clamp DISABLED.")
    else:
        effective = False
        disabled_by = "yaml+cli"
        print("[infer] WARNING: clamp disabled in BOTH YAML and CLI.")

    joint_limits_dict = robot_cfg.get("joint_limits_deg") or {}
    joint_names = list(joint_limits_dict.keys())
    if effective:
        clamp_fn = functools.partial(
            clamp_action,
            joint_limits_deg=joint_limits_dict,
            joint_names=joint_names,
        )
    else:
        clamp_fn = _identity_clamp
    return effective, disabled_by, clamp_fn, joint_limits_dict, joint_names


def build_logger(args, meta_dict: dict, robot_cfg: dict) -> "RolloutLogger | NoopLogger":
    """Construct the logger; cheap (does NOT open wandb)."""
    if args.no_log:
        return NoopLogger()
    return RolloutLogger(
        log_dir=args.log_dir,
        inference_run_id=meta_dict["inference_run_id"],
        meta=meta_dict,
        action_dim=int(robot_cfg.get("action_dim", 6)),
        state_dim=int(robot_cfg.get("action_dim", 6)),
        control_hz=float(args.control_hz),
        chunk_size=int(meta_dict.get("policy_chunk_size", 50)),
        frame_every=int(args.frame_every),
        video=bool(args.video),
    )


def _resolve_verdict(args, termination_reason: str, default_notes: str) -> tuple[str, str]:
    """Implements the verdict dispatch table from the plan.

    Returns (verdict, notes).
    """
    notes = default_notes or ""
    mode = args.verdict
    tr = termination_reason

    if tr in ("policy_nan", "policy_fault", "robot_fault", "init_fault"):
        return "failure", notes
    if tr == "user_abort":
        return "abort", notes
    # tr in ("success", "timeout"):
    if mode == "never":
        return "unset", notes
    if mode == "always":
        if not sys.stdin.isatty():
            raise SystemExit(
                "[infer] --verdict always requires a TTY for the prompt"
            )
        return _prompt_verdict(notes)
    # mode == "auto"
    if sys.stdin.isatty():
        return _prompt_verdict(notes)
    return "unset", notes


def _prompt_verdict(default_notes: str) -> tuple[str, str]:
    print("\n[infer] Rollout finished. Choose a verdict:")
    print("  [s]uccess  [p]artial  [f]ailure  [a]bort  [u]nset (default)")
    try:
        choice = input("verdict> ").strip().lower()
    except EOFError:
        choice = ""
    mapping = {
        "s": "success", "success": "success",
        "p": "partial", "partial": "partial",
        "f": "failure", "failure": "failure",
        "a": "abort", "abort": "abort",
        "u": "unset", "unset": "unset",
        "": "unset",
    }
    verdict = mapping.get(choice, "unset")
    try:
        notes = input("notes> ").strip()
    except EOFError:
        notes = default_notes
    if not notes:
        notes = default_notes or ""
    return verdict, notes


class _DryRunRobot:
    """Synthetic robot for `--dry-run`. Keeps runner.py branch-free."""

    def __init__(self, action_dim: int = 6, camera_key: str = "main"):
        self._action_dim = action_dim
        self._camera_key = camera_key

    def connect(self) -> None:
        return

    def disconnect(self) -> None:
        return

    def capture_observation(self) -> dict:
        return {
            f"observation.images.{self._camera_key}":
                np.zeros((480, 640, 3), dtype=np.uint8),
            "observation.state": np.zeros(self._action_dim, dtype=np.float32),
        }

    def send_action(self, action) -> None:
        print(f"[infer] action: {np.round(np.asarray(action), 3)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local checkpoint dir OR HuggingFace repo id (e.g. user/repo).",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--policy-type", choices=["smolvla", "decoupled"],
                        default="smolvla")
    parser.add_argument("--camera-key", default="main",
                        help="Camera name used during teleop recording. "
                             "Eval 1 dataset uses 'main'.")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="print actions without sending to robot")

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

    parser.add_argument("--verdict", choices=["auto", "always", "never"],
                        default="auto",
                        help="verdict capture; gated by termination_reason")
    parser.add_argument("--notes", default="",
                        help="one-shot operator annotation, written into outcome.json")
    parser.add_argument("--no-clamp", action="store_true",
                        help="debug — disable safety clamp")

    args = parser.parse_args()

    ckpt_path, ckpt_source, ckpt_origin = resolve_checkpoint(args.checkpoint)
    robot_cfg = _load_robot_yaml()

    # ---- Identity / lineage ----
    runtime = capture_runtime_metadata()
    ckpt_short = checkpoint_short_id(args.checkpoint)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    inference_run_id = (
        f"{time.strftime('%Y%m%d-%H%M%S')}_job{job_id}_{ckpt_short}"
    )
    training = resolve_training_run(ckpt_path)

    clamp_effective, clamp_disabled_by, clamp_fn, joint_limits, joint_names = (
        _resolve_clamp(args, robot_cfg)
    )

    meta_dict: dict = {
        "inference_run_id": inference_run_id,
        **runtime,
        "cli_args": vars(args),
        "checkpoint_source": ckpt_source,
        "checkpoint_path": ckpt_path,
        "checkpoint_origin": ckpt_origin,
        "training_run": training,
        "robot_config_snapshot": {
            "control_hz": float(robot_cfg.get("control_hz", args.control_hz)),
            "action_dim": int(robot_cfg.get("action_dim", 6)),
            "joint_limits_deg": joint_limits,
            "effective_camera_key": args.camera_key,
        },
        "clamp_effective": clamp_effective,
        "clamp_disabled_by": clamp_disabled_by,
    }

    # ---- Wandb config sanity (--video w/ --no-log) ----
    if args.video and args.no_log:
        print("[infer] WARNING: --video is a no-op with --no-log "
              "(NoopLogger.maybe_save_frame returns ''); skipping mp4.")

    # ---- Policy + robot setup BEFORE logger so meta.json is complete.
    # If init fails before logger is built, we still want a directory on
    # disk so post-mortem tooling can find a meta.json. Strategy: build
    # the logger now with what we've got; it'll write meta.json with
    # whatever's in meta_dict, and we'll re-write meta.json after policy
    # load (path: logger.run_dir / "meta.json").
    logger = build_logger(args, meta_dict, robot_cfg)
    if isinstance(logger, RolloutLogger):
        print(f"[infer] inference_run_id: {inference_run_id}")
        print(f"[infer] log_dir: {logger.run_dir}")

    verdict, notes = "unset", args.notes
    termination_reason = "init_fault"
    wandb_run = None
    robot = None
    policy = None

    try:
        # Lazy imports
        if args.policy_type == "smolvla":
            from src.models.smolvla_wrapper import SmolVLAWrapper
            policy = SmolVLAWrapper.from_checkpoint(
                ckpt_path, camera_keys=(args.camera_key,)
            )
        else:
            from src.models.decoupled_policy import DecoupledPolicy
            policy = DecoupledPolicy.from_checkpoint(ckpt_path)

        try:
            policy = policy.to("cuda").eval()
        except Exception as e:
            # CPU fallback for smoke tests on no-GPU dev boxes.
            print(f"[infer] cuda not available ({e!r}); using CPU")
            policy = policy.to("cpu").eval()
        try:
            policy.reset()
        except Exception:
            pass
        try:
            param_count = int(policy.active_param_count)
            meta_dict["policy_active_param_count"] = param_count
            print(f"[infer] policy loaded. active params: {param_count:,}")
            # Re-write meta.json now that we know param count.
            if isinstance(logger, RolloutLogger):
                (logger.run_dir / "meta.json").write_text(
                    json.dumps(meta_dict, indent=2, default=str)
                )
        except Exception:
            pass

        # Robot connect
        if args.dry_run:
            robot = _DryRunRobot(
                action_dim=int(robot_cfg.get("action_dim", 6)),
                camera_key=args.camera_key,
            )
            print("[infer] DRY RUN — synthetic robot, no hardware.")
        else:
            from lerobot.common.robot_devices.robots.factory import make_robot  # type: ignore
            robot = make_robot("so101")
            robot.connect()

        # Wandb
        if not args.no_wandb:
            try:
                import wandb
                tags = [ckpt_short]
                tags.append(_sanitize_tag(
                    "robot", str(robot_cfg.get("robot_type", "unknown"))
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
                print(f"[infer] wandb: {getattr(wandb_run, 'url', None) or '(offline)'}")
            except Exception as e:
                print(f"[infer] WARNING: wandb.init failed ({e!r}); continuing offline")
                wandb_run = None

        logger.attach_wandb(wandb_run)

        # ---- Run! ----
        termination_reason = run_rollout(
            policy=policy,
            robot=robot,
            logger=logger,
            control_hz=float(args.control_hz),
            max_seconds=float(args.max_seconds),
            clamp_fn=clamp_fn,
            prompt=args.prompt,
            camera_key=args.camera_key,
        )
        print(f"[infer] termination_reason: {termination_reason}")
        verdict, notes = _resolve_verdict(args, termination_reason, notes)

        # Part C: reverse pointer on the training run.
        if (training is not None
                and wandb_run is not None
                and args.wandb_mode == "online"):
            _push_reverse_pointer(
                training=training,
                rollout_id=inference_run_id,
                rollout_url=getattr(wandb_run, "url", "") or "",
                verdict=verdict,
            )
    finally:
        try:
            logger.close(verdict=verdict, notes=notes, reason=termination_reason)
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
            except Exception:
                pass

    sys.exit(0 if termination_reason in ("success", "timeout") else 2)


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
    rollouts against the same training run finish within seconds of
    each other, the later .summary.update overwrites the earlier
    append. Acceptable under the current operating model (one operator,
    one robot at a time).
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
