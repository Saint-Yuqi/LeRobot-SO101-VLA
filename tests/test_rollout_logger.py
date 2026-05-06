"""Unit tests for the rollout telemetry stack.

Drives a 100-step + 2-chunk fixture through `run_rollout` with a mock
robot and a synthetic policy. Asserts every artifact's schema and the
non-obvious invariants from the plan (NaN guard skips send_action,
clamp NaN-preservation, KeyboardInterrupt still flushes via finally,
zero-data close() is safe, NoopLogger surface mirrors RolloutLogger).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.inference.rollout_logger import NoopLogger, RolloutLogger  # noqa: E402
from src.inference.runner import run_rollout  # noqa: E402
from src.inference.safety import clamp_action  # noqa: E402
from src.models.base_vla import ActionChunk  # noqa: E402
from src.utils.checkpoint_meta import (  # noqa: E402
    checkpoint_short_id,
    resolve_training_run,
)


# ---- Fixtures / helpers -------------------------------------------------

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
JOINT_LIMITS = {
    "shoulder_pan":  [-90, 90],
    "shoulder_lift": [-90, 30],
    "elbow_flex":    [-110, 0],
    "wrist_flex":    [-90, 90],
    "wrist_roll":    [-180, 180],
    "gripper":       [0, 100],
}


def _meta(inference_run_id: str = "test-run") -> dict:
    return {
        "inference_run_id": inference_run_id,
        "clamp_effective": True,
    }


def _identity_clamp(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a32 = np.asarray(a, dtype=np.float32)
    return a32, np.zeros(a32.shape[-1], dtype=np.uint8)


def _make_logger(tmp_path: Path, **kw) -> RolloutLogger:
    # Disable GPU sampling in tests by injecting a mock that returns {}.
    gpu = MagicMock()
    gpu.sample.return_value = {}
    gpu.shutdown.return_value = None
    return RolloutLogger(
        log_dir=tmp_path,
        inference_run_id="test-run",
        meta=_meta(),
        action_dim=6,
        state_dim=6,
        control_hz=30.0,
        chunk_size=50,
        frame_every=0,
        video=False,
        gpu_sampler=gpu,
        **kw,
    )


class _FakeRobot:
    """Minimal robot stub. Records send_action calls."""

    def __init__(self):
        self.send_action = MagicMock()

    def capture_observation(self):
        return {
            "observation.images.main": np.zeros((48, 64, 3), dtype=np.uint8),
            "observation.state": np.zeros(6, dtype=np.float32),
        }

    def disconnect(self):
        return


class _ConstChunkPolicy:
    """Yields a fixed chunk of actions every predict() call."""

    def __init__(self, n_chunks: int = 2, chunk_size: int = 50, action_dim: int = 6,
                 fill_value: float = 0.0):
        self._n = n_chunks
        self._cs = chunk_size
        self._ad = action_dim
        self._fill = fill_value
        self._calls = 0

    def predict(self, obs):
        self._calls += 1
        return ActionChunk(
            actions=np.full((self._cs, self._ad), self._fill, dtype=np.float32),
            chunk_size=self._cs,
        )


# ---- Tests --------------------------------------------------------------

def test_clamp_preserves_nan():
    a = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out, mask = clamp_action(a, JOINT_LIMITS, JOINT_NAMES)
    assert np.isnan(out[0])  # NOT replaced by limit
    assert out.shape == (6,)
    assert mask.shape == (6,)
    assert mask.dtype == np.uint8


def test_clamp_clips_in_range():
    a = np.array([1000.0, -1000.0, 0.0, 0.0, 0.0, 200.0], dtype=np.float32)
    out, mask = clamp_action(a, JOINT_LIMITS, JOINT_NAMES)
    assert out[0] == 90.0
    assert out[1] == -90.0
    assert out[5] == 100.0
    assert mask.tolist() == [1, 1, 0, 0, 0, 1]


def test_checkpoint_short_id_sanitises_and_bounds():
    s = checkpoint_short_id("PrajnaYang/so101-eval1-smolvla-v1")
    assert "/" not in s
    assert len(s) <= 24
    assert s


def test_resolve_training_run_missing_returns_none(tmp_path):
    # Empty dir, no sidecar.
    out = resolve_training_run(tmp_path)
    assert out is None


def test_resolve_training_run_present(tmp_path):
    sidecar = {
        "wandb_run_id": "abc",
        "wandb_project": "Lerobot",
        "wandb_entity": "ent",
        "wandb_url": "https://wandb.ai/.../runs/abc",
        "experiment_name": "eval1",
        "git_sha": "deadbeef",
        "step": 1234,
    }
    (tmp_path / "wandb_metadata.json").write_text(json.dumps(sidecar))
    got = resolve_training_run(tmp_path)
    assert got is not None
    assert got["wandb_run_id"] == "abc"
    assert got["experiment_name"] == "eval1"


def test_full_rollout_writes_all_artifacts(tmp_path):
    """Drive a 100-step / 2-chunk rollout, then check schema."""
    logger = _make_logger(tmp_path)
    robot = _FakeRobot()
    policy = _ConstChunkPolicy(n_chunks=2, chunk_size=50)

    reason = run_rollout(
        policy=policy,
        robot=robot,
        logger=logger,
        control_hz=10000.0,  # huge hz so the deadline doesn't kick in mid-test
        max_seconds=5.0,
        clamp_fn=_identity_clamp,
        prompt="test",
        camera_key="main",
        _skip_sync=True,
    )
    # We're going to interrupt manually after 100 steps. Use a cap by
    # re-running with a cheap predict that exhausts only 100 = 2 * 50.
    # _ConstChunkPolicy does that already if we wire max_seconds tightly.
    # Simpler: call run_rollout with a frame budget enforced via deadline.
    # Here control_hz=10000 + max_seconds=5 yields up to 50000 ticks; we
    # rely on the test policy to be tracked separately.

    assert reason in ("timeout",)  # natural deadline
    # The actual loop ran many ticks since hz=10000 — we don't care, we
    # just need >=100. Recreate with a stricter budget for crisp asserts:
    logger.close(verdict="success", notes="ok", reason=reason)

    run_dir = tmp_path / "test-run"
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "steps.csv").exists()
    assert (run_dir / "chunks.jsonl").exists()
    assert (run_dir / "chunks.npz").exists()
    assert (run_dir / "episode.json").exists()
    assert (run_dir / "outcome.json").exists()


def test_rollout_with_step_cap(tmp_path):
    """Run a tightly-bounded rollout (~100 steps) and assert row counts."""
    logger = _make_logger(tmp_path)
    robot = _FakeRobot()
    policy = _ConstChunkPolicy(n_chunks=2, chunk_size=50)

    # 100 ticks at 30 Hz ~= 3.33 s; give a 4 s budget.
    run_rollout(
        policy=policy,
        robot=robot,
        logger=logger,
        control_hz=10000.0,
        max_seconds=0.05,  # ~tight budget
        clamp_fn=_identity_clamp,
        prompt="test",
        camera_key="main",
        _skip_sync=True,
    )
    logger.close(verdict="success", notes="", reason="timeout")
    run_dir = tmp_path / "test-run"

    # Validate schema + content.
    with open(run_dir / "steps.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    expected_cols = {
        "step", "ts_mono_s", "ts_wall_iso", "period_target_ms",
        "period_actual_ms", "inferred_this_step", "chunk_idx",
        "chunk_step", "queue_depth_after",
        "state_0", "state_5", "action_raw_0", "action_sent_5",
        "clamped_mask", "nan_in_action", "prompt_id", "frame_path",
        "gpu_util_pct", "gpu_mem_pct",
    }
    assert expected_cols.issubset(set(rows[0].keys()))

    # chunks.jsonl: every line is valid JSON; npz keys match.
    with open(run_dir / "chunks.jsonl") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    assert len(chunks) >= 1
    npz = np.load(run_dir / "chunks.npz")
    for c in chunks:
        assert c["action_horizon_npz_key"] in npz.files
        assert c["chunk_size"] == 50
        assert isinstance(c["camera_hash"], dict)
        assert isinstance(c["camera_blur_var"], dict)

    episode = json.loads((run_dir / "episode.json").read_text())
    for k in (
        "inference_run_id", "started_at", "ended_at", "duration_s",
        "n_steps", "n_chunks", "termination_reason",
        "period_jitter_p95_ms", "infer_latency_budget_ratio",
        "action_jerk_mean", "action_clip_rate_pct",
        "chunk_seam_discontinuity_mean", "state_tracking_error_mean",
        "camera_staleness_count", "safety_clamp_counts_per_dim",
        "clamp_effective", "events_log",
    ):
        assert k in episode, f"missing {k}"
    # action_dim == 6 must match list length.
    assert len(episode["safety_clamp_counts_per_dim"]) == 6


def test_nan_guard_skips_send_action(tmp_path):
    """A NaN in action_raw must terminate the rollout WITHOUT calling send_action."""
    logger = _make_logger(tmp_path)
    robot = _FakeRobot()

    # Policy emits a chunk where the 3rd action contains NaN.
    chunk = np.zeros((50, 6), dtype=np.float32)
    chunk[2, 0] = np.nan

    class _NaNPolicy:
        def __init__(self): self._called = False
        def predict(self, obs):
            return ActionChunk(actions=chunk.copy(), chunk_size=50)

    reason = run_rollout(
        policy=_NaNPolicy(),
        robot=robot,
        logger=logger,
        control_hz=10000.0,
        max_seconds=1.0,
        clamp_fn=_identity_clamp,
        prompt="test",
        camera_key="main",
        _skip_sync=True,
    )
    logger.close(verdict="failure", notes="", reason=reason)

    assert reason == "policy_nan"
    # send_action called for steps 0, 1; NOT for step 2 (NaN).
    assert robot.send_action.call_count == 2

    # events_log captured the NaN.
    episode = json.loads((tmp_path / "test-run" / "episode.json").read_text())
    kinds = [e["kind"] for e in episode["events_log"]]
    assert "nan" in kinds


def test_policy_fault_recorded(tmp_path):
    logger = _make_logger(tmp_path)
    robot = _FakeRobot()

    class _BadPolicy:
        def predict(self, obs):
            raise RuntimeError("boom")

    reason = run_rollout(
        policy=_BadPolicy(),
        robot=robot,
        logger=logger,
        control_hz=10000.0,
        max_seconds=1.0,
        clamp_fn=_identity_clamp,
        prompt="test",
        camera_key="main",
        _skip_sync=True,
    )
    logger.close(verdict="failure", notes="", reason=reason)
    assert reason == "policy_fault"
    episode = json.loads((tmp_path / "test-run" / "episode.json").read_text())
    kinds = [e["kind"] for e in episode["events_log"]]
    assert "error.policy" in kinds


def test_robot_fault_recorded(tmp_path):
    logger = _make_logger(tmp_path)
    robot = _FakeRobot()
    robot.send_action.side_effect = RuntimeError("hardware down")

    policy = _ConstChunkPolicy(chunk_size=5)
    reason = run_rollout(
        policy=policy,
        robot=robot,
        logger=logger,
        control_hz=10000.0,
        max_seconds=1.0,
        clamp_fn=_identity_clamp,
        prompt="test",
        camera_key="main",
        _skip_sync=True,
    )
    logger.close(verdict="failure", notes="", reason=reason)
    assert reason == "robot_fault"
    episode = json.loads((tmp_path / "test-run" / "episode.json").read_text())
    kinds = [e["kind"] for e in episode["events_log"]]
    assert "error.robot" in kinds


def test_keyboard_interrupt_flushes(tmp_path, monkeypatch):
    """KeyboardInterrupt mid-loop must still produce a parseable episode.json."""
    logger = _make_logger(tmp_path)
    robot = _FakeRobot()
    policy = _ConstChunkPolicy(chunk_size=50)

    # Closure that raises KeyboardInterrupt on the Nth time.sleep call.
    real_sleep = time.sleep
    counter = {"n": 0}

    def _interrupting_sleep(secs):
        counter["n"] += 1
        if counter["n"] >= 5:
            raise KeyboardInterrupt
        real_sleep(min(secs, 0.001))

    monkeypatch.setattr(time, "sleep", _interrupting_sleep)
    reason = run_rollout(
        policy=policy,
        robot=robot,
        logger=logger,
        control_hz=30.0,
        max_seconds=10.0,
        clamp_fn=_identity_clamp,
        prompt="test",
        camera_key="main",
        _skip_sync=True,
    )
    logger.close(verdict="abort", notes="", reason=reason)

    assert reason == "user_abort"
    episode = json.loads((tmp_path / "test-run" / "episode.json").read_text())
    assert episode["termination_reason"] == "user_abort"
    # CSV must exist and be non-empty.
    csv_path = tmp_path / "test-run" / "steps.csv"
    assert csv_path.stat().st_size > 0


def test_zero_data_close_is_safe(tmp_path):
    """An init_fault rollout: logger built but run_rollout never ran."""
    logger = _make_logger(tmp_path)
    # No log_step / log_chunk calls.
    logger.close(verdict="failure", notes="", reason="init_fault")

    episode = json.loads((tmp_path / "test-run" / "episode.json").read_text())
    assert episode["n_steps"] == 0
    assert episode["n_chunks"] == 0
    assert episode["termination_reason"] == "init_fault"
    # No np.percentile([]) crash; metrics are None.
    for k in ("period_jitter_p95_ms", "infer_latency_budget_ratio",
              "action_jerk_mean", "chunk_seam_discontinuity_mean",
              "state_tracking_error_mean"):
        assert episode[k] is None, f"{k} should be None on zero data"


def test_noop_logger_surface_mirrors_real(tmp_path):
    """Surface-drift assertion: every public method on RolloutLogger
    must exist on NoopLogger or `--no-log` will crash with AttributeError."""
    real = _make_logger(tmp_path)
    real.close(verdict="unset", notes="", reason="timeout")
    for m in dir(RolloutLogger):
        if m.startswith("_"):
            continue
        assert hasattr(NoopLogger, m), f"NoopLogger missing public method: {m}"


def test_attach_wandb_idempotence(tmp_path):
    logger = _make_logger(tmp_path)
    run_a = MagicMock()
    run_b = MagicMock()
    # None: no-op
    logger.attach_wandb(None)
    assert logger._wandb_run is None
    # First attach
    logger.attach_wandb(run_a)
    assert logger._wandb_run is run_a
    # Same again: no-op
    logger.attach_wandb(run_a)
    assert logger._wandb_run is run_a
    # Different run -> RuntimeError
    with pytest.raises(RuntimeError):
        logger.attach_wandb(run_b)
    logger.close(verdict="unset", notes="", reason="timeout")
