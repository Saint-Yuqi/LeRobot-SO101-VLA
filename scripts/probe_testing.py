"""Probe a VLA checkpoint with one grasped-banana frame from task2.

Goal: take a single post-grasp frame, swap the language prompt, and inspect
how the predicted 50-step `shoulder_pan` trajectory changes per prompt.

This is meant as a lightweight offline probe for questions like:
  - does the model react differently to blue vs red vs green prompts?
  - does it pan in opposite directions for left/right targets?
  - is the prompt ignored after the banana is already grasped?

Examples
--------
    conda activate lerobot

    python scripts/probe_testing_task2.py \
        --checkpoint ethrl2026/so101-eval3-smolvla-v4 \
        --dataset ethrl2026/task2_20260509_stage2_random_lighting_augmented_2160

    python scripts/probe_testing_task2.py \
        --checkpoint ethrl2026/so101-eval3-smolvla-v4 \
        --dataset ethrl2026/task2_20260509_stage2_random_lighting_augmented_2160 \
        --episode 12 \
        --prompt "Put the banana in the blue colored bowl." \
        --prompt "Put the banana in the red colored bowl." \
        --prompt "Put the banana in the green colored bowl."
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import snapshot_download

from src.data.phase_labels import (
    DEFAULT_POST_CLOSE_MARGIN,
    compute_phase_labels,
)
from src.models.base_vla import BaseVLA, Observation
from src.models.flower_vla_wrapper import FlowerVLAWrapper
from src.models.smolvla_wrapper import SmolVLAWrapper


DEFAULT_PROMPTS = (
    "Put the banana in the blue colored bowl.",
    "Put the banana in the red colored bowl.",
    "Put the banana in the green colored bowl.",
)
DEFAULT_STATE_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _read_checkpoint_config(checkpoint_path: str) -> dict[str, Any]:
    config_path = Path(checkpoint_path) / "config.json"
    try:
        return json.loads(config_path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to read checkpoint config at {config_path}") from e


def _detect_policy_type(checkpoint_path: str) -> str:
    cfg = _read_checkpoint_config(checkpoint_path)
    policy_type = cfg.get("type")
    if isinstance(policy_type, str) and policy_type:
        if "smolvla" in policy_type.lower():
            return "smolvla"
        if "flower" in policy_type.lower():
            return "flower"
    if "vlm_path" in cfg and "dit_dim" in cfg:
        return "flower"
    return "smolvla"


def _resolve_checkpoint(arg: str) -> str:
    path = Path(arg)
    if path.is_dir() and (path / "config.json").exists():
        return str(path)
    if path.exists():
        raise SystemExit(
            f"--checkpoint '{arg}' exists but is not a checkpoint dir (missing config.json)."
        )
    if "/" not in arg or arg.count("/") > 1:
        raise SystemExit(
            f"--checkpoint '{arg}' is neither a local checkpoint dir nor an HF repo id."
        )
    print(f"[probe] downloading checkpoint {arg} from Hugging Face Hub...")
    local = snapshot_download(repo_id=arg, repo_type="model")
    print(f"[probe] checkpoint cached at {local}")
    return local


def _resolve_dataset_root(repo_id: str, revision: str) -> Path:
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"datasets--{repo_id.replace('/', '--')}"
        / "snapshots"
    )
    if cache_root.exists():
        snaps = sorted(cache_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snaps:
            if (snap / "meta" / "info.json").exists():
                return snap

    print(f"[probe] downloading dataset metadata/parquets for {repo_id}...")
    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["meta/**", "data/**"],
    )
    print(f"[probe] dataset snapshot cached at {local}")
    return Path(local)


def _read_dataset_info(root: Path) -> dict[str, Any]:
    with (root / "meta" / "info.json").open() as fh:
        return json.load(fh)


def _load_task_lookup(root: Path) -> dict[int, str]:
    table = pq.read_table(root / "meta" / "tasks.parquet", columns=["task_index", "task"])
    cols = table.to_pydict()
    return {
        int(task_idx): str(task)
        for task_idx, task in zip(cols["task_index"], cols["task"])
    }


def _load_episode_row(root: Path, episode: int) -> dict[str, Any]:
    files = sorted((root / "meta" / "episodes").rglob("file-*.parquet"))
    for file in files:
        table = pq.read_table(file)
        df = table.to_pandas()
        row_df = df[df["episode_index"] == int(episode)]
        if len(row_df) == 0:
            continue
        row = row_df.iloc[0].to_dict()
        return {str(k): v for k, v in row.items()}
    raise ValueError(f"Episode {episode} not found under {root / 'meta' / 'episodes'}")


def _load_episode_data_rows(root: Path, episode_row: dict[str, Any]) -> dict[str, list[Any]]:
    data_path = root / (
        f"data/chunk-{int(episode_row['data/chunk_index']):03d}/"
        f"file-{int(episode_row['data/file_index']):03d}.parquet"
    )
    table = pq.read_table(
        data_path,
        columns=[
            "episode_index",
            "frame_index",
            "timestamp",
            "observation.state",
            "action",
            "task_index",
        ],
    )
    cols = table.to_pydict()
    keep = [
        i
        for i, ep in enumerate(cols["episode_index"])
        if int(ep) == int(episode_row["episode_index"])
    ]
    return {
        "frame_index": [int(cols["frame_index"][i]) for i in keep],
        "timestamp": [float(cols["timestamp"][i]) for i in keep],
        "observation.state": [np.asarray(cols["observation.state"][i], dtype=np.float32) for i in keep],
        "action": [np.asarray(cols["action"][i], dtype=np.float32) for i in keep],
        "task_index": [int(cols["task_index"][i]) for i in keep],
    }


def _ensure_video_file(
    *,
    repo_id: str,
    revision: str,
    root: Path,
    relative_path: str,
) -> Path:
    video_path = root / relative_path
    if video_path.exists():
        return video_path
    print(f"[probe] downloading video file {relative_path} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[relative_path],
    )
    if not video_path.exists():
        raise FileNotFoundError(f"Expected downloaded video at {video_path}")
    return video_path


def _read_video_frame(video_path: Path, frame_number: int) -> np.ndarray:
    import imageio.v3 as iio

    frame = iio.imread(video_path, index=int(frame_number))
    if frame.ndim != 3:
        raise ValueError(f"Decoded video frame has unexpected shape {frame.shape}")
    return np.asarray(frame, dtype=np.uint8)


def _video_frame_index_for_sample(
    *,
    episode_row: dict[str, Any],
    video_key_full: str,
    timestamp_s: float,
    selected_frame: int,
    fps: float,
) -> tuple[int, str]:
    """Resolve which frame to decode from the episode video.

    Historical datasets are inconsistent: some store
    `videos/<key>/from_timestamp`, others only expose episode-local frame
    indices. Prefer timestamp-based alignment when the metadata exists; fall
    back to the selected episode frame index otherwise.
    """
    from_ts_key = f"videos/{video_key_full}/from_timestamp"
    if from_ts_key in episode_row and episode_row[from_ts_key] is not None:
        video_start_ts = float(episode_row[from_ts_key])
        absolute_video_ts = video_start_ts + float(timestamp_s)
        return int(round(absolute_video_ts * fps)), "timestamp"
    return int(selected_frame), "frame_index"


def _extract_feature_names(meta_feature: Any) -> list[str] | None:
    if meta_feature is None:
        return None
    if isinstance(meta_feature, dict):
        names = meta_feature.get("names")
    else:
        names = getattr(meta_feature, "names", None)
    if not names:
        return None
    return [str(x) for x in names]


def _meta_feature(meta: Any, key: str) -> Any:
    feats = getattr(meta, "features", None) or {}
    if isinstance(feats, dict):
        return feats.get(key)
    getter = getattr(feats, "get", None)
    if callable(getter):
        return getter(key)
    try:
        return feats[key]
    except Exception:
        return None


def _infer_image_key(meta) -> str:
    feats = getattr(meta, "features", None) or {}
    if isinstance(feats, dict):
        keys = feats.keys()
    else:
        keys = getattr(feats, "keys", lambda: [])()
    image_keys = sorted(
        k.removeprefix("observation.images.")
        for k in keys
        if isinstance(k, str) and k.startswith("observation.images.")
    )
    if not image_keys:
        raise RuntimeError("No observation.images.* feature found in dataset metadata.")
    return image_keys[0]


def _joint_index(names: list[str], joint_name: str) -> int:
    wanted = {joint_name, f"{joint_name}.pos"}
    for idx, name in enumerate(names):
        if name in wanted:
            return idx
    raise ValueError(f"Joint {joint_name!r} not found in names={names}")


def _to_uint8_rgb(img_t) -> np.ndarray:
    if hasattr(img_t, "detach"):
        arr = img_t.detach().cpu().numpy()
    else:
        arr = np.asarray(img_t)
    if arr.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape {arr.shape}")
    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _auto_episode(repo_id: str, root: Path, revision: str) -> tuple[int, int]:
    phase = compute_phase_labels(repo_id=repo_id, root=root, revision=revision)
    for episode in sorted(phase.grasp_frames):
        grasp_t = phase.grasp_frames[episode]
        if grasp_t is not None:
            return int(episode), int(grasp_t)
    raise RuntimeError(f"No stable grasp frame found in dataset {repo_id}")


def _select_probe_frame(
    *,
    repo_id: str,
    root: Path,
    revision: str,
    episode: int | None,
    frame: int | None,
    grasp_offset: int,
    chunk_size: int,
) -> tuple[int, int, int | None]:
    if episode is None:
        auto_ep, auto_grasp = _auto_episode(repo_id, root, revision)
        episode = auto_ep
        grasp_t = auto_grasp
    else:
        phase = compute_phase_labels(
            repo_id=repo_id,
            root=root,
            episodes=[episode],
            revision=revision,
        )
        grasp_t = phase.grasp_frames.get(int(episode))

    if frame is not None:
        return int(episode), int(frame), None if grasp_t is None else int(grasp_t)

    if grasp_t is None:
        raise RuntimeError(
            f"Episode {episode} has no detected stable grasp frame; pass --frame manually."
        )
    return int(episode), int(grasp_t + grasp_offset), int(grasp_t)


def _trajectory_summary(
    values: np.ndarray,
    current_value: float,
    direction_threshold: float,
    positive_pan_means: str | None,
) -> dict[str, Any]:
    delta = values - float(current_value)
    peak_abs = float(np.max(np.abs(delta)))
    final_delta = float(delta[-1])
    if abs(final_delta) < direction_threshold and peak_abs < direction_threshold:
        trend = "stable"
    elif final_delta > 0:
        trend = "positive_pan"
    else:
        trend = "negative_pan"

    out: dict[str, Any] = {
        "final_value": float(values[-1]),
        "final_delta_from_current": final_delta,
        "mean_delta_from_current": float(delta.mean()),
        "min_delta_from_current": float(delta.min()),
        "max_delta_from_current": float(delta.max()),
        "peak_to_peak": float(values.max() - values.min()),
        "peak_abs_delta_from_current": peak_abs,
        "direction_sign": trend,
    }
    if positive_pan_means is not None and trend != "stable":
        if positive_pan_means == "left":
            out["robot_side_guess"] = "left" if trend == "positive_pan" else "right"
        else:
            out["robot_side_guess"] = "right" if trend == "positive_pan" else "left"
    return out


def _cos_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float32).reshape(-1)
    b_flat = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def _rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(diff) / np.sqrt(max(diff.size, 1)))


def _first_step_at_or_above(values: np.ndarray, threshold: float) -> int | None:
    if threshold <= 0:
        return 1
    idxs = np.where(np.asarray(values, dtype=np.float32) >= float(threshold))[0]
    if len(idxs) == 0:
        return None
    return int(idxs[0]) + 1  # 1-based chunk step for readability


def _slice_from_step(arr: np.ndarray, start_step_1based: int) -> np.ndarray:
    start_idx = max(0, int(start_step_1based) - 1)
    start_idx = min(start_idx, max(len(arr) - 1, 0))
    return arr[start_idx:]


def _pairwise_metrics(
    *,
    chunk_a: np.ndarray,
    chunk_b: np.ndarray,
    joint_a: np.ndarray,
    joint_b: np.ndarray,
    late_start_step: int,
    joint_diverge_threshold: float,
    chunk_diverge_threshold: float,
) -> dict[str, Any]:
    late_chunk_a = _slice_from_step(chunk_a, late_start_step)
    late_chunk_b = _slice_from_step(chunk_b, late_start_step)
    late_joint_a = _slice_from_step(joint_a, late_start_step)
    late_joint_b = _slice_from_step(joint_b, late_start_step)

    per_step_joint_abs = np.abs(joint_a - joint_b)
    per_step_chunk_rms = np.linalg.norm(chunk_a - chunk_b, axis=1) / np.sqrt(chunk_a.shape[1])

    return {
        "joint_full_cos": _cos_similarity(joint_a, joint_b),
        "joint_late_cos": _cos_similarity(late_joint_a, late_joint_b),
        "joint_full_rms": _rms_distance(joint_a, joint_b),
        "joint_late_rms": _rms_distance(late_joint_a, late_joint_b),
        "chunk_full_cos": _cos_similarity(chunk_a, chunk_b),
        "chunk_late_cos": _cos_similarity(late_chunk_a, late_chunk_b),
        "chunk_full_rms": _rms_distance(chunk_a, chunk_b),
        "chunk_late_rms": _rms_distance(late_chunk_a, late_chunk_b),
        "max_joint_abs_delta": float(per_step_joint_abs.max()),
        "max_chunk_step_rms": float(per_step_chunk_rms.max()),
        "first_joint_diverge_step": _first_step_at_or_above(
            per_step_joint_abs,
            joint_diverge_threshold,
        ),
        "first_chunk_diverge_step": _first_step_at_or_above(
            per_step_chunk_rms,
            chunk_diverge_threshold,
        ),
    }


def _plot_probe(
    output_path: Path,
    *,
    selected_image: np.ndarray,
    joint_name: str,
    current_value: float,
    recorded_trajectory: np.ndarray,
    results: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    late_start_step: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        gridspec_kw={"height_ratios": [3.0, 1.6]},
    )
    axes[0, 0].imshow(selected_image)
    axes[0, 0].set_title("Probe frame")
    axes[0, 0].axis("off")

    xs = np.arange(1, len(results[0]["joint_trajectory"]) + 1)
    ax_traj = axes[0, 1]
    ax_traj.axhline(
        current_value,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"current {joint_name}",
    )
    ax_traj.axvspan(
        late_start_step - 0.5,
        xs[-1] + 0.5,
        color="gold",
        alpha=0.12,
        label=f"late chunk ({late_start_step}+)",
    )
    ax_traj.plot(
        xs,
        recorded_trajectory,
        color="0.5",
        linestyle="--",
        linewidth=1.5,
        label="recorded action",
    )
    for row in results:
        ax_traj.plot(xs, row["joint_trajectory"], linewidth=2.0, label=row["prompt_label"])
    ax_traj.set_xlabel("chunk step")
    ax_traj.set_ylabel(joint_name)
    ax_traj.set_title(f"Predicted {len(xs)}-step {joint_name} trajectory")
    ax_traj.grid(alpha=0.3)
    ax_traj.legend(fontsize=8)

    ax_delta = axes[1, 0]
    for row in results:
        ax_delta.plot(xs, row["delta_from_current"], linewidth=2.0, label=row["prompt_label"])
    ax_delta.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_delta.axvspan(
        late_start_step - 0.5,
        xs[-1] + 0.5,
        color="gold",
        alpha=0.12,
    )
    ax_delta.set_xlabel("chunk step")
    ax_delta.set_ylabel(f"{joint_name} delta")
    ax_delta.set_title("Delta from current joint state")
    ax_delta.grid(alpha=0.3)

    ax_text = axes[1, 1]
    ax_text.axis("off")
    lines = [
        "Pairwise prompt diagnostics",
        f"late chunk starts at step {late_start_step}",
        "",
    ]
    for row in pairwise_rows[:8]:
        lines.append(
            f"{row['prompt_a_label']} vs {row['prompt_b_label']}: "
            f"late cos={row['chunk_late_cos']:.3f}, "
            f"late rms={row['chunk_late_rms']:.3f}, "
            f"joint div={row['first_joint_diverge_step']}, "
            f"chunk div={row['first_chunk_diverge_step']}"
        )
    ax_text.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_csv(output_path: Path, results: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["chunk_step", "prompt_label", "prompt_role", "joint_value", "delta_from_current"])
        for row in results:
            for step_idx, (val, delta) in enumerate(
                zip(row["joint_trajectory"], row["delta_from_current"]),
                start=1,
            ):
                writer.writerow([step_idx, row["prompt_label"], row["prompt_role"], val, delta])


def _write_pairwise_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_a_label",
        "prompt_b_label",
        "prompt_a_role",
        "prompt_b_role",
        "joint_full_cos",
        "joint_late_cos",
        "joint_full_rms",
        "joint_late_rms",
        "chunk_full_cos",
        "chunk_late_cos",
        "chunk_full_rms",
        "chunk_late_rms",
        "max_joint_abs_delta",
        "max_chunk_step_rms",
        "first_joint_diverge_step",
        "first_chunk_diverge_step",
    ]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _load_policy(args, *, checkpoint_path: str, image_key: str, device: str) -> BaseVLA:
    policy_type = (
        _detect_policy_type(checkpoint_path)
        if args.policy_type == "auto"
        else args.policy_type
    )
    common = {
        "camera_keys": (image_key,),
        "chunk_size": args.chunk_size,
        "device": device,
    }
    if policy_type == "smolvla":
        wrapper = SmolVLAWrapper.from_checkpoint(checkpoint_path, **common)
        setattr(wrapper, "_probe_policy_type", policy_type)
        return wrapper.to(device).eval()
    if policy_type == "flower":
        wrapper = FlowerVLAWrapper.from_checkpoint(
            checkpoint_path,
            module_name=args.flower_module,
            class_name=args.flower_class,
            module_path=args.flower_pythonpath,
            **common,
        )
        setattr(wrapper, "_probe_policy_type", policy_type)
        return wrapper.to(device).eval()
    raise ValueError(f"Unsupported policy type {policy_type!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="ethrl2026/so101-eval3-smolvla-v4",
        help="Local checkpoint dir or Hugging Face model repo id.",
    )
    parser.add_argument(
        "--policy-type",
        choices=["auto", "smolvla", "flower"],
        default="auto",
        help="Which policy wrapper to use for the checkpoint. Default: auto-detect from config.json.",
    )
    parser.add_argument(
        "--dataset",
        default="ethrl2026/task2_20260509_stage2_random_lighting_augmented_2160",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument("--revision", default="v3.0", help="Dataset revision.")
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to probe. Default: first episode with a detected grasp.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame index within the selected episode. Overrides auto post-grasp selection.",
    )
    parser.add_argument(
        "--grasp-offset",
        type=int,
        default=DEFAULT_POST_CLOSE_MARGIN,
        help="If --frame is omitted, probe at grasp_frame + grasp_offset.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="How many predicted action steps to inspect.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to probe. Repeat this flag to test multiple prompts.",
    )
    parser.add_argument(
        "--control-prompt",
        action="append",
        default=[],
        help="Optional control prompt(s), e.g. nonsense or mismatched prompts.",
    )
    parser.add_argument(
        "--include-recorded-prompt",
        action="store_true",
        help="Also probe the prompt recorded in the dataset for this frame.",
    )
    parser.add_argument(
        "--include-empty-prompt",
        action="store_true",
        help="Also probe an empty-string prompt as a control.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto | cuda | cpu | mps",
    )
    parser.add_argument(
        "--flower-module",
        default="src.flower.policy",
        help="Python module that defines FlowerVLAPolicy when --policy-type=flower.",
    )
    parser.add_argument(
        "--flower-class",
        default="FlowerVLAPolicy",
        help="Class name to load from --flower-module when --policy-type=flower.",
    )
    parser.add_argument(
        "--flower-pythonpath",
        default=None,
        help="Optional repo root to prepend to sys.path before importing the Flower module.",
    )
    parser.add_argument(
        "--image-key",
        default="auto",
        help="Dataset image key, without the observation.images. prefix. Default: auto.",
    )
    parser.add_argument(
        "--joint-name",
        default="shoulder_pan",
        help="Joint to summarize. Default: shoulder_pan.",
    )
    parser.add_argument(
        "--direction-threshold",
        type=float,
        default=3.0,
        help="Absolute delta threshold for calling the trajectory stable.",
    )
    parser.add_argument(
        "--late-start-step",
        type=int,
        default=31,
        help="1-based chunk step where the 'late chunk' diagnostic starts.",
    )
    parser.add_argument(
        "--joint-diverge-threshold",
        type=float,
        default=2.0,
        help="Absolute chosen-joint delta that counts as prompt divergence.",
    )
    parser.add_argument(
        "--chunk-diverge-threshold",
        type=float,
        default=1.0,
        help="Per-step all-joint RMS delta that counts as prompt divergence.",
    )
    parser.add_argument(
        "--positive-pan-means",
        choices=["left", "right"],
        default=None,
        help="Optional physical interpretation of positive shoulder_pan deltas.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/probes/task2_shoulder_pan",
        help="Where to write JSON/CSV/PNG outputs.",
    )
    args = parser.parse_args()

    prompts = args.prompt or list(DEFAULT_PROMPTS)
    output_dir = REPO_ROOT / args.output_dir

    dataset_root = _resolve_dataset_root(args.dataset, args.revision)
    dataset_info = _read_dataset_info(dataset_root)
    episode, selected_frame, grasp_frame = _select_probe_frame(
        repo_id=args.dataset,
        root=dataset_root,
        revision=args.revision,
        episode=args.episode,
        frame=args.frame,
        grasp_offset=args.grasp_offset,
        chunk_size=args.chunk_size,
    )

    image_key = _infer_image_key(type("MetaProxy", (), {"features": dataset_info.get("features", {})})()) if args.image_key == "auto" else args.image_key
    state_names = _extract_feature_names(dataset_info.get("features", {}).get("observation.state"))
    if state_names is None:
        state_names = list(DEFAULT_STATE_NAMES)
    action_names = _extract_feature_names(dataset_info.get("features", {}).get("action"))
    if action_names is None:
        action_names = list(state_names)

    joint_idx = _joint_index(action_names, args.joint_name)
    state_joint_idx = _joint_index(state_names, args.joint_name)

    task_lookup = _load_task_lookup(dataset_root)
    episode_row = _load_episode_row(dataset_root, episode)
    episode_data = _load_episode_data_rows(dataset_root, episode_row)
    episode_len = int(episode_row["length"])
    max_frame = episode_len - 1
    usable_last = max(0, episode_len - args.chunk_size)
    if selected_frame > usable_last:
        print(
            f"[probe] requested frame {selected_frame} is too close to the episode end; "
            f"clamping to {usable_last} so the chunk is fully defined."
        )
        selected_frame = usable_last
    if selected_frame < 0 or selected_frame > max_frame:
        raise SystemExit(
            f"--frame {selected_frame} is outside episode {episode} with {episode_len} frames."
        )

    try:
        row_idx = episode_data["frame_index"].index(int(selected_frame))
    except ValueError as e:
        raise SystemExit(
            f"Frame {selected_frame} not found inside episode {episode}; "
            f"available range appears to be {episode_data['frame_index'][:3]} ... "
            f"{episode_data['frame_index'][-3:]}"
        ) from e

    state = episode_data["observation.state"][row_idx].astype(np.float32)
    timestamp_s = float(episode_data["timestamp"][row_idx])
    task_index = int(episode_data["task_index"][row_idx])
    current_joint = float(state[state_joint_idx])
    recorded_task = task_lookup.get(task_index, f"<missing task_index={task_index}>")
    recorded_chunk = np.stack(
        episode_data["action"][row_idx : row_idx + args.chunk_size], axis=0
    ).astype(np.float32)
    recorded_joint_traj = recorded_chunk[:, joint_idx]
    recorded_step1 = float(recorded_joint_traj[0])
    recorded_step1_delta = float(recorded_step1 - current_joint)

    video_key_full = f"observation.images.{image_key}"
    video_path_template = str(dataset_info["video_path"])
    relative_video_path = video_path_template.format(
        video_key=video_key_full,
        chunk_index=int(episode_row[f"videos/{video_key_full}/chunk_index"]),
        file_index=int(episode_row[f"videos/{video_key_full}/file_index"]),
    )
    video_path = _ensure_video_file(
        repo_id=args.dataset,
        revision=args.revision,
        root=dataset_root,
        relative_path=relative_video_path,
    )
    fps = float(dataset_info["fps"])
    video_frame_idx, video_alignment = _video_frame_index_for_sample(
        episode_row=episode_row,
        video_key_full=video_key_full,
        timestamp_s=timestamp_s,
        selected_frame=selected_frame,
        fps=fps,
    )
    image_uint8 = _read_video_frame(video_path, video_frame_idx)

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    device = _choose_device(args.device)
    wrapper = _load_policy(args, checkpoint_path=checkpoint_path, image_key=image_key, device=device)
    resolved_policy_type = getattr(wrapper, "_probe_policy_type", args.policy_type)

    print(f"[probe] checkpoint      : {args.checkpoint}")
    print(f"[probe] policy type     : {resolved_policy_type}")
    print(f"[probe] dataset         : {args.dataset}")
    print(f"[probe] episode/frame   : {episode} / {selected_frame}")
    print(f"[probe] grasp frame     : {grasp_frame}")
    print(f"[probe] video frame idx : {video_frame_idx} ({video_alignment})")
    print(f"[probe] image key       : {image_key}")
    print(f"[probe] device          : {device}")
    print(f"[probe] current {args.joint_name}: {current_joint:.4f}")
    print(
        f"[probe] recorded step1 {args.joint_name}: {recorded_step1:.4f} "
        f"(delta {recorded_step1_delta:+.4f})"
    )
    print(f"[probe] recorded task   : {recorded_task}")

    prompt_specs: list[dict[str, str]] = []
    for idx, prompt in enumerate(prompts, start=1):
        prompt_specs.append(
            {
                "label": f"main{idx}",
                "role": "main",
                "prompt": prompt,
            }
        )
    for idx, prompt in enumerate(args.control_prompt, start=1):
        prompt_specs.append(
            {
                "label": f"control{idx}",
                "role": "control",
                "prompt": prompt,
            }
        )
    if args.include_recorded_prompt:
        prompt_specs.append(
            {
                "label": "recorded",
                "role": "recorded",
                "prompt": recorded_task,
            }
        )
    if args.include_empty_prompt:
        prompt_specs.append(
            {
                "label": "empty",
                "role": "control",
                "prompt": "",
            }
        )

    results: list[dict[str, Any]] = []
    for spec in prompt_specs:
        prompt = spec["prompt"]
        wrapper.reset()
        obs = Observation(images={image_key: image_uint8}, state=state, prompt=prompt)
        pred_chunk = wrapper.predict(obs).actions
        joint_traj = pred_chunk[:, joint_idx].astype(np.float32)
        delta = joint_traj - current_joint
        summary = _trajectory_summary(
            joint_traj,
            current_joint,
            args.direction_threshold,
            args.positive_pan_means,
        )
        row = {
            "prompt": prompt,
            "prompt_label": spec["label"],
            "prompt_role": spec["role"],
            "joint_trajectory": joint_traj.astype(float).tolist(),
            "delta_from_current": delta.astype(float).tolist(),
            "predicted_chunk": pred_chunk.astype(float).tolist(),
            "summary": summary,
        }
        results.append(row)

        side_text = summary.get("robot_side_guess")
        suffix = f", side≈{side_text}" if side_text else ""
        print(
            "[probe] "
            f"{spec['label']} ({spec['role']}) prompt={prompt!r} | "
            f"final={summary['final_value']:.4f} "
            f"delta={summary['final_delta_from_current']:+.4f} "
            f"peak_abs={summary['peak_abs_delta_from_current']:.4f} "
            f"trend={summary['direction_sign']}{suffix}"
        )

    late_start_step = max(1, min(args.late_start_step, args.chunk_size))
    pairwise_rows: list[dict[str, Any]] = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            row_i = results[i]
            row_j = results[j]
            chunk_i = np.asarray(row_i["predicted_chunk"], dtype=np.float32)
            chunk_j = np.asarray(row_j["predicted_chunk"], dtype=np.float32)
            joint_i = np.asarray(row_i["joint_trajectory"], dtype=np.float32)
            joint_j = np.asarray(row_j["joint_trajectory"], dtype=np.float32)
            metrics = _pairwise_metrics(
                chunk_a=chunk_i,
                chunk_b=chunk_j,
                joint_a=joint_i,
                joint_b=joint_j,
                late_start_step=late_start_step,
                joint_diverge_threshold=args.joint_diverge_threshold,
                chunk_diverge_threshold=args.chunk_diverge_threshold,
            )
            pair_row = {
                "prompt_a_label": row_i["prompt_label"],
                "prompt_b_label": row_j["prompt_label"],
                "prompt_a_role": row_i["prompt_role"],
                "prompt_b_role": row_j["prompt_role"],
                "prompt_a": row_i["prompt"],
                "prompt_b": row_j["prompt"],
                **metrics,
            }
            pairwise_rows.append(pair_row)
            print(
                "[probe] pair "
                f"{pair_row['prompt_a_label']} vs {pair_row['prompt_b_label']} | "
                f"chunk late cos={pair_row['chunk_late_cos']:.4f} "
                f"late rms={pair_row['chunk_late_rms']:.4f} "
                f"joint div step={pair_row['first_joint_diverge_step']} "
                f"chunk div step={pair_row['first_chunk_diverge_step']}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ep{episode:04d}_frame{selected_frame:04d}_{args.joint_name}"
    json_path = output_dir / f"{tag}.json"
    csv_path = output_dir / f"{tag}.csv"
    pairwise_csv_path = output_dir / f"{tag}_pairs.csv"
    png_path = output_dir / f"{tag}.png"
    frame_path = output_dir / f"{tag}_frame.npy"

    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_local_path": checkpoint_path,
        "policy_type": resolved_policy_type,
        "dataset": args.dataset,
        "dataset_revision": args.revision,
        "episode_index": int(episode),
        "selected_frame_index": int(selected_frame),
        "grasp_frame_index": None if grasp_frame is None else int(grasp_frame),
        "recorded_task": recorded_task,
        "image_key": image_key,
        "state_names": state_names,
        "action_names": action_names,
        "joint_name": args.joint_name,
        "joint_index": int(joint_idx),
        "late_start_step": int(late_start_step),
        "joint_diverge_threshold": float(args.joint_diverge_threshold),
        "chunk_diverge_threshold": float(args.chunk_diverge_threshold),
        "current_joint_value": current_joint,
        "recorded_joint_step1_value": recorded_step1,
        "recorded_joint_step1_delta_from_current": recorded_step1_delta,
        "recorded_joint_trajectory": recorded_joint_traj.astype(float).tolist(),
        "prompts": results,
        "pairwise_metrics": pairwise_rows,
    }
    json_path.write_text(json.dumps(report, indent=2))
    _write_csv(csv_path, results)
    _write_pairwise_csv(pairwise_csv_path, pairwise_rows)
    _plot_probe(
        png_path,
        selected_image=image_uint8,
        joint_name=args.joint_name,
        current_value=current_joint,
        recorded_trajectory=recorded_joint_traj,
        results=results,
        pairwise_rows=pairwise_rows,
        late_start_step=late_start_step,
    )
    np.save(frame_path, image_uint8)

    print(f"[probe] wrote JSON: {json_path}")
    print(f"[probe] wrote CSV : {csv_path}")
    print(f"[probe] wrote pairwise CSV : {pairwise_csv_path}")
    print(f"[probe] wrote PNG : {png_path}")
    print(f"[probe] wrote NPY : {frame_path}")


if __name__ == "__main__":
    main()
