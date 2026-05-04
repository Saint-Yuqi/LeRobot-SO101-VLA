"""Per-sample prompt augmentation for compositional VLA fine-tuning.

The recorded `tasks.parquet` only contains direct color labels
("Put the banana in the [color] colored bowl."). Eval-day compositional
prompts (ordinal / relational / negation) are never observed at training,
so without augmentation the model has to rely solely on the VLM backbone's
zero-shot reasoning to bridge from "2nd from the left" to "the green bowl"
at inference — a high-variance bet.

This module wraps a `LeRobotDataset` and rewrites the per-sample `task`
field on the fly using the bowl arrangement for that episode. The same
target color is described with multiple equivalent phrasings; the model
is exposed to all four eval-time families during training.

Usage
-----
    from src.data.prompt_aug import PromptAugmentingDataset, load_arrangements

    arr = load_arrangements("configs/data/arrangements.json", repo_id)
    train_ds = PromptAugmentingDataset(base=lerobot_ds, arrangements=arr)

The wrapper transparently forwards `__len__` and any other attribute
access to the base dataset, so it composes with `ConcatDataset` and the
existing DataLoader path without further changes.

Constraint: no external LLM is invoked. Phrasings are drawn from a fixed
template pool — same model class the eval rules permit.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

POSITIONS = ("left", "middle", "right")
ORDINAL_FROM_LEFT = ("1st", "2nd", "3rd")
ORDINAL_FROM_RIGHT = ("3rd", "2nd", "1st")  # same indices reversed


def load_arrangements(path: str | Path, repo_id: str) -> dict[int, list[str]]:
    """Read the JSON arrangement file and expand episode ranges to a flat
    `{episode_index: [left, middle, right]}` map for one source.

    Returns an empty dict (no augmentation) if the source isn't listed.
    """
    p = Path(path)
    with p.open() as f:
        data = json.load(f)
    if repo_id not in data:
        return {}
    out: dict[int, list[str]] = {}
    for entry in data[repo_id]:
        a, b = entry["episode_range"]
        for ep in range(int(a), int(b) + 1):
            out[ep] = list(entry["arrangement"])
    return out


def _direct(target: str) -> list[str]:
    return [
        f"Put the banana in the {target} colored bowl.",
        f"Put the banana into the {target} bowl.",
        f"Place the banana in the {target} colored bowl.",
        f"Pick up the banana and place it in the {target} bowl.",
    ]


def _ordinal(arrangement: list[str], target: str) -> list[str]:
    i = arrangement.index(target)  # 0=left, 1=middle, 2=right
    out = [
        f"Put the banana into the {ORDINAL_FROM_LEFT[i]} bowl from the left.",
        f"Put the banana into the {ORDINAL_FROM_RIGHT[i]} bowl from the right.",
    ]
    if i == 1:
        out.append("Put the banana into the middle bowl.")
        out.append("Put the banana into the bowl in the middle.")
    return out


def _relational(arrangement: list[str], target: str) -> list[str]:
    i = arrangement.index(target)
    out: list[str] = []
    # left neighbour ⇒ "right of <left>"
    if i - 1 >= 0:
        left_color = arrangement[i - 1]
        out.append(f"Put the banana into the bowl on the right of the {left_color} bowl.")
        out.append(f"Put the banana into the bowl to the right of the {left_color} bowl.")
    # right neighbour ⇒ "left of <right>"
    if i + 1 < len(arrangement):
        right_color = arrangement[i + 1]
        out.append(f"Put the banana into the bowl on the left of the {right_color} bowl.")
        out.append(f"Put the banana into the bowl to the left of the {right_color} bowl.")
    # only the middle bowl has both neighbours; ends only have one neighbour each
    return out


def _negation(arrangement: list[str], target: str) -> list[str]:
    others = [c for c in arrangement if c != target]
    if len(others) != 2:
        return []
    a, b = others
    return [
        f"Put the banana into the bowl that is not {a} and not {b}.",
        f"Put the banana into the bowl that is not {b} and not {a}.",
        f"Put the banana into the bowl that is neither {a} nor {b}.",
        f"Put the banana into the bowl that is neither {b} nor {a}.",
    ]


def build_prompt_pool(arrangement: list[str], target: str) -> list[str]:
    """Return all valid phrasings for (arrangement, target). Always non-empty:
    direct color is unconditional; the others depend on geometry."""
    pool = []
    pool.extend(_direct(target))
    pool.extend(_ordinal(arrangement, target))
    pool.extend(_relational(arrangement, target))
    pool.extend(_negation(arrangement, target))
    return pool


COLORS = ("blue", "red", "green")


def _target_from_task(task_text: str) -> str | None:
    """Extract the single target color from a task string (returns None if
    the prompt doesn't name exactly one of {blue, red, green})."""
    import re
    found = [c for c in COLORS if re.search(rf"\b{c}\b", task_text, re.IGNORECASE)]
    return found[0] if len(found) == 1 else None


class PromptAugmentingDataset(torch.utils.data.Dataset):
    """Wraps a `LeRobotDataset` (or compatible) and rewrites `sample["task"]`
    with a randomly-chosen equivalent phrasing for the episode's arrangement.

    The base dataset must expose an `episode_index` field per sample; samples
    whose episode is not in the arrangement map are returned untouched.

    Workers each have their own RNG state (forked at process spawn), so two
    DataLoader workers will diverge after the first call — that's the desired
    behavior for augmentation (each epoch sees different phrasings)."""

    def __init__(
        self,
        base: torch.utils.data.Dataset,
        arrangements: dict[int, list[str]],
        seed: int = 42,
    ) -> None:
        self.base = base
        self.arrangements = arrangements
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base[idx]
        ep_field = sample.get("episode_index")
        if ep_field is None:
            return sample
        ep = int(ep_field.item() if torch.is_tensor(ep_field) else ep_field)
        arr = self.arrangements.get(ep)
        if arr is None:
            return sample
        original = sample.get("task", "")
        target = _target_from_task(original) if isinstance(original, str) else None
        if target is None or target not in arr:
            return sample
        pool = build_prompt_pool(arr, target)
        sample["task"] = self._rng.choice(pool)
        return sample

    # Forward attribute access for anything else the trainer might query
    # (e.g. `meta`, `features`) so this wrapper is transparent. Guarded so
    # that unpickling (which sets `__dict__` directly and may probe dunder
    # attributes before `self.base` exists) doesn't infinitely recurse on
    # `getattr(self.base, ...)`.
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "base":
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "base")
        except AttributeError as e:
            raise AttributeError(name) from e
        return getattr(base, name)
