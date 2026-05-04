"""Tests for `src/data/prompt_aug.py` — compositional prompt augmentation.

Run with: `pytest tests/test_prompt_aug.py -q` inside the lerobot env.

These tests use a mock base dataset so they don't need any video on disk
or torch.cuda — they exercise only the wrapper logic (arrangement loading,
prompt pool construction, sample rewriting, passthrough).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest
import torch

from src.data.prompt_aug import (
    PromptAugmentingDataset,
    _target_from_task,
    build_prompt_pool,
    load_arrangements,
)

REPO_T1 = "ethrl2026/so101_pickup_20260503_153511_task1"
REPO_T2 = "ethrl2026/so101_pickup_20260503_165245_task2"
ARRS = Path(__file__).resolve().parent.parent / "configs/data/arrangements.json"


class _MockBase(torch.utils.data.Dataset):
    """Minimal LeRobotDataset stand-in: stores items as dicts keyed by index."""
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = dict(self.items[i])
        s["episode_index"] = torch.tensor(s["episode_index"])
        return s


# ---------- target extraction ----------

def test_target_extraction_word_boundary():
    # The same regression as splits._classify_tasks: 'red' must not match
    # the substring inside 'colored'.
    assert _target_from_task("Put the banana in the blue colored bowl.") == "blue"
    assert _target_from_task("Put the banana in the red colored bowl.") == "red"
    assert _target_from_task("Put the banana in the green colored bowl.") == "green"


def test_target_extraction_returns_none_when_ambiguous():
    assert _target_from_task("blue bowl, red bowl") is None
    assert _target_from_task("nothing here") is None


# ---------- prompt pool construction ----------

@pytest.mark.parametrize("arr,target,expect_middle", [
    (["blue", "red", "green"],   "red",   True),
    (["red", "green", "blue"],   "green", True),
    (["blue", "red", "green"],   "blue",  False),  # left end
    (["blue", "red", "green"],   "green", False),  # right end
])
def test_pool_covers_all_four_families(arr, target, expect_middle):
    pool = build_prompt_pool(arr, target)
    has_direct = any("colored" in p for p in pool)
    has_ordinal = any("from the left" in p or "from the right" in p or "middle" in p for p in pool)
    has_relational = any(" of the " in p for p in pool)
    has_negation = any("not" in p or "neither" in p for p in pool)
    assert has_direct
    assert has_ordinal
    assert has_relational
    assert has_negation
    assert ("middle" in " ".join(pool)) == expect_middle


def test_pool_negation_excludes_target_color():
    arr, target = ["blue", "red", "green"], "red"
    pool = build_prompt_pool(arr, target)
    for p in pool:
        if "not" in p or "neither" in p:
            assert "red" not in p, f"target leaked into negation: {p!r}"
            assert "blue" in p and "green" in p


def test_pool_ordinal_position_matches_index():
    # left end → 1st from left / 3rd from right
    p_left = build_prompt_pool(["blue", "red", "green"], "blue")
    assert any("1st bowl from the left" in p for p in p_left)
    assert any("3rd bowl from the right" in p for p in p_left)
    # right end
    p_right = build_prompt_pool(["blue", "red", "green"], "green")
    assert any("3rd bowl from the left" in p for p in p_right)
    assert any("1st bowl from the right" in p for p in p_right)


def test_pool_relational_uses_neighbor_color():
    # red is in the middle of [blue, red, green] → left neighbour blue, right green
    pool = build_prompt_pool(["blue", "red", "green"], "red")
    assert any("right of the blue bowl" in p for p in pool)
    assert any("left of the green bowl" in p for p in pool)


# ---------- arrangement loading ----------

def test_load_arrangements_full_coverage():
    a1 = load_arrangements(ARRS, REPO_T1)
    a2 = load_arrangements(ARRS, REPO_T2)
    assert len(a1) == 90 and len(a2) == 75
    assert a1[0] == ["blue", "red", "green"]
    assert a1[89] == ["blue", "red", "green"]
    assert a2[0] == ["red", "green", "blue"]
    assert a2[14] == ["red", "green", "blue"]
    assert a2[15] == ["green", "blue", "red"]
    assert a2[60] == ["blue", "red", "green"]


def test_load_arrangements_unknown_repo_returns_empty():
    out = load_arrangements(ARRS, "no-such/dataset")
    assert out == {}


# ---------- wrapper behaviour ----------

def test_wrapper_rewrites_in_arrangement_range():
    arrs = load_arrangements(ARRS, REPO_T2)
    base = _MockBase([
        {"episode_index": 0,  "task": "Put the banana in the blue colored bowl."},
        {"episode_index": 5,  "task": "Put the banana in the red colored bowl."},
        {"episode_index": 30, "task": "Put the banana in the blue colored bowl."},
    ])
    w = PromptAugmentingDataset(base, arrs, seed=42)
    seen_per_idx = []
    for i in range(len(base)):
        bucket = collections.Counter()
        for _ in range(50):
            bucket[w[i]["task"]] += 1
        seen_per_idx.append(bucket)
        assert len(bucket) >= 5  # at least 5 distinct rewrites in 50 draws


def test_wrapper_passes_through_unknown_episode():
    arrs = load_arrangements(ARRS, REPO_T2)
    base = _MockBase([
        {"episode_index": 999, "task": "Put the banana in the red colored bowl."},
    ])
    w = PromptAugmentingDataset(base, arrs, seed=1)
    for _ in range(20):
        assert w[0]["task"] == "Put the banana in the red colored bowl."


def test_wrapper_forwards_attributes():
    class WithExtra(_MockBase):
        @property
        def special(self):
            return "VALUE"
    arrs = load_arrangements(ARRS, REPO_T2)
    w = PromptAugmentingDataset(
        WithExtra([{"episode_index": 0, "task": "Put the banana in the blue colored bowl."}]),
        arrs, seed=1,
    )
    assert w.special == "VALUE"


def test_wrapper_len_matches_base():
    arrs = load_arrangements(ARRS, REPO_T2)
    base = _MockBase([
        {"episode_index": i, "task": "Put the banana in the blue colored bowl."}
        for i in range(75)
    ])
    w = PromptAugmentingDataset(base, arrs, seed=42)
    assert len(w) == 75
