"""Unit tests for color-stratified episode splitting.

Run with `pytest tests/test_splits.py -q` inside the lerobot env.
These tests use synthetic by_color dicts so they don't need any dataset
on disk and run in milliseconds.
"""
from __future__ import annotations

import pytest

from src.data.splits import _classify_tasks, train_val_episode_split


def test_classify_single_color():
    assert _classify_tasks(["pick the banana and put it in the blue bowl"]) == "blue"
    assert _classify_tasks(["put it in the GREEN bowl"]) == "green"


def test_classify_no_color_is_other():
    assert _classify_tasks(["do the thing"]) == "other"


def test_classify_multiple_colors_is_other():
    assert _classify_tasks(["blue bowl", "red bowl"]) == "other"


def test_classify_word_boundary_does_not_match_substring():
    # Regression: substring match used to fire on "red" inside "colored",
    # so every "Put the banana in the blue colored bowl." was bucketed as
    # `other` (multi-color hit blue+red). Word-boundary regex prevents this.
    assert _classify_tasks(["Put the banana in the blue colored bowl."]) == "blue"
    assert _classify_tasks(["Put the banana in the red colored bowl."]) == "red"
    assert _classify_tasks(["Put the banana in the green colored bowl."]) == "green"


def test_split_per_color_is_deterministic():
    by = {"blue": list(range(20)), "red": list(range(20, 40)), "green": list(range(40, 60))}
    a_train, a_val = train_val_episode_split(by, per_color=2, seed=42)
    b_train, b_val = train_val_episode_split(by, per_color=2, seed=42)
    assert a_train == b_train
    assert a_val == b_val


def test_split_different_seed_gives_different_val():
    by = {"blue": list(range(20)), "red": list(range(20, 40)), "green": list(range(40, 60))}
    _, val_a = train_val_episode_split(by, per_color=2, seed=42)
    _, val_b = train_val_episode_split(by, per_color=2, seed=7)
    assert val_a != val_b


def test_split_no_overlap_and_covers_all():
    by = {"blue": list(range(20)), "red": list(range(20, 40)), "green": list(range(40, 60))}
    train, val = train_val_episode_split(by, per_color=2, seed=42)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == sorted(sum(by.values(), []))


def test_split_per_color_holds_out_exact_count():
    by = {"blue": list(range(20)), "red": list(range(20, 40)), "green": list(range(40, 60))}
    _, val = train_val_episode_split(by, per_color=3, seed=42)
    assert len(val) == 9  # 3 per color × 3 colors


def test_split_fraction_rounds_up():
    by = {"blue": list(range(11))}  # ceil(0.1 * 11) = 2
    _, val = train_val_episode_split(by, fraction=0.1, seed=42)
    assert len(val) == 2


def test_split_fraction_floor_one():
    by = {"blue": list(range(5))}  # ceil(0.1 * 5) = 1
    _, val = train_val_episode_split(by, fraction=0.1, seed=42)
    assert len(val) == 1


def test_split_skips_empty_color():
    by = {"blue": list(range(20)), "red": [], "green": list(range(20, 40))}
    train, val = train_val_episode_split(by, per_color=2, seed=42)
    assert len(val) == 4  # 2 colors × 2


def test_split_min_train_per_color_violation_raises():
    by = {"blue": list(range(4))}  # 4 - 2 = 2 < min_train_per_color=3
    with pytest.raises(ValueError, match="blue"):
        train_val_episode_split(by, per_color=2, min_train_per_color=3, seed=42)


def test_split_requires_exactly_one_of_per_color_or_fraction():
    by = {"blue": list(range(20))}
    with pytest.raises(ValueError, match="exactly one"):
        train_val_episode_split(by, seed=42)
    with pytest.raises(ValueError, match="exactly one"):
        train_val_episode_split(by, per_color=1, fraction=0.1, seed=42)


def test_split_fraction_out_of_range():
    by = {"blue": list(range(20))}
    with pytest.raises(ValueError):
        train_val_episode_split(by, fraction=1.0, seed=42)
    with pytest.raises(ValueError):
        train_val_episode_split(by, fraction=0.0, seed=42)


def test_split_per_color_must_be_positive():
    by = {"blue": list(range(20))}
    with pytest.raises(ValueError):
        train_val_episode_split(by, per_color=0, seed=42)


def test_split_works_with_single_color_only():
    by = {"blue": list(range(20))}  # red/green missing — pre-collection state
    train, val = train_val_episode_split(by, fraction=0.1, seed=42)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(20))
    assert len(val) == 2  # ceil(0.1 * 20) = 2
