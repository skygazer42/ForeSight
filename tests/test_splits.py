import pytest

from foresight.splits import rolling_origin_splits


def test_rolling_origin_splits_counts_windows():
    splits = list(
        rolling_origin_splits(
            10,
            horizon=2,
            step_size=2,
            min_train_size=4,
        )
    )
    # train_end in {4,6,8} -> 3 windows
    assert len(splits) == 3
    assert splits[0].train_start == 0
    assert splits[0].train_end == 4
    assert splits[0].test_start == 4
    assert splits[0].test_end == 6


def test_rolling_origin_splits_with_max_train_size_rolls():
    splits = list(
        rolling_origin_splits(
            10,
            horizon=2,
            step_size=2,
            min_train_size=4,
            max_train_size=4,
        )
    )
    assert splits[0].train_start == 0  # 0:4
    assert splits[1].train_start == 2  # 2:6
    assert splits[2].train_start == 4  # 4:8


def test_rolling_origin_splits_invalid_raises():
    with pytest.raises(ValueError):
        list(rolling_origin_splits(3, horizon=2, step_size=1, min_train_size=3))
