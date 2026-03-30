import pytest

from foresight.splits import rolling_origin_split_sequence, rolling_origin_splits


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


def test_rolling_origin_split_sequence_keeps_first_n_windows() -> None:
    splits = rolling_origin_split_sequence(
        10,
        horizon=2,
        step_size=2,
        min_train_size=4,
        limit=2,
        keep="first",
        limit_error="limit must be >= 1",
    )

    assert len(splits) == 2
    assert [split.train_end for split in splits] == [4, 6]


def test_rolling_origin_split_sequence_keeps_last_n_windows() -> None:
    splits = rolling_origin_split_sequence(
        10,
        horizon=2,
        step_size=2,
        min_train_size=4,
        limit=2,
        keep="last",
        limit_error="limit must be >= 1",
    )

    assert len(splits) == 2
    assert [split.train_end for split in splits] == [6, 8]


def test_rolling_origin_split_sequence_invalid_limit_raises() -> None:
    with pytest.raises(ValueError, match="limit must be >= 1"):
        rolling_origin_split_sequence(
            10,
            horizon=2,
            step_size=2,
            min_train_size=4,
            limit=0,
            keep="first",
            limit_error="limit must be >= 1",
        )
