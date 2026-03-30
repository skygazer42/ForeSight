from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class RollingOriginSplit:
    """
    A single rolling-origin (a.k.a. rolling/expanding window) split.

    All indices follow python slicing conventions:
      - train = y[train_start:train_end]
      - test  = y[test_start:test_end]
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def rolling_origin_splits(
    n_obs: int,
    *,
    horizon: int,
    step_size: int = 1,
    min_train_size: int,
    max_train_size: int | None = None,
) -> Iterator[RollingOriginSplit]:
    """
    Generate rolling-origin splits for a 1D time series of length `n_obs`.

    Parameters mirror common time-series CV APIs (e.g. step_size / horizon / input_size).

    If `max_train_size` is None, the training window expands from 0.
    If set, the training window rolls with a fixed maximum length.
    """
    n_obs_int = int(n_obs)
    if n_obs_int <= 0:
        raise ValueError("n_obs must be >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if step_size <= 0:
        raise ValueError("step_size must be >= 1")
    if min_train_size <= 0:
        raise ValueError("min_train_size must be >= 1")
    if max_train_size is not None and max_train_size <= 0:
        raise ValueError("max_train_size must be >= 1")

    max_train_end = n_obs_int - int(horizon)
    if max_train_end < int(min_train_size):
        raise ValueError(
            "Not enough data: need at least min_train_size+horizon="
            f"{int(min_train_size) + int(horizon)}, got {n_obs_int}"
        )

    for train_end in range(int(min_train_size), max_train_end + 1, int(step_size)):
        train_start = 0
        if max_train_size is not None:
            train_start = max(0, int(train_end) - int(max_train_size))

        test_start = int(train_end)
        test_end = int(train_end) + int(horizon)

        yield RollingOriginSplit(
            train_start=int(train_start),
            train_end=int(train_end),
            test_start=int(test_start),
            test_end=int(test_end),
        )


def rolling_origin_split_sequence(
    n_obs: int,
    *,
    horizon: int,
    step_size: int = 1,
    min_train_size: int,
    max_train_size: int | None = None,
    limit: int | None = None,
    keep: str = "first",
    limit_error: str = "limit must be >= 1",
) -> tuple[RollingOriginSplit, ...]:
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")

    splits = list(
        rolling_origin_splits(
            int(n_obs),
            horizon=int(horizon),
            step_size=int(step_size),
            min_train_size=int(min_train_size),
            max_train_size=max_train_size,
        )
    )
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError(str(limit_error))
        count = int(limit)
        splits = splits[:count] if keep == "first" else splits[-count:]
    return tuple(splits)
