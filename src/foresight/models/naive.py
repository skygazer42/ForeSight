from __future__ import annotations

from typing import Any

import numpy as np


def naive_last(train: Any, horizon: int) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.size == 0:
        raise ValueError("naive_last requires at least 1 training point")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    return np.full(shape=(horizon,), fill_value=float(x[-1]), dtype=float)


def seasonal_naive(train: Any, horizon: int, *, season_length: int) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if season_length <= 0:
        raise ValueError("season_length must be >= 1")
    if x.size < season_length:
        raise ValueError(f"seasonal_naive requires at least {season_length} points, got {x.size}")

    last_season = x[-season_length:]
    idx = np.arange(horizon) % season_length
    return last_season[idx].astype(float, copy=False)

