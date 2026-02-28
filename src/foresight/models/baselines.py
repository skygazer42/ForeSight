from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def mean_forecast(train: Any, horizon: int) -> np.ndarray:
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size == 0:
        raise ValueError("mean_forecast requires at least 1 training point")
    m = float(np.mean(x))
    return np.full(shape=(horizon,), fill_value=m, dtype=float)


def median_forecast(train: Any, horizon: int) -> np.ndarray:
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size == 0:
        raise ValueError("median_forecast requires at least 1 training point")
    m = float(np.median(x))
    return np.full(shape=(horizon,), fill_value=m, dtype=float)


def drift_forecast(train: Any, horizon: int) -> np.ndarray:
    """
    Random walk with drift (a.k.a. "drift" method).

    Forecast h steps ahead:
        y_T + h * (y_T - y_1) / (T - 1)
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 2:
        raise ValueError("drift_forecast requires at least 2 training points")
    slope = (float(x[-1]) - float(x[0])) / float(x.size - 1)
    steps = np.arange(1, horizon + 1, dtype=float)
    return float(x[-1]) + slope * steps


def moving_average_forecast(train: Any, horizon: int, *, window: int) -> np.ndarray:
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if window <= 0:
        raise ValueError("window must be >= 1")
    if x.size < window:
        raise ValueError(f"moving_average_forecast requires at least {window} points, got {x.size}")

    m = float(np.mean(x[-window:]))
    return np.full(shape=(horizon,), fill_value=m, dtype=float)


def seasonal_mean_forecast(train: Any, horizon: int, *, season_length: int) -> np.ndarray:
    """
    Seasonal mean baseline.

    For each position within the season, take the mean of all historical values
    at that position (index modulo `season_length`), then repeat those means
    to fill the horizon.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if season_length <= 0:
        raise ValueError("season_length must be >= 1")
    if x.size < season_length:
        raise ValueError(
            f"seasonal_mean_forecast requires at least {season_length} points, got {x.size}"
        )

    means = np.array(
        [float(np.mean(x[i::season_length])) for i in range(season_length)], dtype=float
    )
    idx = np.arange(horizon) % season_length
    return means[idx].astype(float, copy=False)
