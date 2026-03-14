from __future__ import annotations

from typing import Any

import numpy as np

HORIZON_MIN_ERROR = "horizon must be >= 1"
WINDOW_MIN_ERROR = "window must be >= 1"
SEASON_LENGTH_MIN_ERROR = "season_length must be >= 1"


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def mean_forecast(train: Any, horizon: int) -> np.ndarray:
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if x.size == 0:
        raise ValueError("mean_forecast requires at least 1 training point")
    m = float(np.mean(x))
    return np.full(shape=(horizon,), fill_value=m, dtype=float)


def median_forecast(train: Any, horizon: int) -> np.ndarray:
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
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
        raise ValueError(HORIZON_MIN_ERROR)
    if x.size < 2:
        raise ValueError("drift_forecast requires at least 2 training points")
    slope = (float(x[-1]) - float(x[0])) / float(x.size - 1)
    steps = np.arange(1, horizon + 1, dtype=float)
    return float(x[-1]) + slope * steps


def moving_average_forecast(train: Any, horizon: int, *, window: int) -> np.ndarray:
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if window <= 0:
        raise ValueError(WINDOW_MIN_ERROR)
    if x.size < window:
        raise ValueError(f"moving_average_forecast requires at least {window} points, got {x.size}")

    m = float(np.mean(x[-window:]))
    return np.full(shape=(horizon,), fill_value=m, dtype=float)


def weighted_moving_average_forecast(train: Any, horizon: int, *, window: int) -> np.ndarray:
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if window <= 0:
        raise ValueError(WINDOW_MIN_ERROR)
    if x.size < window:
        raise ValueError(
            f"weighted_moving_average_forecast requires at least {window} points, got {x.size}"
        )

    weights = np.arange(1, int(window) + 1, dtype=float)
    m = float(np.dot(x[-window:], weights) / np.sum(weights))
    return np.full(shape=(horizon,), fill_value=m, dtype=float)


def moving_median_forecast(train: Any, horizon: int, *, window: int) -> np.ndarray:
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if window <= 0:
        raise ValueError(WINDOW_MIN_ERROR)
    if x.size < window:
        raise ValueError(f"moving_median_forecast requires at least {window} points, got {x.size}")

    m = float(np.median(x[-window:]))
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
        raise ValueError(HORIZON_MIN_ERROR)
    if season_length <= 0:
        raise ValueError(SEASON_LENGTH_MIN_ERROR)
    if x.size < season_length:
        raise ValueError(
            f"seasonal_mean_forecast requires at least {season_length} points, got {x.size}"
        )

    means = np.array(
        [float(np.mean(x[i::season_length])) for i in range(season_length)], dtype=float
    )
    idx = np.arange(horizon) % season_length
    return means[idx].astype(float, copy=False)


def seasonal_drift_forecast(train: Any, horizon: int, *, season_length: int) -> np.ndarray:
    """
    Seasonal drift baseline using the last two complete seasons.

    For each seasonal position, estimate the season-over-season delta from the
    previous season to the latest season, then extend that delta forward for
    each future season.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if season_length <= 0:
        raise ValueError(SEASON_LENGTH_MIN_ERROR)
    if x.size < 2 * season_length:
        raise ValueError(
            "seasonal_drift_forecast requires at least two full seasons "
            f"({2 * season_length} points), got {x.size}"
        )

    prev_season = x[-2 * season_length : -season_length]
    last_season = x[-season_length:]
    seasonal_delta = last_season - prev_season

    steps = np.arange(horizon, dtype=int)
    idx = steps % season_length
    seasons_ahead = (steps // season_length) + 1
    return last_season[idx].astype(float, copy=False) + seasonal_delta[idx].astype(
        float, copy=False
    ) * seasons_ahead.astype(float, copy=False)
