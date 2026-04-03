from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Series contains NaN/inf")
    return x


def _validate_analog_forecast_inputs(
    x: np.ndarray,
    *,
    horizon: int,
    lags: int,
    k: int,
) -> tuple[int, int]:
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if k <= 0:
        raise ValueError("k must be >= 1")

    horizon_count = int(horizon)
    lag_count = int(lags)
    n = int(x.size)
    if n <= lag_count:
        raise ValueError(f"Need > lags points (lags={lag_count}), got {n}")
    return horizon_count, lag_count


def _maybe_normalize_window_bank(windows: np.ndarray, *, normalize: bool) -> np.ndarray:
    if not normalize:
        return windows

    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True) + 1e-8
    return (windows - means) / stds


def _analog_window_distances(
    window_bank: np.ndarray,
    query_window: np.ndarray,
    *,
    normalize: bool,
) -> np.ndarray:
    query = np.asarray(query_window, dtype=float)
    if bool(normalize):
        query_mean = float(query.mean())
        query_std = float(query.std()) + 1e-8
        query = (query - query_mean) / query_std
    return np.sum((window_bank - query) ** 2, axis=1)


def _analog_neighbor_prediction(
    y_next: np.ndarray,
    distances: np.ndarray,
    *,
    k_eff: int,
    weights: str,
) -> float:
    idx = np.argpartition(distances, k_eff - 1)[:k_eff]
    if weights == "uniform":
        return float(np.mean(y_next[idx]))

    dist_subset = distances[idx]
    near = dist_subset <= 1e-12
    if np.any(near):
        return float(np.mean(y_next[idx][near]))

    weight_values = 1.0 / dist_subset
    return float(np.sum(weight_values * y_next[idx]) / np.sum(weight_values))


def analog_knn_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 12,
    k: int = 5,
    normalize: bool = True,
    weights: str = "uniform",
) -> np.ndarray:
    """
    Analog forecasting via kNN over lag windows (non-parametric).

    - Build all historical windows of length `lags`
    - At each forecast step, find the nearest windows to the most recent window
    - Predict the next value from the neighbors' subsequent observations

    Notes:
    - This is a simple baseline; it can be slow for very long series.
    - `normalize=True` z-scores each window before computing distances.
    """
    x = _as_1d_float_array(train)
    horizon_count, lag_count = _validate_analog_forecast_inputs(
        x,
        horizon=horizon,
        lags=lags,
        k=k,
    )
    n = int(x.size)

    # X: windows (n-lags rows), y_next: next value after each window
    X = np.lib.stride_tricks.sliding_window_view(x, lag_count)[:-1]
    y_next = x[lag_count:]

    weights_s = str(weights).strip().lower()
    if weights_s not in {"uniform", "distance"}:
        raise ValueError("weights must be 'uniform' or 'distance'")

    normalized = bool(normalize)
    x_work = _maybe_normalize_window_bank(X, normalize=normalized)
    k_eff = min(int(k), int(X.shape[0]))
    hist = np.empty((n + horizon_count,), dtype=float)
    hist[:n] = x

    preds = np.empty((horizon_count,), dtype=float)
    for h in range(horizon_count):
        end = n + h
        w = hist[end - lag_count : end]
        distances = _analog_window_distances(
            x_work,
            w,
            normalize=normalized,
        )

        pred = _analog_neighbor_prediction(
            y_next,
            distances,
            k_eff=k_eff,
            weights=weights_s,
        )
        preds[h] = pred
        hist[end] = pred

    return preds
