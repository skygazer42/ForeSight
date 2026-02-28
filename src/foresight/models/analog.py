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
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if k <= 0:
        raise ValueError("k must be >= 1")

    n = int(x.size)
    lag_count = int(lags)
    if n <= lag_count:
        raise ValueError(f"Need > lags points (lags={lag_count}), got {n}")

    # X: windows (n-lags rows), y_next: next value after each window
    X = np.lib.stride_tricks.sliding_window_view(x, lag_count)[:-1]
    y_next = x[lag_count:]

    X_work = X
    if bool(normalize):
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True) + 1e-8
        X_work = (X - means) / stds

    weights_s = str(weights).strip().lower()
    if weights_s not in {"uniform", "distance"}:
        raise ValueError("weights must be 'uniform' or 'distance'")

    k_eff = min(int(k), int(X.shape[0]))
    hist = np.empty((n + int(horizon),), dtype=float)
    hist[:n] = x

    preds = np.empty((int(horizon),), dtype=float)
    for h in range(int(horizon)):
        end = n + h
        w = hist[end - lag_count : end]
        if bool(normalize):
            wm = float(w.mean())
            ws = float(w.std()) + 1e-8
            w_work = (w - wm) / ws
            dist = np.sum((X_work - w_work) ** 2, axis=1)
        else:
            dist = np.sum((X_work - w) ** 2, axis=1)

        idx = np.argpartition(dist, k_eff - 1)[:k_eff]
        if weights_s == "uniform":
            pred = float(np.mean(y_next[idx]))
        else:
            d = dist[idx]
            near = d <= 1e-12
            if np.any(near):
                pred = float(np.mean(y_next[idx][near]))
            else:
                wts = 1.0 / d
                pred = float(np.sum(wts * y_next[idx]) / np.sum(wts))

        preds[h] = pred
        hist[end] = pred

    return preds
