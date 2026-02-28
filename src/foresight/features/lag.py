from __future__ import annotations

from typing import Any

import numpy as np


def make_lagged_xy(y: Any, *, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D series into a supervised learning dataset for 1-step forecasting.

    For each time t (starting at `lags`), build:
      X_t = [y_{t-lags}, ..., y_{t-1}]
      y_t = y_t
    """
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"make_lagged_xy expects 1D input, got shape {arr.shape}")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if arr.size <= lags:
        raise ValueError(f"Need > lags points (lags={lags}), got {arr.size}")

    n = int(arr.size)
    rows = n - lags
    X = np.empty((rows, lags), dtype=float)
    yt = np.empty((rows,), dtype=float)
    for i in range(rows):
        t = i + lags
        X[i, :] = arr[t - lags : t]
        yt[i] = arr[t]
    return X, yt
