from __future__ import annotations

from typing import Any

import numpy as np

from .tabular import normalize_int_tuple, normalize_lag_steps


def make_lagged_xy(
    y: Any, *, lags: Any, start_t: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D series into a supervised learning dataset for 1-step forecasting.

    For each time t (starting at `lags`), build:
      X_t = [y_{t-lags}, ..., y_{t-1}]
      y_t = y_t
    """
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"make_lagged_xy expects 1D input, got shape {arr.shape}")
    lag_steps = normalize_lag_steps(lags, allow_zero=False, name="lags")
    if not lag_steps:
        raise ValueError("lags must be >= 1")

    t0 = int(max(lag_steps)) if start_t is None else max(int(max(lag_steps)), int(start_t))
    if arr.size <= t0:
        raise ValueError(f"Need > start_t points (start_t={t0}), got {arr.size}")

    n = int(arr.size)
    rows = n - t0
    X = np.empty((rows, len(lag_steps)), dtype=float)
    yt = np.empty((rows,), dtype=float)
    for i in range(rows):
        t = i + t0
        X[i, :] = np.asarray([arr[t - lag] for lag in lag_steps], dtype=float)
        yt[i] = arr[t]
    return X, yt


def make_lagged_xy_multi(
    y: Any,
    *,
    lags: Any,
    horizon: int,
    start_t: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a 1D series into a supervised learning dataset for direct multi-step forecasting.

    For each time t, build:
      X_t = [y_{t-lags}, ..., y_{t-1}]
      Y_t = [y_t, y_{t+1}, ..., y_{t+horizon-1}]

    Returns:
      (X, Y, t_index)
        - X: (rows, n_lags)
        - Y: (rows, horizon)
        - t_index: (rows,) starting indices for each target window
    """
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"make_lagged_xy_multi expects 1D input, got shape {arr.shape}")
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")

    lag_steps = normalize_lag_steps(lags, allow_zero=False, name="lags")
    if not lag_steps:
        raise ValueError("lags must be >= 1")

    t0 = int(max(lag_steps)) if start_t is None else max(int(max(lag_steps)), int(start_t))
    if arr.size < (t0 + h):
        raise ValueError(f"Need >= start_t+horizon points (start_t={t0}, horizon={h}), got {arr.size}")

    n = int(arr.size)
    last_t = n - h
    rows = int(last_t - t0 + 1)
    if rows <= 0:
        raise ValueError("Not enough points to build lagged multi-horizon dataset")

    X = np.empty((rows, len(lag_steps)), dtype=float)
    Y = np.empty((rows, h), dtype=float)
    t_index = np.empty((rows,), dtype=int)
    for i in range(rows):
        t = i + t0
        X[i, :] = np.asarray([arr[t - lag] for lag in lag_steps], dtype=float)
        Y[i, :] = arr[t : t + h]
        t_index[i] = int(t)
    return X, Y, t_index


def build_seasonal_lag_features(
    y: Any,
    *,
    t: Any,
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
) -> tuple[np.ndarray, list[str]]:
    """
    Build seasonal lag/diff features from a 1D series for given target indices `t`.

    Definitions (for each target index t_i):
      - seasonal_lags: include y[t_i - p] for each p
      - seasonal_diff_lags: include y[t_i - 1] - y[t_i - 1 - p] for each p

    Notes:
      - `t` may include the value n (one-step-ahead) as long as y has length n.
      - All features depend only on past values (no target leakage).
    """
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {arr.shape}")
    if arr.size == 0:
        return np.empty((0, 0), dtype=float), []
    if not np.all(np.isfinite(arr)):
        raise ValueError("Non-finite values in series")

    tt = np.asarray(t, dtype=int).reshape(-1)
    if tt.size == 0:
        return np.empty((0, 0), dtype=float), []
    if np.any(tt < 0):
        raise ValueError("t indices must be >= 0")
    if int(np.max(tt)) > int(arr.size):
        raise ValueError("t indices must be <= len(series)")

    lags_tup = tuple(sorted(set(normalize_int_tuple(seasonal_lags))))
    diffs_tup = tuple(sorted(set(normalize_int_tuple(seasonal_diff_lags))))
    if not lags_tup and not diffs_tup:
        return np.empty((int(tt.size), 0), dtype=float), []
    if any(int(p) <= 0 for p in lags_tup):
        raise ValueError("seasonal_lags must be >= 1")
    if any(int(p) <= 0 for p in diffs_tup):
        raise ValueError("seasonal_diff_lags must be >= 1")

    feats: list[np.ndarray] = []
    names: list[str] = []

    for p in lags_tup:
        idx = tt - int(p)
        if np.any(idx < 0):
            raise ValueError("seasonal_lags require t >= lag")
        feats.append(arr[idx].reshape(-1, 1))
        names.append(f"season_lag_{int(p)}")

    if diffs_tup:
        idx1 = tt - 1
        if np.any(idx1 < 0):
            raise ValueError("seasonal_diff_lags require t >= 1")
        last = arr[idx1]
        for p in diffs_tup:
            idx0 = tt - 1 - int(p)
            if np.any(idx0 < 0):
                raise ValueError("seasonal_diff_lags require t >= lag+1")
            feats.append((last - arr[idx0]).reshape(-1, 1))
            names.append(f"season_diff_{int(p)}")

    X = np.concatenate(feats, axis=1) if feats else np.empty((int(tt.size), 0), dtype=float)
    X = X.astype(float, copy=False)
    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite values in seasonal lag features")
    return X, names
