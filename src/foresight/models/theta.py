from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def _require_01(name: str, v: float) -> float:
    vf = float(v)
    if not (0.0 <= vf <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {vf}")
    return vf


def _ses_level(x: np.ndarray, *, alpha: float) -> float:
    level = float(x[0])
    for t in range(1, x.size):
        level = alpha * float(x[t]) + (1.0 - alpha) * level
    return float(level)


def _linear_slope_ols(x: np.ndarray) -> float:
    """
    OLS slope of y ~ a + b*t where t = 1..n.
    """
    n = int(x.size)
    t = np.arange(1.0, n + 1.0, dtype=float)
    t_mean = float(np.mean(t))
    y_mean = float(np.mean(x))
    denom = float(np.sum((t - t_mean) ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum((t - t_mean) * (x - y_mean)) / denom)


def theta_forecast(train: Any, horizon: int, *, alpha: float = 0.2) -> np.ndarray:
    """
    A lightweight Theta-style baseline.

    The classic Theta method is closely related to SES with drift. Here we:
    - compute the SES level (with `alpha`)
    - compute a linear OLS slope
    - add drift = 0.5 * slope per step
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 2:
        raise ValueError("theta_forecast requires at least 2 training points")
    a = _require_01("alpha", alpha)

    level = _ses_level(x, alpha=a)
    slope = _linear_slope_ols(x)
    drift = 0.5 * slope

    steps = np.arange(1, horizon + 1, dtype=float)
    return level + drift * steps


def _theta_one_step_sse(x: np.ndarray, *, alpha: float) -> float:
    a = _require_01("alpha", alpha)
    if x.size < 2:
        return 0.0

    # Precompute prefix sums for OLS slope over y ~ c + b*t where t=1..m.
    # This slope does not depend on alpha.
    n = int(x.size)
    sum_y = np.cumsum(x, dtype=float)
    t = np.arange(1.0, n + 1.0, dtype=float)
    sum_ty = np.cumsum(t * x, dtype=float)

    level = float(x[0])
    sse = 0.0
    for i in range(1, n):
        m = i  # prefix length (using x[:i])
        if m <= 1:
            slope = 0.0
        else:
            y_mean = float(sum_y[m - 1] / float(m))
            t_mean = (float(m) + 1.0) / 2.0
            denom = float(m * (m * m - 1.0) / 12.0)
            if denom <= 0.0:
                slope = 0.0
            else:
                num = float(sum_ty[m - 1] - float(m) * t_mean * y_mean)
                slope = num / denom

        yhat = level + 0.5 * slope
        err = float(x[i]) - float(yhat)
        sse += err * err

        level = a * float(x[i]) + (1.0 - a) * level

    return float(sse)


def theta_auto_forecast(train: Any, horizon: int, *, grid_size: int = 19) -> np.ndarray:
    """
    Auto-tuned Theta-style baseline via a simple grid search over alpha.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 3:
        raise ValueError("theta_auto_forecast requires at least 3 training points")
    if grid_size <= 1:
        raise ValueError("grid_size must be >= 2")

    grid = np.linspace(0.05, 0.95, int(grid_size), dtype=float)
    best_alpha = float(grid[0])
    best_sse = float("inf")
    for a in grid:
        sse = _theta_one_step_sse(x, alpha=float(a))
        if sse < best_sse:
            best_sse = sse
            best_alpha = float(a)

    return theta_forecast(x, horizon, alpha=best_alpha)
