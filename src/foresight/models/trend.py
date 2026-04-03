from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def _validated_poly_trend_inputs(
    train: Any,
    *,
    horizon: int,
    degree: int,
) -> tuple[np.ndarray, int, int]:
    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    d = int(degree)
    if d < 0:
        raise ValueError("degree must be >= 0")
    if x.size < d + 2:
        raise ValueError("poly_trend_forecast requires more points than the polynomial degree")
    return x, h, d


def _poly_design_matrix(t: np.ndarray, *, degree: int) -> np.ndarray:
    return np.vander(np.asarray(t, dtype=float), N=int(degree) + 1, increasing=True)


def poly_trend_forecast(train: Any, horizon: int, *, degree: int = 1) -> np.ndarray:
    """
    Polynomial trend regression on time index t, forecast forward.
    """
    x, h, d = _validated_poly_trend_inputs(train, horizon=horizon, degree=degree)

    n = int(x.size)
    t = np.arange(n, dtype=float)
    X = _poly_design_matrix(t, degree=d)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)

    tf = np.arange(n, n + h, dtype=float)
    x_future = _poly_design_matrix(tf, degree=d)
    yhat = x_future @ coef
    return np.asarray(yhat, dtype=float)
