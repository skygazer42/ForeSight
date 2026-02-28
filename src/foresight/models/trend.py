from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def poly_trend_forecast(train: Any, horizon: int, *, degree: int = 1) -> np.ndarray:
    """
    Polynomial trend regression on time index t, forecast forward.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if degree < 0:
        raise ValueError("degree must be >= 0")
    if x.size < degree + 2:
        raise ValueError("poly_trend_forecast requires more points than the polynomial degree")

    n = int(x.size)
    t = np.arange(n, dtype=float)
    # Vandermonde with [1, t, t^2, ...]
    X = np.vander(t, N=int(degree) + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)

    tf = np.arange(n, n + int(horizon), dtype=float)
    Xf = np.vander(tf, N=int(degree) + 1, increasing=True)
    yhat = Xf @ coef
    return np.asarray(yhat, dtype=float)
