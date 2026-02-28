from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .backtesting import walk_forward

ForecasterFn = Callable[[Any, int], np.ndarray]


def _as_1d_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {arr.shape}")
    return arr


def bootstrap_intervals(
    train: Any,
    *,
    horizon: int,
    forecaster: ForecasterFn,
    min_train_size: int,
    n_samples: int = 1000,
    quantiles: tuple[float, float] = (0.1, 0.9),
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Residual bootstrap prediction intervals for point forecasts.

    This is a lightweight baseline: we estimate an empirical residual distribution
    from 1-step walk-forward errors on the training set, then add sampled residuals
    to the horizon-step point forecasts.
    """
    y = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if n_samples <= 0:
        raise ValueError("n_samples must be >= 1")
    q_lo, q_hi = map(float, quantiles)
    if not (0.0 < q_lo < q_hi < 1.0):
        raise ValueError("quantiles must satisfy 0 < q_lo < q_hi < 1")

    point = np.asarray(forecaster(y, int(horizon)), dtype=float)
    if point.shape != (int(horizon),):
        raise ValueError(f"forecaster must return shape ({horizon},), got {point.shape}")

    def _one_step(train_window: Any, _h: int) -> np.ndarray:
        return np.asarray(forecaster(train_window, 1), dtype=float)

    wf = walk_forward(
        y,
        horizon=1,
        step=1,
        min_train_size=int(min_train_size),
        forecaster=_one_step,
    )
    residuals = (wf.y_true.reshape(-1) - wf.y_pred.reshape(-1)).astype(float, copy=False)
    if residuals.size == 0:
        raise ValueError(
            "No residuals available to bootstrap; check min_train_size and data length."
        )

    rng = np.random.default_rng(seed)
    draws = rng.choice(residuals, size=(int(n_samples), int(horizon)), replace=True)
    samples = point.reshape(1, -1) + draws

    lower = np.quantile(samples, q_lo, axis=0)
    upper = np.quantile(samples, q_hi, axis=0)

    return {
        "yhat": point,
        "lower": lower,
        "upper": upper,
        "quantiles": (q_lo, q_hi),
        "n_samples": int(n_samples),
    }
