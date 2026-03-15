from __future__ import annotations

from typing import Any

import numpy as np

from .smoothing import ses_forecast

HORIZON_MIN_ERROR = "horizon must be >= 1"


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


def _all_zero(x: np.ndarray) -> bool:
    return not np.any(x)


def croston_classic_forecast(train: Any, horizon: int, *, alpha: float = 0.1) -> np.ndarray:
    """
    Croston's classic method for intermittent demand.

    Maintains exponentially smoothed estimates of:
      - demand size (z)
      - inter-demand interval (p)

    Forecast is constant: z / p.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if x.size == 0:
        raise ValueError("croston_classic_forecast requires at least 1 training point")
    a = _require_01("alpha", alpha)

    # If all demands are zero, forecast zeros.
    if _all_zero(x):
        return np.zeros((int(horizon),), dtype=float)

    # Initialize with first non-zero demand.
    nz_idx = np.flatnonzero(x > 0.0)
    if nz_idx.size == 0:
        return np.zeros((int(horizon),), dtype=float)

    first = int(nz_idx[0])
    z = float(x[first])
    p = float(first + 1)  # interval length in periods (>=1)
    q = 1.0  # time since last demand (in periods)

    for t in range(first + 1, x.size):
        xt = float(x[t])
        if xt > 0.0:
            z = a * xt + (1.0 - a) * z
            p = a * q + (1.0 - a) * p
            q = 1.0
        else:
            q += 1.0

    fc = 0.0 if p <= 0.0 else z / p
    return np.full((int(horizon),), float(fc), dtype=float)


def croston_sba_forecast(train: Any, horizon: int, *, alpha: float = 0.1) -> np.ndarray:
    """
    Croston-SBA (Syntetos-Boylan Approximation) for intermittent demand.

    Bias correction: multiply Croston's classic forecast by (1 - alpha/2).
    """
    a = _require_01("alpha", alpha)
    fc = croston_classic_forecast(train, horizon, alpha=a)
    return np.asarray(fc * (1.0 - a / 2.0), dtype=float)


def croston_sbj_forecast(train: Any, horizon: int, *, alpha: float = 0.1) -> np.ndarray:
    """
    Croston-SBJ (Syntetos-Boylan-Johnston) bias-corrected intermittent demand.

    Bias correction: multiply Croston's classic forecast by (1 - alpha/(2 - alpha)).
    """
    a = _require_01("alpha", alpha)
    fc = croston_classic_forecast(train, horizon, alpha=a)
    corr = 1.0 - (a / (2.0 - a))
    return np.asarray(fc * corr, dtype=float)


def _croston_sse(x: np.ndarray, *, alpha: float) -> float:
    a = _require_01("alpha", alpha)
    if x.size < 2 or _all_zero(x):
        return 0.0

    nz_idx = np.flatnonzero(x > 0.0)
    if nz_idx.size == 0:
        return 0.0

    first = int(nz_idx[0])
    z = float(x[first])
    p = float(first + 1)
    q = 1.0
    sse = 0.0

    for t in range(first + 1, x.size):
        yhat = 0.0 if p <= 0.0 else z / p
        err = float(x[t]) - yhat
        sse += err * err

        xt = float(x[t])
        if xt > 0.0:
            z = a * xt + (1.0 - a) * z
            p = a * q + (1.0 - a) * p
            q = 1.0
        else:
            q += 1.0

    return float(sse)


def croston_optimized_forecast(
    train: Any,
    horizon: int,
    *,
    grid_size: int = 19,
) -> np.ndarray:
    """
    Croston classic with alpha selected by a simple in-sample SSE grid search.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if x.size == 0:
        raise ValueError("croston_optimized_forecast requires at least 1 training point")
    if grid_size <= 1:
        raise ValueError("grid_size must be >= 2")

    if _all_zero(x):
        return np.zeros((int(horizon),), dtype=float)

    grid = np.linspace(0.05, 0.95, int(grid_size), dtype=float)
    best_alpha = float(grid[0])
    best_sse = float("inf")
    for a in grid:
        sse = _croston_sse(x, alpha=float(a))
        if sse < best_sse:
            best_sse = sse
            best_alpha = float(a)

    return croston_classic_forecast(x, horizon, alpha=best_alpha)


def _les_decay_value(y_hat: float, tau_hat: float, beta: float, tau: float) -> float:
    if tau_hat <= 0.0:
        return 0.0
    factor = 1.0 - (beta * tau) / (2.0 * tau_hat)
    if factor <= 0.0:
        return 0.0
    return float((y_hat / tau_hat) * factor)


def _les_update_state(
    y_hat: float,
    tau_hat: float,
    tau: float,
    y: float,
    *,
    alpha: float,
    beta: float,
) -> tuple[float, float, float, float]:
    if y > 0.0:
        y_hat = alpha * y + (1.0 - alpha) * y_hat
        tau_hat = beta * tau + (1.0 - beta) * tau_hat
        forecast = 0.0 if tau_hat <= 0.0 else y_hat / tau_hat
        return y_hat, tau_hat, 1.0, float(forecast)

    forecast = _les_decay_value(y_hat, tau_hat, beta, tau)
    return y_hat, tau_hat, tau + 1.0, forecast


def les_forecast(
    train: Any,
    horizon: int,
    *,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> np.ndarray:
    """
    Linear-Exponential Smoothing (LES) for intermittent demand and obsolescence.

    When demand is zero, forecasts decay linearly towards 0 (clamped at 0).
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if x.size == 0:
        raise ValueError("les_forecast requires at least 1 training point")

    a = _require_01("alpha", alpha)
    b = _require_01("beta", beta)

    if _all_zero(x):
        return np.zeros((int(horizon),), dtype=float)

    nz_idx = np.flatnonzero(x > 0.0)
    if nz_idx.size == 0:
        return np.zeros((int(horizon),), dtype=float)

    first = int(nz_idx[0])
    y_hat = float(x[first])
    tau = float(first + 1)  # interval length from start to first demand (>=1)
    tau_hat = float(tau)
    tau = 1.0
    f = 0.0 if tau_hat <= 0.0 else y_hat / tau_hat

    for t in range(first + 1, x.size):
        y_hat, tau_hat, tau, f = _les_update_state(
            y_hat,
            tau_hat,
            tau,
            float(x[t]),
            alpha=a,
            beta=b,
        )

    preds = np.empty((int(horizon),), dtype=float)
    for h in range(int(horizon)):
        preds[h] = float(f)
        f = _les_decay_value(y_hat, tau_hat, b, tau)
        tau += 1.0

    return preds


def tsb_forecast(
    train: Any,
    horizon: int,
    *,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> np.ndarray:
    """
    Teunter-Syntetos-Babai (TSB) method for intermittent demand.

    Smooths:
      - demand size (z) when demand occurs
      - demand probability (p) every period

    Forecast is constant: p * z.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if x.size == 0:
        raise ValueError("tsb_forecast requires at least 1 training point")

    a = _require_01("alpha", alpha)
    b = _require_01("beta", beta)

    if _all_zero(x):
        return np.zeros((int(horizon),), dtype=float)

    # Initialize
    z = float(np.mean(x[x > 0.0])) if np.any(x > 0.0) else 0.0
    p = float(np.mean(x > 0.0))

    for t in range(x.size):
        xt = float(x[t])
        it = 1.0 if xt > 0.0 else 0.0
        p = b * it + (1.0 - b) * p
        if xt > 0.0:
            z = a * xt + (1.0 - a) * z

    return np.full((int(horizon),), float(p * z), dtype=float)


def adida_forecast(
    train: Any,
    horizon: int,
    *,
    agg_period: int = 4,
    base: str = "ses",
    alpha: float = 0.2,
) -> np.ndarray:
    """
    ADIDA-style aggregation/disaggregation baseline for intermittent demand.

    1) Aggregate the series into blocks of length `agg_period` using sums.
    2) Forecast the aggregated series using a simple base method.
    3) Disaggregate the forecast by dividing evenly across `agg_period`.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if agg_period <= 0:
        raise ValueError("agg_period must be >= 1")
    if x.size == 0:
        raise ValueError("adida_forecast requires at least 1 training point")

    if _all_zero(x):
        return np.zeros((int(horizon),), dtype=float)

    m = int(agg_period)
    n_full = (x.size // m) * m
    if n_full < m:
        # Not enough for even one full block; fall back to SES/mean on original.
        if base == "ses":
            return ses_forecast(x, horizon, alpha=float(alpha))
        return np.full((int(horizon),), float(np.mean(x)), dtype=float)

    agg = x[:n_full].reshape(-1, m).sum(axis=1)
    agg_h = int(np.ceil(int(horizon) / float(m)))

    base_key = str(base).strip().lower()
    if base_key == "naive-last":
        agg_fc = np.full((agg_h,), float(agg[-1]), dtype=float)
    elif base_key == "mean":
        agg_fc = np.full((agg_h,), float(np.mean(agg)), dtype=float)
    elif base_key == "ses":
        agg_fc = ses_forecast(agg, agg_h, alpha=float(alpha))
    else:
        raise ValueError("base must be one of: naive-last, mean, ses")

    per_period = np.repeat(agg_fc / float(m), m)[: int(horizon)]
    return per_period.astype(float, copy=False)
