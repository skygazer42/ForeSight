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


def _require_positive(name: str, v: float) -> float:
    vf = float(v)
    if vf <= 0.0:
        raise ValueError(f"{name} must be > 0, got {vf}")
    return vf


def kalman_local_level_forecast(
    train: Any,
    horizon: int,
    *,
    process_variance: float | None = None,
    obs_variance: float | None = None,
) -> np.ndarray:
    """
    Local-level state space model (random walk level) with a simple Kalman filter.

    Model:
      level_t = level_{t-1} + eta_t,   eta_t ~ N(0, q)
      y_t     = level_t + eps_t,       eps_t ~ N(0, r)

    Returns the mean forecast (point predictions).
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size == 0:
        raise ValueError("kalman_local_level_forecast requires at least 1 training point")

    base = float(np.var(np.diff(x))) if x.size >= 2 else float(np.var(x))
    base = base if base > 0.0 else 1e-6
    q = _require_positive(
        "process_variance", base * 0.01 if process_variance is None else process_variance
    )
    r = _require_positive("obs_variance", base * 0.1 if obs_variance is None else obs_variance)

    level = float(x[0])
    P = 1e6

    for y in x:
        # Predict
        level_pred = level
        p_pred = P + q

        # Update
        S = p_pred + r
        K = p_pred / S
        level = level_pred + K * (float(y) - level_pred)
        P = (1.0 - K) * p_pred

    return np.full((int(horizon),), float(level), dtype=float)


def kalman_local_linear_trend_forecast(
    train: Any,
    horizon: int,
    *,
    level_variance: float | None = None,
    trend_variance: float | None = None,
    obs_variance: float | None = None,
) -> np.ndarray:
    """
    Local linear trend model with a simple Kalman filter.

    State:
      [level_t, trend_t]

    Transition:
      level_t = level_{t-1} + trend_{t-1} + eta_t,   eta_t ~ N(0, q_level)
      trend_t = trend_{t-1} + zeta_t,                zeta_t ~ N(0, q_trend)

    Observation:
      y_t = level_t + eps_t, eps_t ~ N(0, r)
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 2:
        raise ValueError("kalman_local_linear_trend_forecast requires at least 2 training points")

    base = float(np.var(np.diff(x))) if x.size >= 2 else float(np.var(x))
    base = base if base > 0.0 else 1e-6

    q_level = _require_positive(
        "level_variance", base * 0.01 if level_variance is None else level_variance
    )
    q_trend = _require_positive(
        "trend_variance", base * 0.001 if trend_variance is None else trend_variance
    )
    r = _require_positive("obs_variance", base * 0.1 if obs_variance is None else obs_variance)

    # State and covariance
    level = float(x[0])
    trend = float(x[1] - x[0])
    P = np.diag([1e6, 1e6]).astype(float)

    F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
    Q = np.array([[q_level, 0.0], [0.0, q_trend]], dtype=float)
    H = np.array([1.0, 0.0], dtype=float)  # observation matrix (row vector)

    for y in x:
        # Predict
        state = np.array([level, trend], dtype=float)
        state_pred = F @ state
        p_pred = F @ P @ F.T + Q

        # Update (scalar observation)
        y_pred = float(H @ state_pred)
        S = float(H @ p_pred @ H.T + r)
        if S <= 0.0:
            raise ValueError("Numerical issue: non-positive innovation variance")
        K = (p_pred @ H.T) / S  # shape (2,)
        innov = float(y) - y_pred
        state_upd = state_pred + K * innov
        P = (np.eye(2) - np.outer(K, H)) @ p_pred

        level = float(state_upd[0])
        trend = float(state_upd[1])

    steps = np.arange(1, int(horizon) + 1, dtype=float)
    return np.asarray(level + trend * steps, dtype=float)
