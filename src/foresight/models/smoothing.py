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


def ses_forecast(train: Any, horizon: int, *, alpha: float) -> np.ndarray:
    """
    Simple Exponential Smoothing (SES) point forecast.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size == 0:
        raise ValueError("ses_forecast requires at least 1 training point")
    a = _require_01("alpha", alpha)

    level = float(x[0])
    for t in range(1, x.size):
        level = a * float(x[t]) + (1.0 - a) * level

    return np.full(shape=(horizon,), fill_value=float(level), dtype=float)


def _ses_sse(x: np.ndarray, *, alpha: float) -> float:
    a = _require_01("alpha", alpha)
    level = float(x[0])
    sse = 0.0
    for t in range(1, x.size):
        yhat = level
        err = float(x[t]) - yhat
        sse += err * err
        level = a * float(x[t]) + (1.0 - a) * level
    return float(sse)


def ses_auto_forecast(train: Any, horizon: int, *, grid_size: int = 19) -> np.ndarray:
    """
    Auto-tuned SES via a simple grid search over alpha.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 2:
        raise ValueError("ses_auto_forecast requires at least 2 training points")
    if grid_size <= 1:
        raise ValueError("grid_size must be >= 2")

    grid = np.linspace(0.05, 0.95, int(grid_size), dtype=float)
    best_alpha = float(grid[0])
    best_sse = float("inf")
    for a in grid:
        sse = _ses_sse(x, alpha=float(a))
        if sse < best_sse:
            best_sse = sse
            best_alpha = float(a)

    return ses_forecast(x, horizon, alpha=best_alpha)


def holt_forecast(train: Any, horizon: int, *, alpha: float, beta: float) -> np.ndarray:
    """
    Holt's linear trend method (additive trend).
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 2:
        raise ValueError("holt_forecast requires at least 2 training points")
    a = _require_01("alpha", alpha)
    b = _require_01("beta", beta)

    level = float(x[0])
    trend = float(x[1] - x[0])

    for t in range(1, x.size):
        prev_level = level
        level = a * float(x[t]) + (1.0 - a) * (level + trend)
        trend = b * (level - prev_level) + (1.0 - b) * trend

    steps = np.arange(1, horizon + 1, dtype=float)
    return level + trend * steps


def holt_damped_forecast(
    train: Any,
    horizon: int,
    *,
    alpha: float,
    beta: float,
    phi: float = 0.9,
) -> np.ndarray:
    """
    Holt's damped trend method (additive damped trend).

    `phi` is the damping parameter in [0, 1]. phi=1 reduces to Holt.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 2:
        raise ValueError("holt_damped_forecast requires at least 2 training points")
    a = _require_01("alpha", alpha)
    b = _require_01("beta", beta)
    phi_f = float(phi)
    if not (0.0 <= phi_f <= 1.0):
        raise ValueError("phi must be in [0, 1]")

    level = float(x[0])
    trend = float(x[1] - x[0])

    for t in range(1, x.size):
        prev_level = level
        level = a * float(x[t]) + (1.0 - a) * (level + phi_f * trend)
        trend = b * (level - prev_level) + (1.0 - b) * (phi_f * trend)

    steps = np.arange(1, horizon + 1, dtype=float)
    if abs(1.0 - phi_f) < 1e-12:
        damp = steps
    else:
        damp = phi_f * (1.0 - np.power(phi_f, steps)) / (1.0 - phi_f)

    return level + trend * damp


def _holt_sse(x: np.ndarray, *, alpha: float, beta: float) -> float:
    a = _require_01("alpha", alpha)
    b = _require_01("beta", beta)

    level = float(x[0])
    trend = float(x[1] - x[0])
    sse = 0.0

    for t in range(1, x.size):
        yhat = level + trend
        err = float(x[t]) - yhat
        sse += err * err

        prev_level = level
        level = a * float(x[t]) + (1.0 - a) * (level + trend)
        trend = b * (level - prev_level) + (1.0 - b) * trend

    return float(sse)


def holt_auto_forecast(train: Any, horizon: int, *, grid_size: int = 10) -> np.ndarray:
    """
    Auto-tuned Holt via a simple grid search over (alpha, beta).
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 3:
        raise ValueError("holt_auto_forecast requires at least 3 training points")
    if grid_size <= 1:
        raise ValueError("grid_size must be >= 2")

    grid = np.linspace(0.1, 0.9, int(grid_size), dtype=float)
    best_alpha = float(grid[0])
    best_beta = float(grid[0])
    best_sse = float("inf")
    for a in grid:
        for b in grid:
            sse = _holt_sse(x, alpha=float(a), beta=float(b))
            if sse < best_sse:
                best_sse = sse
                best_alpha = float(a)
                best_beta = float(b)

    return holt_forecast(x, horizon, alpha=best_alpha, beta=best_beta)


def holt_winters_additive_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """
    Holt-Winters additive seasonality + additive trend.

    This is a lightweight implementation intended for experimentation and
    baselines (not a full-featured stats package replacement).
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if season_length <= 0:
        raise ValueError("season_length must be >= 1")
    if x.size < 2 * season_length:
        raise ValueError(
            f"holt_winters_additive_forecast requires at least 2*season_length={2 * season_length} points, got {x.size}"
        )

    a = _require_01("alpha", alpha)
    b = _require_01("beta", beta)
    g = _require_01("gamma", gamma)

    m = int(season_length)

    # Initialization using two seasons.
    season1 = x[:m]
    season2 = x[m : 2 * m]
    level = float(np.mean(season1))
    trend = (float(np.mean(season2)) - float(np.mean(season1))) / float(m)
    seasonals = (season1 - level).astype(float, copy=True)  # length m

    for t in range(x.size):
        idx = t % m
        prev_level = level
        prev_season = float(seasonals[idx])

        level = a * (float(x[t]) - prev_season) + (1.0 - a) * (level + trend)
        trend = b * (level - prev_level) + (1.0 - b) * trend
        seasonals[idx] = g * (float(x[t]) - level) + (1.0 - g) * prev_season

    steps = np.arange(1, horizon + 1, dtype=float)
    seasonal_idx = (np.arange(horizon) + x.size) % m
    return level + trend * steps + seasonals[seasonal_idx]


def _hw_additive_sse(
    x: np.ndarray,
    *,
    season_length: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    a = _require_01("alpha", alpha)
    b = _require_01("beta", beta)
    g = _require_01("gamma", gamma)

    m = int(season_length)
    if x.size < 2 * m:
        raise ValueError("Need at least 2 full seasons for Holt-Winters initialization.")

    season1 = x[:m]
    season2 = x[m : 2 * m]
    level = float(np.mean(season1))
    trend = (float(np.mean(season2)) - float(np.mean(season1))) / float(m)
    seasonals = (season1 - level).astype(float, copy=True)

    sse = 0.0
    for t in range(x.size):
        idx = t % m
        yhat = level + trend + float(seasonals[idx])
        err = float(x[t]) - yhat
        sse += err * err

        prev_level = level
        prev_season = float(seasonals[idx])

        level = a * (float(x[t]) - prev_season) + (1.0 - a) * (level + trend)
        trend = b * (level - prev_level) + (1.0 - b) * trend
        seasonals[idx] = g * (float(x[t]) - level) + (1.0 - g) * prev_season

    return float(sse)


def holt_winters_additive_auto_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    grid_size: int = 7,
) -> np.ndarray:
    """
    Auto-tuned Holt-Winters additive via a small grid search over (alpha,beta,gamma).
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if season_length <= 0:
        raise ValueError("season_length must be >= 1")
    if x.size < 2 * int(season_length):
        raise ValueError(
            "holt_winters_additive_auto_forecast requires at least 2 full seasons of data"
        )
    if grid_size <= 1:
        raise ValueError("grid_size must be >= 2")

    grid = np.linspace(0.1, 0.9, int(grid_size), dtype=float)
    best = (float(grid[0]), float(grid[0]), float(grid[0]))
    best_sse = float("inf")
    for a in grid:
        for b in grid:
            for g in grid:
                sse = _hw_additive_sse(
                    x,
                    season_length=int(season_length),
                    alpha=float(a),
                    beta=float(b),
                    gamma=float(g),
                )
                if sse < best_sse:
                    best_sse = sse
                    best = (float(a), float(b), float(g))

    a_best, b_best, g_best = best
    return holt_winters_additive_forecast(
        x,
        horizon,
        season_length=int(season_length),
        alpha=a_best,
        beta=b_best,
        gamma=g_best,
    )


def holt_winters_multiplicative_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """
    Holt-Winters multiplicative seasonality + additive trend.

    This variant is useful when seasonal amplitude scales with the series level.

    Notes:
      - Requires strictly positive values (division-based seasonality).
      - This is a lightweight baseline implementation (not a full stats package replacement).
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if season_length <= 0:
        raise ValueError("season_length must be >= 1")
    if x.size < 2 * int(season_length):
        raise ValueError(
            "holt_winters_multiplicative_forecast requires at least "
            f"2*season_length={2 * int(season_length)} points, got {x.size}"
        )
    if np.any(x <= 0.0):
        raise ValueError("holt_winters_multiplicative_forecast requires strictly positive values")

    a = _require_01("alpha", alpha)
    b = _require_01("beta", beta)
    g = _require_01("gamma", gamma)
    m = int(season_length)
    eps = 1e-12

    # Initialization using two seasons.
    season1 = x[:m]
    season2 = x[m : 2 * m]
    level = float(np.mean(season1))
    if abs(level) < eps:
        level = eps
    trend = (float(np.mean(season2)) - float(np.mean(season1))) / float(m)
    seasonals = (season1 / level).astype(float, copy=True)  # length m
    seasonals = np.where(np.abs(seasonals) < eps, eps, seasonals)

    for t in range(x.size):
        idx = t % m
        prev_level = level
        prev_season = float(seasonals[idx])
        denom_season = prev_season if abs(prev_season) >= eps else eps

        level = a * (float(x[t]) / denom_season) + (1.0 - a) * (level + trend)
        if abs(level) < eps:
            level = eps
        trend = b * (level - prev_level) + (1.0 - b) * trend
        seasonals[idx] = g * (float(x[t]) / level) + (1.0 - g) * prev_season
        if abs(float(seasonals[idx])) < eps:
            seasonals[idx] = eps

    steps = np.arange(1, horizon + 1, dtype=float)
    seasonal_idx = (np.arange(horizon) + x.size) % m
    return (level + trend * steps) * seasonals[seasonal_idx]


def _hw_multiplicative_sse(
    x: np.ndarray,
    *,
    season_length: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    a = _require_01("alpha", alpha)
    b = _require_01("beta", beta)
    g = _require_01("gamma", gamma)

    m = int(season_length)
    if x.size < 2 * m:
        raise ValueError("Need at least 2 full seasons for Holt-Winters initialization.")
    if np.any(x <= 0.0):
        raise ValueError("Holt-Winters multiplicative requires strictly positive values.")

    eps = 1e-12
    season1 = x[:m]
    season2 = x[m : 2 * m]
    level = float(np.mean(season1))
    if abs(level) < eps:
        level = eps
    trend = (float(np.mean(season2)) - float(np.mean(season1))) / float(m)
    seasonals = (season1 / level).astype(float, copy=True)
    seasonals = np.where(np.abs(seasonals) < eps, eps, seasonals)

    sse = 0.0
    for t in range(x.size):
        idx = t % m
        yhat = (level + trend) * float(seasonals[idx])
        err = float(x[t]) - yhat
        sse += err * err

        prev_level = level
        prev_season = float(seasonals[idx])
        denom_season = prev_season if abs(prev_season) >= eps else eps

        level = a * (float(x[t]) / denom_season) + (1.0 - a) * (level + trend)
        if abs(level) < eps:
            level = eps
        trend = b * (level - prev_level) + (1.0 - b) * trend
        seasonals[idx] = g * (float(x[t]) / level) + (1.0 - g) * prev_season
        if abs(float(seasonals[idx])) < eps:
            seasonals[idx] = eps

    return float(sse)


def holt_winters_multiplicative_auto_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    grid_size: int = 7,
) -> np.ndarray:
    """
    Auto-tuned Holt-Winters multiplicative via a small grid search over (alpha,beta,gamma).
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if season_length <= 0:
        raise ValueError("season_length must be >= 1")
    if x.size < 2 * int(season_length):
        raise ValueError(
            "holt_winters_multiplicative_auto_forecast requires at least 2 full seasons of data"
        )
    if grid_size <= 1:
        raise ValueError("grid_size must be >= 2")

    grid = np.linspace(0.1, 0.9, int(grid_size), dtype=float)
    best = (float(grid[0]), float(grid[0]), float(grid[0]))
    best_sse = float("inf")
    for a in grid:
        for b in grid:
            for g in grid:
                sse = _hw_multiplicative_sse(
                    x,
                    season_length=int(season_length),
                    alpha=float(a),
                    beta=float(b),
                    gamma=float(g),
                )
                if sse < best_sse:
                    best_sse = sse
                    best = (float(a), float(b), float(g))

    a_best, b_best, g_best = best
    return holt_winters_multiplicative_forecast(
        x,
        horizon,
        season_length=int(season_length),
        alpha=a_best,
        beta=b_best,
        gamma=g_best,
    )
