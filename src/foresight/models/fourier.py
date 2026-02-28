from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def fourier_regression_forecast(
    train: Any,
    horizon: int,
    *,
    period: int,
    order: int = 2,
    include_trend: bool = True,
) -> np.ndarray:
    """
    Linear regression with Fourier seasonal terms (sin/cos) + optional linear trend.

    This is a lightweight alternative to Prophet-style seasonality modeling.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if period <= 0:
        raise ValueError("period must be >= 1")
    if order < 0:
        raise ValueError("order must be >= 0")
    if x.size < 3:
        raise ValueError("fourier_regression_forecast requires at least 3 training points")

    n = int(x.size)
    t = np.arange(n, dtype=float)

    cols: list[np.ndarray] = [np.ones((n,), dtype=float)]
    if include_trend:
        cols.append(t)

    w = 2.0 * np.pi / float(period)
    for k in range(1, int(order) + 1):
        cols.append(np.sin(w * float(k) * t))
        cols.append(np.cos(w * float(k) * t))

    X = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)

    tf = np.arange(n, n + int(horizon), dtype=float)
    cols_f: list[np.ndarray] = [np.ones((int(horizon),), dtype=float)]
    if include_trend:
        cols_f.append(tf)
    for k in range(1, int(order) + 1):
        cols_f.append(np.sin(w * float(k) * tf))
        cols_f.append(np.cos(w * float(k) * tf))

    Xf = np.stack(cols_f, axis=1)
    yhat = Xf @ coef
    return np.asarray(yhat, dtype=float)


def _normalize_periods(periods: Any) -> tuple[int, ...]:
    if periods is None:
        raise ValueError("periods must be provided")
    if isinstance(periods, int | float):
        p = int(periods)
        return (p,)
    if isinstance(periods, str):
        s = periods.strip()
        if not s:
            raise ValueError("periods must be non-empty")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return tuple(int(p) for p in parts)
    if isinstance(periods, list | tuple):
        return tuple(int(p) for p in periods)
    return (int(periods),)


def _normalize_orders(orders: Any, *, n_periods: int) -> tuple[int, ...]:
    if orders is None:
        return tuple([2] * int(n_periods))
    if isinstance(orders, int | float):
        return tuple([int(orders)] * int(n_periods))
    if isinstance(orders, str):
        s = orders.strip()
        if not s:
            return tuple([2] * int(n_periods))
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) == 1 and n_periods > 1:
            return tuple([int(parts[0])] * int(n_periods))
        if len(parts) != int(n_periods):
            raise ValueError("orders must be an int or have the same length as periods")
        return tuple(int(p) for p in parts)
    if isinstance(orders, list | tuple):
        if len(orders) == 1 and n_periods > 1:
            return tuple([int(orders[0])] * int(n_periods))
        if len(orders) != int(n_periods):
            raise ValueError("orders must be an int or have the same length as periods")
        return tuple(int(o) for o in orders)
    return tuple([int(orders)] * int(n_periods))


def fourier_multi_regression_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    include_trend: bool = True,
) -> np.ndarray:
    """
    Fourier regression with multiple seasonalities.

    Example for daily data:
      periods=(7, 365), orders=(3, 10)
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 3:
        raise ValueError("fourier_multi_regression_forecast requires at least 3 training points")

    periods_tup = _normalize_periods(periods)
    if not periods_tup:
        raise ValueError("periods must be non-empty")
    if any(int(p) <= 0 for p in periods_tup):
        raise ValueError("All periods must be >= 1")
    periods_tup = tuple(int(p) for p in periods_tup)

    orders_tup = _normalize_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("All orders must be >= 0")
    orders_tup = tuple(int(o) for o in orders_tup)

    n = int(x.size)
    t = np.arange(n, dtype=float)

    cols: list[np.ndarray] = [np.ones((n,), dtype=float)]
    if bool(include_trend):
        cols.append(t)

    for period, order in zip(periods_tup, orders_tup, strict=True):
        w = 2.0 * np.pi / float(period)
        for k in range(1, int(order) + 1):
            cols.append(np.sin(w * float(k) * t))
            cols.append(np.cos(w * float(k) * t))

    X = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)

    tf = np.arange(n, n + int(horizon), dtype=float)
    cols_f: list[np.ndarray] = [np.ones((int(horizon),), dtype=float)]
    if bool(include_trend):
        cols_f.append(tf)

    for period, order in zip(periods_tup, orders_tup, strict=True):
        w = 2.0 * np.pi / float(period)
        for k in range(1, int(order) + 1):
            cols_f.append(np.sin(w * float(k) * tf))
            cols_f.append(np.cos(w * float(k) * tf))

    Xf = np.stack(cols_f, axis=1)
    yhat = Xf @ coef
    return np.asarray(yhat, dtype=float)
