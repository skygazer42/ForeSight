from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_time_features(
    ds: Any,
    *,
    add_time_idx: bool = True,
    add_dow: bool = True,
    add_month: bool = True,
    add_doy: bool = True,
    add_hour: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Build lightweight, dependency-free time features from a datetime-like array.

    Returns:
      (X, names)
        - X: float array of shape (n, n_features)
        - names: feature names aligned with X columns

    Notes:
      - If `ds` cannot be parsed as datetimes, falls back to a single `time_idx` feature.
      - Features are encoded as sin/cos pairs for cyclical components.
    """
    s = pd.Series(ds)

    dt: pd.Series | None
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = s
    else:
        try:
            dt = pd.to_datetime(s, errors="raise")
        except Exception:
            dt = None

    feats: list[np.ndarray] = []
    names: list[str] = []

    if add_time_idx:
        t = np.arange(len(s), dtype=float)
        denom = max(1.0, float(len(s) - 1))
        time_idx = t / denom
        feats.append(time_idx.reshape(-1, 1))
        names.append("time_idx")

    if dt is None:
        # Fallback: keep feature *shape* stable by emitting zeros for cyclical features.
        zeros = np.zeros((len(s), 1), dtype=float)
        if add_dow:
            feats.extend([zeros, zeros])
            names.extend(["dow_sin", "dow_cos"])
        if add_month:
            feats.extend([zeros, zeros])
            names.extend(["month_sin", "month_cos"])
        if add_doy:
            feats.extend([zeros, zeros])
            names.extend(["doy_sin", "doy_cos"])
        if add_hour:
            feats.extend([zeros, zeros])
            names.extend(["hour_sin", "hour_cos"])

        X = np.concatenate(feats, axis=1) if feats else np.empty((len(s), 0), dtype=float)
        return X.astype(float, copy=False), names

    # Normalize to naive datetimes to avoid tz issues in .dt accessors
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(None)

    two_pi = 2.0 * np.pi

    if add_dow:
        dow = dt.dt.dayofweek.to_numpy(dtype=float, copy=False)
        feats.append(np.sin(two_pi * dow / 7.0).reshape(-1, 1))
        feats.append(np.cos(two_pi * dow / 7.0).reshape(-1, 1))
        names.extend(["dow_sin", "dow_cos"])

    if add_month:
        month = dt.dt.month.to_numpy(dtype=float, copy=False)
        feats.append(np.sin(two_pi * (month - 1.0) / 12.0).reshape(-1, 1))
        feats.append(np.cos(two_pi * (month - 1.0) / 12.0).reshape(-1, 1))
        names.extend(["month_sin", "month_cos"])

    if add_doy:
        doy = dt.dt.dayofyear.to_numpy(dtype=float, copy=False)
        feats.append(np.sin(two_pi * (doy - 1.0) / 365.25).reshape(-1, 1))
        feats.append(np.cos(two_pi * (doy - 1.0) / 365.25).reshape(-1, 1))
        names.extend(["doy_sin", "doy_cos"])

    if add_hour:
        hour = dt.dt.hour.to_numpy(dtype=float, copy=False)
        # For daily/monthly data hour is 0, but sin/cos are still well-defined.
        feats.append(np.sin(two_pi * hour / 24.0).reshape(-1, 1))
        feats.append(np.cos(two_pi * hour / 24.0).reshape(-1, 1))
        names.extend(["hour_sin", "hour_cos"])

    X = np.concatenate(feats, axis=1) if feats else np.empty((len(s), 0), dtype=float)
    X = X.astype(float, copy=False)

    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite values in generated time features")

    return X, names


def build_fourier_features(
    t: Any,
    *,
    periods: Any,
    orders: Any = 2,
) -> tuple[np.ndarray, list[str]]:
    """
    Build Fourier sin/cos features for a 1D time index array.

    Parameters:
      - t: 1D array-like time index (typically integer step index).
      - periods: seasonal period(s) in *steps*, e.g. 7, 24, 365 or "7,365".
      - orders: harmonic order(s). If an int, uses the same order for all periods.
               If a sequence/string, must be length 1 or match periods length.

    Returns:
      (X, names)
        - X: float array of shape (n, 2*sum(orders))
        - names: aligned feature names (fourier_{period}_{sin|cos}_{k})
    """
    tt = np.asarray(t, dtype=float).reshape(-1)
    if tt.size == 0:
        return np.empty((0, 0), dtype=float), []
    if not np.all(np.isfinite(tt)):
        raise ValueError("Non-finite values in time index")

    # Normalize periods
    if periods is None:
        return np.empty((int(tt.size), 0), dtype=float), []
    if isinstance(periods, str):
        s = periods.strip()
        if not s:
            return np.empty((int(tt.size), 0), dtype=float), []
        parts = [p.strip() for p in s.split(",") if p.strip()]
        periods_tup = tuple(int(float(p)) for p in parts)
    elif isinstance(periods, list | tuple):
        periods_tup = tuple(int(float(p)) for p in periods)
    else:
        periods_tup = (int(float(periods)),)

    if not periods_tup:
        return np.empty((int(tt.size), 0), dtype=float), []
    if any(int(p) <= 1 for p in periods_tup):
        raise ValueError("fourier periods must be >= 2")
    periods_tup = tuple(int(p) for p in periods_tup)

    # Normalize orders
    if orders is None:
        orders_tup = tuple([2] * len(periods_tup))
    elif isinstance(orders, int | float):
        orders_tup = tuple([int(float(orders))] * len(periods_tup))
    elif isinstance(orders, str):
        s = orders.strip()
        if not s:
            orders_tup = tuple([2] * len(periods_tup))
        else:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) == 1 and len(periods_tup) > 1:
                orders_tup = tuple([int(float(parts[0]))] * len(periods_tup))
            else:
                if len(parts) != len(periods_tup):
                    raise ValueError("fourier orders must be an int or match periods length")
                orders_tup = tuple(int(float(p)) for p in parts)
    elif isinstance(orders, list | tuple):
        if len(orders) == 1 and len(periods_tup) > 1:
            orders_tup = tuple([int(float(orders[0]))] * len(periods_tup))
        else:
            if len(orders) != len(periods_tup):
                raise ValueError("fourier orders must be an int or match periods length")
            orders_tup = tuple(int(float(o)) for o in orders)
    else:
        orders_tup = tuple([int(float(orders))] * len(periods_tup))

    if any(int(o) <= 0 for o in orders_tup):
        raise ValueError("fourier orders must be >= 1")
    orders_tup = tuple(int(o) for o in orders_tup)

    feats: list[np.ndarray] = []
    names: list[str] = []
    two_pi = 2.0 * np.pi
    for period, order in zip(periods_tup, orders_tup, strict=True):
        w = two_pi / float(period)
        for k in range(1, int(order) + 1):
            ang = w * float(k) * tt
            feats.append(np.sin(ang).reshape(-1, 1))
            feats.append(np.cos(ang).reshape(-1, 1))
            names.append(f"fourier_{int(period)}_sin_{k}")
            names.append(f"fourier_{int(period)}_cos_{k}")

    X = np.concatenate(feats, axis=1) if feats else np.empty((int(tt.size), 0), dtype=float)
    X = X.astype(float, copy=False)
    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite values in generated Fourier features")
    return X, names
