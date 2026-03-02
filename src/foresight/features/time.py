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
