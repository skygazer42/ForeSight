from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_CYCLICAL_COMPONENT_SPECS = (
    ("dow", "dayofweek", 7.0, 0.0),
    ("month", "month", 12.0, 1.0),
    ("doy", "dayofyear", 365.25, 1.0),
    ("hour", "hour", 24.0, 0.0),
)


def _coerce_datetime_series(ds: Any) -> pd.Series | None:
    s = pd.Series(ds)
    dtype = getattr(s, "dtype", None)
    if pd.api.types.is_datetime64_any_dtype(s) or isinstance(dtype, pd.DatetimeTZDtype):
        return s
    try:
        return pd.to_datetime(s, errors="raise")
    except Exception:
        return None


def _append_time_index_feature(feats: list[np.ndarray], names: list[str], *, size: int) -> None:
    t = np.arange(size, dtype=float)
    denom = max(1.0, float(size - 1))
    feats.append((t / denom).reshape(-1, 1))
    names.append("time_idx")


def _selected_cyclical_components(
    *,
    add_dow: bool,
    add_month: bool,
    add_doy: bool,
    add_hour: bool,
) -> tuple[tuple[str, str, float, float], ...]:
    enabled_flags = (add_dow, add_month, add_doy, add_hour)
    return tuple(
        spec
        for spec, enabled in zip(_CYCLICAL_COMPONENT_SPECS, enabled_flags, strict=True)
        if enabled
    )


def _append_zero_cyclical_feature_pairs(
    feats: list[np.ndarray],
    names: list[str],
    *,
    size: int,
    components: tuple[tuple[str, str, float, float], ...],
) -> None:
    zeros = np.zeros((size, 1), dtype=float)
    for name, _attr, _period, _offset in components:
        feats.extend([zeros, zeros])
        names.extend([f"{name}_sin", f"{name}_cos"])


def _append_cyclical_feature_pair(
    feats: list[np.ndarray],
    names: list[str],
    values: np.ndarray,
    *,
    period: float,
    name: str,
    offset: float = 0.0,
) -> None:
    two_pi = 2.0 * np.pi
    angles = two_pi * (np.asarray(values, dtype=float) - float(offset)) / float(period)
    feats.append(np.sin(angles).reshape(-1, 1))
    feats.append(np.cos(angles).reshape(-1, 1))
    names.extend([f"{name}_sin", f"{name}_cos"])


def _normalize_datetime_series(dt: pd.Series) -> pd.Series:
    if getattr(dt.dt, "tz", None) is None:
        return dt
    return dt.dt.tz_convert(None)


def _append_datetime_cyclical_feature_pairs(
    dt: pd.Series,
    feats: list[np.ndarray],
    names: list[str],
    *,
    components: tuple[tuple[str, str, float, float], ...],
) -> None:
    normalized = _normalize_datetime_series(dt)
    for name, attr, period, offset in components:
        values = getattr(normalized.dt, attr).to_numpy(dtype=float, copy=False)
        _append_cyclical_feature_pair(feats, names, values, period=period, name=name, offset=offset)


def _finalize_feature_matrix(
    feats: list[np.ndarray],
    names: list[str],
    *,
    rows: int,
    error_message: str,
) -> tuple[np.ndarray, list[str]]:
    matrix = np.concatenate(feats, axis=1) if feats else np.empty((rows, 0), dtype=float)
    matrix = matrix.astype(float, copy=False)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(error_message)
    return matrix, names


def _split_csv_values(items: str) -> list[str]:
    s = items.strip()
    if not s:
        return []
    return [part.strip() for part in s.split(",") if part.strip()]


def _normalize_fourier_periods(periods: Any) -> tuple[int, ...]:
    if periods is None:
        return ()
    if isinstance(periods, str):
        values = tuple(int(float(part)) for part in _split_csv_values(periods))
    elif isinstance(periods, list | tuple):
        values = tuple(int(float(part)) for part in periods)
    else:
        values = (int(float(periods)),)
    if not values:
        return ()
    periods_tup = tuple(int(period) for period in values)
    if any(period <= 1 for period in periods_tup):
        raise ValueError("fourier periods must be >= 2")
    return periods_tup


def _broadcast_fourier_order(value: Any, *, count: int) -> tuple[int, ...]:
    return tuple([int(float(value))] * int(count))


def _coerce_fourier_order_values(values: Any, *, n_periods: int) -> tuple[int, ...]:
    if len(values) != n_periods:
        raise ValueError("fourier orders must be an int or match periods length")
    return tuple(int(float(value)) for value in values)


def _normalize_string_fourier_orders(orders: str, *, n_periods: int) -> tuple[int, ...]:
    parts = _split_csv_values(orders)
    if not parts:
        return _broadcast_fourier_order(2, count=n_periods)
    if len(parts) == 1 and n_periods > 1:
        return _broadcast_fourier_order(parts[0], count=n_periods)
    return _coerce_fourier_order_values(parts, n_periods=n_periods)


def _normalize_sequence_fourier_orders(orders: list | tuple, *, n_periods: int) -> tuple[int, ...]:
    if len(orders) == 1 and n_periods > 1:
        return _broadcast_fourier_order(orders[0], count=n_periods)
    return _coerce_fourier_order_values(orders, n_periods=n_periods)


def _validate_fourier_orders(orders: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(int(order) for order in orders)
    if any(order <= 0 for order in normalized):
        raise ValueError("fourier orders must be >= 1")
    return normalized


def _normalize_fourier_orders(orders: Any, *, n_periods: int) -> tuple[int, ...]:
    if orders is None:
        orders_tup = _broadcast_fourier_order(2, count=n_periods)
    elif isinstance(orders, int | float):
        orders_tup = _broadcast_fourier_order(orders, count=n_periods)
    elif isinstance(orders, str):
        orders_tup = _normalize_string_fourier_orders(orders, n_periods=n_periods)
    elif isinstance(orders, list | tuple):
        orders_tup = _normalize_sequence_fourier_orders(orders, n_periods=n_periods)
    else:
        orders_tup = _broadcast_fourier_order(orders, count=n_periods)
    return _validate_fourier_orders(orders_tup)


def _append_fourier_period_features(
    tt: np.ndarray,
    period: int,
    order: int,
    feats: list[np.ndarray],
    names: list[str],
) -> None:
    w = (2.0 * np.pi) / float(period)
    for k in range(1, int(order) + 1):
        ang = w * float(k) * tt
        feats.append(np.sin(ang).reshape(-1, 1))
        feats.append(np.cos(ang).reshape(-1, 1))
        names.append(f"fourier_{int(period)}_sin_{k}")
        names.append(f"fourier_{int(period)}_cos_{k}")


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
    dt = _coerce_datetime_series(s)

    feats: list[np.ndarray] = []
    names: list[str] = []
    components = _selected_cyclical_components(
        add_dow=add_dow,
        add_month=add_month,
        add_doy=add_doy,
        add_hour=add_hour,
    )

    if add_time_idx:
        _append_time_index_feature(feats, names, size=len(s))

    if dt is None:
        _append_zero_cyclical_feature_pairs(feats, names, size=len(s), components=components)
        return _finalize_feature_matrix(
            feats,
            names,
            rows=len(s),
            error_message="Non-finite values in generated time features",
        )

    _append_datetime_cyclical_feature_pairs(dt, feats, names, components=components)
    return _finalize_feature_matrix(
        feats,
        names,
        rows=len(s),
        error_message="Non-finite values in generated time features",
    )


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

    periods_tup = _normalize_fourier_periods(periods)
    if not periods_tup:
        return np.empty((int(tt.size), 0), dtype=float), []
    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))

    feats: list[np.ndarray] = []
    names: list[str] = []
    for period, order in zip(periods_tup, orders_tup, strict=True):
        _append_fourier_period_features(tt, period, order, feats, names)

    return _finalize_feature_matrix(
        feats,
        names,
        rows=int(tt.size),
        error_message="Non-finite values in generated Fourier features",
    )
