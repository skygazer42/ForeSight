from __future__ import annotations

from typing import Any

import numpy as np

_ROLL_STAT_ORDER = ("mean", "std", "min", "max", "median", "iqr", "mad", "skew", "kurt", "slope")
_ALLOWED_ROLL_STATS = frozenset(_ROLL_STAT_ORDER)


def _as_2d_float_array(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if arr.size == 0:
        return arr.astype(float, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Non-finite values in lag matrix")
    return arr.astype(float, copy=False)


def normalize_int_tuple(items: Any) -> tuple[int, ...]:
    """
    Normalize user input into a tuple of ints.

    Accepts:
      - None / "" -> ()
      - 7 -> (7,)
      - "3,7,14" -> (3,7,14)
      - (3,7) / [3,7] -> (3,7)
    """
    if items is None:
        return ()

    if isinstance(items, str):
        s = items.strip()
        if not s:
            return ()
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return tuple(int(float(p)) for p in parts)

    if isinstance(items, list | tuple):
        out: list[int] = []
        for it in items:
            if it is None:
                continue
            out.append(int(float(it)))
        return tuple(out)

    return (int(float(items)),)


def normalize_lag_steps(
    items: Any,
    *,
    allow_zero: bool = False,
    name: str = "lags",
) -> tuple[int, ...]:
    """
    Normalize lag specs into deterministic descending lag offsets.

    Scalar ints are interpreted as window lengths:
      - allow_zero=False -> n => (n, ..., 1)
      - allow_zero=True  -> n => (n-1, ..., 0)

    Explicit lists/tuples/strings are treated as lag offsets directly.
    """
    if items is None:
        return ()

    if isinstance(items, str | list | tuple):
        values = normalize_int_tuple(items)
        if not values:
            return ()
        lower = 0 if allow_zero else 1
        if any(int(v) < lower for v in values):
            if allow_zero:
                raise ValueError(f"{name} must be >= 0")
            raise ValueError(f"{name} must be >= 1")
        return tuple(sorted({int(v) for v in values}, reverse=True))

    n = int(float(items))
    if n <= 0:
        raise ValueError(f"{name} must be >= 1")
    if allow_zero:
        return tuple(range(n - 1, -1, -1))
    return tuple(range(n, 0, -1))


def normalize_str_tuple(items: Any) -> tuple[str, ...]:
    """
    Normalize user input into a tuple of non-empty strings.
    """
    if items is None:
        return ()

    if isinstance(items, str):
        s = items.strip()
        if not s:
            return ()
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return tuple(parts)

    if isinstance(items, list | tuple):
        out: list[str] = []
        for it in items:
            s = str(it).strip()
            if s:
                out.append(s)
        return tuple(out)

    s = str(items).strip()
    return (s,) if s else ()


def _prepare_lag_feature_inputs(values: Any, t: Any) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("Non-finite values in lag feature source")

    tt = np.asarray(t, dtype=int).reshape(-1)
    if tt.size == 0:
        return arr, tt
    if np.any(tt < 0):
        raise ValueError("t indices must be >= 0")
    if int(np.max(tt)) >= int(arr.shape[0]):
        raise ValueError("t indices must be < len(values)")
    return arr, tt


def _resolve_column_names(column_names: Any, *, width: int) -> list[str]:
    names_in = normalize_str_tuple(column_names)
    if names_in:
        if len(names_in) != int(width):
            raise ValueError("column_names must match values.shape[1]")
        return list(names_in)
    return [f"col{i}" for i in range(int(width))]


def _build_lag_feature_blocks(
    arr: np.ndarray,
    tt: np.ndarray,
    lag_steps: tuple[int, ...],
    cols: list[str],
    *,
    prefix: str,
) -> tuple[list[np.ndarray], list[str]]:
    feats: list[np.ndarray] = []
    names: list[str] = []
    for lag in lag_steps:
        idx = tt - int(lag)
        if np.any(idx < 0):
            raise ValueError(f"{prefix}_lags require t >= lag")
        feats.append(arr[idx, :].astype(float, copy=False))
        names.extend(f"{prefix}_{col}_lag{int(lag)}" for col in cols)
    return feats, names


def _normalize_roll_stat_names(roll_stats: Any) -> tuple[str, ...]:
    stats = tuple(sorted({s.lower().strip() for s in normalize_str_tuple(roll_stats) if s}))
    bad = sorted({s for s in stats if s not in _ALLOWED_ROLL_STATS})
    if bad:
        raise ValueError(
            f"roll_stats contains unknown values: {bad}. Allowed: {sorted(_ALLOWED_ROLL_STATS)}"
        )
    return stats


def _normalize_roll_feature_config(
    roll_windows: Any,
    roll_stats: Any,
    diff_lags: Any,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
    windows = tuple(sorted(set(normalize_int_tuple(roll_windows))))
    stats = _normalize_roll_stat_names(roll_stats)
    diffs = tuple(sorted(set(normalize_int_tuple(diff_lags))))
    return windows, stats, diffs


def _validate_roll_window(window: int, *, lags: int) -> int:
    ww = int(window)
    if ww <= 0:
        raise ValueError("roll_windows must be >= 1")
    if ww > lags:
        raise ValueError(f"roll_window {ww} exceeds lags={lags}")
    return ww


def _roll_mean_feature(tail: np.ndarray) -> np.ndarray:
    return np.mean(tail, axis=1).reshape(-1, 1)


def _roll_std_feature(tail: np.ndarray) -> np.ndarray:
    return np.std(tail, axis=1).reshape(-1, 1)


def _roll_min_feature(tail: np.ndarray) -> np.ndarray:
    return np.min(tail, axis=1).reshape(-1, 1)


def _roll_max_feature(tail: np.ndarray) -> np.ndarray:
    return np.max(tail, axis=1).reshape(-1, 1)


def _roll_median_feature(tail: np.ndarray) -> np.ndarray:
    return np.median(tail, axis=1).reshape(-1, 1)


def _roll_iqr_feature(tail: np.ndarray) -> np.ndarray:
    q25, q75 = np.percentile(tail, [25.0, 75.0], axis=1)
    return (q75 - q25).reshape(-1, 1)


def _roll_mad_feature(tail: np.ndarray) -> np.ndarray:
    med = np.median(tail, axis=1).reshape(-1, 1)
    return np.median(np.abs(tail - med), axis=1).reshape(-1, 1)


def _roll_slope_feature(tail: np.ndarray) -> np.ndarray:
    width = int(tail.shape[1])
    x = np.arange(width, dtype=float)
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x))
    if denom <= 0.0:
        raise RuntimeError("Internal error: slope denom must be > 0")
    slope = (tail @ x.reshape(-1, 1)).reshape(-1) / denom
    return slope.reshape(-1, 1)


def _roll_moment_context(tail: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(tail, axis=1, dtype=float).reshape(-1, 1)
    centered = tail - mu
    m2 = np.mean(centered * centered, axis=1, dtype=float).reshape(-1)
    std = np.sqrt(np.maximum(m2, 0.0))
    ok = std > 0.0
    return centered, std, ok


def _roll_skew_feature(
    tail: np.ndarray,
    moment_context: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if moment_context is None:
        moment_context = _roll_moment_context(tail)
    centered, std, ok = moment_context
    rows = int(tail.shape[0])
    m3 = np.mean(centered * centered * centered, axis=1, dtype=float).reshape(-1)
    skew = np.zeros((rows,), dtype=float)
    skew[ok] = m3[ok] / (std[ok] ** 3)
    return skew.reshape(-1, 1)


def _roll_kurt_feature(
    tail: np.ndarray,
    moment_context: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if moment_context is None:
        moment_context = _roll_moment_context(tail)
    centered, std, ok = moment_context
    rows = int(tail.shape[0])
    m4 = np.mean(centered**4, axis=1, dtype=float).reshape(-1)
    kurt = np.zeros((rows,), dtype=float)
    kurt[ok] = (m4[ok] / (std[ok] ** 4)) - 3.0
    return kurt.reshape(-1, 1)


_ROLL_STAT_BUILDERS = {
    "mean": _roll_mean_feature,
    "std": _roll_std_feature,
    "min": _roll_min_feature,
    "max": _roll_max_feature,
    "median": _roll_median_feature,
    "iqr": _roll_iqr_feature,
    "mad": _roll_mad_feature,
    "slope": _roll_slope_feature,
}


def _build_roll_stat_feature(
    tail: np.ndarray,
    stat_name: str,
    moment_context: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if stat_name == "skew":
        return _roll_skew_feature(tail, moment_context)
    if stat_name == "kurt":
        return _roll_kurt_feature(tail, moment_context)
    return _ROLL_STAT_BUILDERS[stat_name](tail)


def _append_roll_stat_features(
    tail: np.ndarray,
    ww: int,
    stats: tuple[str, ...],
    feats: list[np.ndarray],
    names: list[str],
) -> None:
    if not stats:
        return
    stat_set = set(stats)
    moment_context = None
    if "skew" in stat_set or "kurt" in stat_set:
        moment_context = _roll_moment_context(tail)
    for stat_name in _ROLL_STAT_ORDER:
        if stat_name not in stat_set:
            continue
        feats.append(_build_roll_stat_feature(tail, stat_name, moment_context))
        names.append(f"roll_{stat_name}_{ww}")


def _append_diff_features(
    X: np.ndarray,
    diffs: tuple[int, ...],
    feats: list[np.ndarray],
    names: list[str],
    *,
    lags: int,
) -> None:
    if not diffs:
        return
    last = X[:, -1]
    for k in diffs:
        kk = int(k)
        if kk <= 0:
            raise ValueError("diff_lags must be >= 1")
        if kk >= lags:
            raise ValueError(f"diff_lag {kk} must be < lags={lags}")
        feats.append((last - X[:, -1 - kk]).reshape(-1, 1))
        names.append(f"diff_{kk}")


def _finalize_feature_matrix(
    feats: list[np.ndarray],
    names: list[str],
    *,
    rows: int,
    error_message: str,
) -> tuple[np.ndarray, list[str]]:
    if not feats:
        return np.empty((rows, 0), dtype=float), []
    matrix = np.concatenate(feats, axis=1).astype(float, copy=False)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(error_message)
    return matrix, names


def build_column_lag_features(
    values: Any,
    *,
    t: Any,
    lags: Any,
    column_names: Any = (),
    prefix: str = "x",
    allow_zero: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Build lagged column features from a 1D/2D source for target indices `t`.
    """
    arr, tt = _prepare_lag_feature_inputs(values, t)
    if tt.size == 0:
        return np.empty((0, 0), dtype=float), []

    lag_steps = normalize_lag_steps(lags, allow_zero=allow_zero, name=f"{prefix}_lags")
    if not lag_steps:
        return np.empty((int(tt.size), 0), dtype=float), []

    cols = _resolve_column_names(column_names, width=int(arr.shape[1]))
    feats, names = _build_lag_feature_blocks(arr, tt, lag_steps, cols, prefix=prefix)
    return _finalize_feature_matrix(
        feats,
        names,
        rows=int(tt.size),
        error_message="Non-finite values in generated lag features",
    )


def build_lag_derived_features(
    lag_matrix: Any,
    *,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
) -> tuple[np.ndarray, list[str]]:
    """
    Build derived features from a lag matrix `lag_matrix` (rows, lags).

    The design goal is simple, leakage-safe feature engineering that only depends on the
    lag window values themselves (no future targets).

    Parameters:
      - roll_windows: window sizes (<= lags) applied to the *tail* of the lag vector.
      - roll_stats: stats to compute per window. Supported:
          mean, std, min, max, median, slope
      - diff_lags: compute last-minus-previous differences:
          diff_k = X[:, -1] - X[:, -1-k]   for k in diff_lags

    Returns:
      (F, names)
        - F: float array of shape (rows, n_features)
        - names: aligned feature names
    """
    X = _as_2d_float_array(lag_matrix)
    rows = int(X.shape[0])
    lags = int(X.shape[1])
    if rows == 0 or lags == 0:
        return np.empty((rows, 0), dtype=float), []

    windows, stats, diffs = _normalize_roll_feature_config(roll_windows, roll_stats, diff_lags)
    feats: list[np.ndarray] = []
    names: list[str] = []

    for w in windows:
        ww = _validate_roll_window(int(w), lags=lags)
        tail = X[:, -ww:]
        _append_roll_stat_features(tail, ww, stats, feats, names)

    _append_diff_features(X, diffs, feats, names, lags=lags)
    return _finalize_feature_matrix(
        feats,
        names,
        rows=rows,
        error_message="Non-finite values in derived lag features",
    )
