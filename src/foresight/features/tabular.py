from __future__ import annotations

from typing import Any

import numpy as np


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


def build_lag_derived_features(
    X_lags: Any,
    *,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
) -> tuple[np.ndarray, list[str]]:
    """
    Build derived features from a lag matrix `X_lags` (rows, lags).

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
    X = _as_2d_float_array(X_lags)
    rows = int(X.shape[0])
    lags = int(X.shape[1])
    if rows == 0 or lags == 0:
        return np.empty((rows, 0), dtype=float), []

    windows = tuple(sorted(set(normalize_int_tuple(roll_windows))))
    stats = tuple(sorted(set(s.lower().strip() for s in normalize_str_tuple(roll_stats) if s)))
    diffs = tuple(sorted(set(normalize_int_tuple(diff_lags))))

    allowed = {"mean", "std", "min", "max", "median", "slope", "iqr", "mad", "skew", "kurt"}
    bad = sorted({s for s in stats if s not in allowed})
    if bad:
        raise ValueError(f"roll_stats contains unknown values: {bad}. Allowed: {sorted(allowed)}")

    feats: list[np.ndarray] = []
    names: list[str] = []

    for w in windows:
        ww = int(w)
        if ww <= 0:
            raise ValueError("roll_windows must be >= 1")
        if ww > lags:
            raise ValueError(f"roll_window {ww} exceeds lags={lags}")
        tail = X[:, -ww:]

        if "mean" in stats:
            feats.append(np.mean(tail, axis=1).reshape(-1, 1))
            names.append(f"roll_mean_{ww}")
        if "std" in stats:
            feats.append(np.std(tail, axis=1).reshape(-1, 1))
            names.append(f"roll_std_{ww}")
        if "min" in stats:
            feats.append(np.min(tail, axis=1).reshape(-1, 1))
            names.append(f"roll_min_{ww}")
        if "max" in stats:
            feats.append(np.max(tail, axis=1).reshape(-1, 1))
            names.append(f"roll_max_{ww}")
        if "median" in stats:
            feats.append(np.median(tail, axis=1).reshape(-1, 1))
            names.append(f"roll_median_{ww}")
        if "iqr" in stats:
            q25, q75 = np.percentile(tail, [25.0, 75.0], axis=1)
            feats.append((q75 - q25).reshape(-1, 1))
            names.append(f"roll_iqr_{ww}")
        if "mad" in stats:
            med = np.median(tail, axis=1).reshape(-1, 1)
            feats.append(np.median(np.abs(tail - med), axis=1).reshape(-1, 1))
            names.append(f"roll_mad_{ww}")
        if "skew" in stats or "kurt" in stats:
            mu = np.mean(tail, axis=1, dtype=float).reshape(-1, 1)
            centered = tail - mu
            m2 = np.mean(centered * centered, axis=1, dtype=float).reshape(-1)
            std = np.sqrt(np.maximum(m2, 0.0))
            ok = std > 0.0
            if "skew" in stats:
                m3 = np.mean(centered * centered * centered, axis=1, dtype=float).reshape(-1)
                skew = np.zeros((rows,), dtype=float)
                skew[ok] = m3[ok] / (std[ok] ** 3)
                feats.append(skew.reshape(-1, 1))
                names.append(f"roll_skew_{ww}")
            if "kurt" in stats:
                m4 = np.mean(centered**4, axis=1, dtype=float).reshape(-1)
                kurt = np.zeros((rows,), dtype=float)
                kurt[ok] = (m4[ok] / (std[ok] ** 4)) - 3.0  # excess kurtosis
                feats.append(kurt.reshape(-1, 1))
                names.append(f"roll_kurt_{ww}")
        if "slope" in stats:
            x = np.arange(ww, dtype=float)
            x = x - float(np.mean(x))
            denom = float(np.sum(x * x))
            if denom <= 0.0:
                raise RuntimeError("Internal error: slope denom must be > 0")
            # slope = cov(x,y) / var(x), computed along each row.
            slope = (tail @ x.reshape(-1, 1)).reshape(-1) / denom
            feats.append(slope.reshape(-1, 1))
            names.append(f"roll_slope_{ww}")

    if diffs:
        last = X[:, -1]
        for k in diffs:
            kk = int(k)
            if kk <= 0:
                raise ValueError("diff_lags must be >= 1")
            if kk >= lags:
                raise ValueError(f"diff_lag {kk} must be < lags={lags}")
            feats.append((last - X[:, -1 - kk]).reshape(-1, 1))
            names.append(f"diff_{kk}")

    if not feats:
        return np.empty((rows, 0), dtype=float), []

    F = np.concatenate(feats, axis=1).astype(float, copy=False)
    if not np.all(np.isfinite(F)):
        raise ValueError("Non-finite values in derived lag features")
    return F, names
