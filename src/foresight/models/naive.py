from __future__ import annotations

from typing import Any

import numpy as np

HORIZON_MIN_ERROR = "horizon must be >= 1"


def _validated_naive_input(
    train: Any,
    *,
    horizon: int,
    subject: str,
) -> tuple[np.ndarray, int]:
    x = np.asarray(train, dtype=float)
    if x.size == 0:
        raise ValueError(f"{subject} requires at least 1 training point")
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    return x, h


def naive_last(train: Any, horizon: int) -> np.ndarray:
    x, h = _validated_naive_input(train, horizon=horizon, subject="naive_last")
    return np.full(shape=(h,), fill_value=float(x[-1]), dtype=float)


def seasonal_naive(train: Any, horizon: int, *, season_length: int) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if season_length <= 0:
        raise ValueError("season_length must be >= 1")
    if x.size < season_length:
        raise ValueError(f"seasonal_naive requires at least {season_length} points, got {x.size}")

    last_season = x[-season_length:]
    idx = np.arange(horizon) % season_length
    return last_season[idx].astype(float, copy=False)


def _pearson_corr(x0: np.ndarray, x1: np.ndarray) -> float:
    a = np.asarray(x0, dtype=float).reshape(-1)
    b = np.asarray(x1, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    am = float(np.mean(a))
    bm = float(np.mean(b))
    da = a - am
    db = b - bm
    denom = float(np.sqrt(np.sum(da * da) * np.sum(db * db)))
    if not np.isfinite(denom) or denom <= 0.0:
        return 0.0
    num = float(np.sum(da * db))
    return float(num / denom)


def _best_seasonal_naive_lag(
    scan: np.ndarray,
    *,
    min_season_length: int,
    max_season_length: int,
) -> tuple[int, float] | None:
    min_s = int(min_season_length)
    upper = min(int(max_season_length), int(scan.size // 2))
    if upper < min_s:
        return None

    best_lag = int(min_s)
    best_corr = float("-inf")
    eps = 1e-12
    for lag in range(int(min_s), int(upper) + 1):
        a = scan[:-lag]
        b = scan[lag:]
        corr = _pearson_corr(a, b)
        if (corr > best_corr + eps) or (abs(corr - best_corr) <= eps and lag < best_lag):
            best_corr = float(corr)
            best_lag = int(lag)
    return best_lag, best_corr


def seasonal_naive_auto(
    train: Any,
    horizon: int,
    *,
    min_season_length: int = 2,
    max_season_length: int = 24,
    detrend: bool = True,
    min_corr: float = 0.2,
) -> np.ndarray:
    """
    Auto seasonal-naive baseline: infer season_length via a simple ACF scan.

    If a clear seasonal lag is not detected (or there is insufficient history),
    falls back to `naive_last`.
    """
    x, h = _validated_naive_input(train, horizon=horizon, subject="seasonal_naive_auto")

    min_s = int(min_season_length)
    max_s = int(max_season_length)
    if min_s <= 0:
        raise ValueError("min_season_length must be >= 1")
    if max_s < min_s:
        raise ValueError("max_season_length must be >= min_season_length")

    scan = np.diff(x) if bool(detrend) and x.size >= 3 else x
    best = _best_seasonal_naive_lag(
        scan,
        min_season_length=min_s,
        max_season_length=max_s,
    )
    if best is None:
        return naive_last(x, h)
    best_lag, best_corr = best

    if not np.isfinite(best_corr) or best_corr < float(min_corr):
        return naive_last(x, h)

    return seasonal_naive(x, h, season_length=int(best_lag))
