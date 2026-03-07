from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr


def _require_same_shape(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true{y_true.shape} vs y_pred{y_pred.shape}")


def mae(y_true: Any, y_pred: Any) -> float:
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    _require_same_shape(yt, yp)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: Any, y_pred: Any) -> float:
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    _require_same_shape(yt, yp)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: Any, y_pred: Any, *, eps: float = 1e-8) -> float:
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    _require_same_shape(yt, yp)
    denom = np.where(np.abs(yt) < eps, eps, np.abs(yt))
    return float(np.mean(np.abs((yt - yp) / denom)))


def smape(y_true: Any, y_pred: Any, *, eps: float = 1e-8) -> float:
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    _require_same_shape(yt, yp)
    denom = np.abs(yt) + np.abs(yp)
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(2.0 * np.abs(yt - yp) / denom))


def mse(y_true: Any, y_pred: Any) -> float:
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    _require_same_shape(yt, yp)
    return float(np.mean((yt - yp) ** 2))


def wape(y_true: Any, y_pred: Any, *, eps: float = 1e-8) -> float:
    """
    Weighted Absolute Percentage Error:
        sum(|y - yhat|) / sum(|y|)
    """
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    _require_same_shape(yt, yp)
    denom = float(np.sum(np.abs(yt)))
    if denom < eps:
        denom = float(eps)
    return float(np.sum(np.abs(yt - yp)) / denom)


def _as_1d_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {arr.shape}")
    return arr


def mase(
    y_true: Any,
    y_pred: Any,
    *,
    y_train: Any,
    seasonality: int = 1,
    eps: float = 1e-8,
) -> float:
    """
    Mean Absolute Scaled Error (MASE).

    Scales MAE by the in-sample seasonal naive MAE.
    """
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    _require_same_shape(yt, yp)

    train = _as_1d_float_array(y_train)
    if seasonality <= 0:
        raise ValueError("seasonality must be >= 1")
    if train.size <= seasonality:
        raise ValueError("y_train too short for the requested seasonality")

    diffs = np.abs(train[seasonality:] - train[:-seasonality])
    scale = float(np.mean(diffs))
    if scale < eps:
        scale = float(eps)

    return float(np.mean(np.abs(yt - yp)) / scale)


def rmsse(
    y_true: Any,
    y_pred: Any,
    *,
    y_train: Any,
    seasonality: int = 1,
    eps: float = 1e-8,
) -> float:
    """
    Root Mean Squared Scaled Error (RMSSE).

    Scales RMSE by the in-sample seasonal naive RMSE (squared diffs mean).
    """
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    _require_same_shape(yt, yp)

    train = _as_1d_float_array(y_train)
    if seasonality <= 0:
        raise ValueError("seasonality must be >= 1")
    if train.size <= seasonality:
        raise ValueError("y_train too short for the requested seasonality")

    diffs2 = (train[seasonality:] - train[:-seasonality]) ** 2
    scale = float(np.mean(diffs2))
    if scale < eps:
        scale = float(eps)

    return float(np.sqrt(np.mean((yt - yp) ** 2) / scale))


def pinball_loss(y_true: Any, y_pred: Any, *, q: float) -> float:
    """
    Pinball loss (quantile loss) for quantile forecasts.

    For quantile q in (0,1):
      L_q(y, yhat) = max(q*(y - yhat), (q-1)*(y - yhat))
    """
    q_f = float(q)
    if not (0.0 < q_f < 1.0):
        raise ValueError("q must be in (0, 1)")

    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    _require_same_shape(yt, yp)

    u = yt - yp
    return float(np.mean(np.maximum(q_f * u, (q_f - 1.0) * u)))


def interval_coverage(y_true: Any, lower: Any, upper: Any) -> float:
    """
    Fraction of observations that fall inside [lower, upper].
    """
    yt = _as_float_array(y_true)
    lo = _as_float_array(lower)
    hi = _as_float_array(upper)
    _require_same_shape(yt, lo)
    _require_same_shape(yt, hi)
    return float(np.mean((yt >= lo) & (yt <= hi)))


def mean_interval_width(lower: Any, upper: Any) -> float:
    lo = _as_float_array(lower)
    hi = _as_float_array(upper)
    _require_same_shape(lo, hi)
    return float(np.mean(hi - lo))


def interval_score(y_true: Any, lower: Any, upper: Any, *, alpha: float) -> float:
    """
    Interval score (Gneiting & Raftery) for a central (1-alpha) interval.
    """
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    yt = _as_float_array(y_true)
    lo = _as_float_array(lower)
    hi = _as_float_array(upper)
    _require_same_shape(yt, lo)
    _require_same_shape(yt, hi)

    width = hi - lo
    below = (lo - yt) * (yt < lo)
    above = (yt - hi) * (yt > hi)
    score = width + (2.0 / a) * below + (2.0 / a) * above
    return float(np.mean(score))


def winkler_score(y_true: Any, lower: Any, upper: Any, *, alpha: float) -> float:
    """
    Winkler score alias for the interval score used for central prediction intervals.
    """
    return interval_score(y_true, lower, upper, alpha=float(alpha))


def weighted_interval_score(
    y_true: Any,
    median: Any,
    *,
    intervals: list[tuple[Any, Any, float]] | tuple[tuple[Any, Any, float], ...],
) -> float:
    """
    Weighted Interval Score (WIS) from a median forecast and one or more central intervals.

    Uses the standard weighted combination:
      (0.5 * |y - median| + sum_k (alpha_k / 2) * IS_alpha_k) / (K + 0.5)
    """
    if not intervals:
        raise ValueError("intervals must be non-empty")

    yt = _as_float_array(y_true)
    med = _as_float_array(median)
    _require_same_shape(yt, med)

    total = 0.5 * float(np.mean(np.abs(yt - med)))
    k = 0
    for lower, upper, alpha in intervals:
        a = float(alpha)
        if not (0.0 < a < 1.0):
            raise ValueError("interval alpha must be in (0, 1)")
        total += (a / 2.0) * float(interval_score(yt, lower, upper, alpha=a))
        k += 1

    return float(total / (float(k) + 0.5))


def crps_from_quantiles(
    y_true: Any,
    quantile_forecasts: Any,
    *,
    quantiles: tuple[float, ...] | list[float],
) -> float:
    """
    Approximate CRPS from a grid of quantile forecasts using
    `2 * mean(pinball losses across quantiles)`.

    Expected `quantile_forecasts` shape: (n_obs, n_quantiles). A 1D array is also accepted
    when `y_true` has a single observation.
    """
    yt = _as_float_array(y_true).reshape(-1)
    qs = tuple(float(q) for q in quantiles)
    if not qs:
        raise ValueError("quantiles must be non-empty")
    if any(not (0.0 < q < 1.0) for q in qs):
        raise ValueError("quantiles must be in (0, 1)")

    qhat = np.asarray(quantile_forecasts, dtype=float)
    if qhat.ndim == 1:
        qhat = qhat.reshape(1, -1)
    if qhat.shape != (yt.size, len(qs)):
        raise ValueError(
            f"quantile_forecasts must have shape ({yt.size}, {len(qs)}), got {qhat.shape}"
        )

    pinballs = [pinball_loss(yt, qhat[:, i], q=qs[i]) for i in range(len(qs))]
    return float(2.0 * np.mean(np.asarray(pinballs, dtype=float)))


def msis(
    y_true: Any,
    lower: Any,
    upper: Any,
    *,
    y_train: Any,
    seasonality: int = 1,
    alpha: float = 0.1,
    eps: float = 1e-8,
) -> float:
    """
    Mean Scaled Interval Score (MSIS-like), scaling `interval_score` by in-sample
    seasonal naive MAE (same scale as MASE).
    """
    train = _as_1d_float_array(y_train)
    if seasonality <= 0:
        raise ValueError("seasonality must be >= 1")
    if train.size <= seasonality:
        raise ValueError("y_train too short for the requested seasonality")

    diffs = np.abs(train[seasonality:] - train[:-seasonality])
    scale = float(np.mean(diffs))
    if scale < eps:
        scale = float(eps)

    return float(interval_score(y_true, lower, upper, alpha=float(alpha)) / scale)
