from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    crps_from_quantiles,
    weighted_interval_score,
)


def _sorted_unique_steps(step_values: Any) -> list[int]:
    steps = np.asarray(step_values)
    return sorted({int(step) for step in steps.tolist()})


def _step_group_inverse_counts(step_values: Any) -> tuple[list[int], np.ndarray, np.ndarray]:
    steps = np.asarray(step_values)
    uniq = _sorted_unique_steps(steps)
    if not uniq:
        return [], np.empty((0,), dtype=int), np.empty((0,), dtype=float)

    step_index = {int(step): idx for idx, step in enumerate(uniq)}
    inverse = np.asarray([step_index[int(step)] for step in steps], dtype=int)
    counts = np.bincount(inverse, minlength=len(uniq)).astype(float, copy=False)
    return uniq, inverse, counts


def _mean_by_step_from_inverse(
    values: np.ndarray,
    *,
    inverse: np.ndarray,
    counts: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    return np.bincount(inverse, weights=values, minlength=n_steps) / counts


def _quantile_column_map(df: pd.DataFrame, *, yhat_prefix: str) -> dict[int, str]:
    q_cols: dict[int, str] = {}
    prefix = str(yhat_prefix)
    for column in df.columns:
        column_s = str(column)
        if not column_s.startswith(prefix):
            continue
        pct_s = column_s[len(prefix) :]
        if not pct_s.isdigit():
            continue
        pct = int(pct_s)
        if 0 < pct < 100:
            q_cols[pct] = column_s
    return q_cols


def _symmetric_interval_levels(pcts: list[int], q_cols: dict[int, str]) -> list[int]:
    levels: list[int] = []
    for lo_pct in pcts:
        if lo_pct >= 50:
            continue
        hi_pct = 100 - lo_pct
        if hi_pct in q_cols:
            levels.append(100 - 2 * lo_pct)
    return sorted(set(levels))


def _validated_interval_arrays(
    y: Any,
    lo: Any,
    hi: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=float)
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)
    if y_arr.shape != lo_arr.shape or y_arr.shape != hi_arr.shape:
        raise ValueError(f"Shape mismatch: y{y_arr.shape} lo{lo_arr.shape} hi{hi_arr.shape}")
    return y_arr, lo_arr, hi_arr


def _interval_score_vector(
    y: Any,
    lo: Any,
    hi: Any,
    *,
    alpha: float,
) -> np.ndarray:
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    y_arr, lo_arr, hi_arr = _validated_interval_arrays(y, lo, hi)

    width = hi_arr - lo_arr
    below = (lo_arr - y_arr) * (y_arr < lo_arr)
    above = (y_arr - hi_arr) * (y_arr > hi_arr)
    return width + (2.0 / a) * below + (2.0 / a) * above


def _vectorized_interval_metrics(
    y: Any,
    lo: Any,
    hi: Any,
    *,
    alpha: float,
    step_values: Any | None = None,
) -> dict[str, Any]:
    y_arr, lo_arr, hi_arr = _validated_interval_arrays(y, lo, hi)

    width = hi_arr - lo_arr
    score = _interval_score_vector(y_arr, lo_arr, hi_arr, alpha=alpha)
    coverage = ((y_arr >= lo_arr) & (y_arr <= hi_arr)).astype(float)

    out: dict[str, Any] = {
        "coverage": float(np.mean(coverage)),
        "mean_width": float(np.mean(width)),
        "interval_score": float(np.mean(score)),
        "winkler_score": float(np.mean(score)),
    }
    if step_values is None:
        return out

    uniq, inverse, counts = _step_group_inverse_counts(step_values)
    if not uniq:
        return out
    n_steps = len(uniq)

    out.update(
        {
            "coverage_by_step": _mean_by_step_from_inverse(
                coverage, inverse=inverse, counts=counts, n_steps=n_steps
            ).tolist(),
            "mean_width_by_step": _mean_by_step_from_inverse(
                width, inverse=inverse, counts=counts, n_steps=n_steps
            ).tolist(),
            "interval_score_by_step": _mean_by_step_from_inverse(
                score, inverse=inverse, counts=counts, n_steps=n_steps
            ).tolist(),
            "winkler_score_by_step": _mean_by_step_from_inverse(
                score, inverse=inverse, counts=counts, n_steps=n_steps
            ).tolist(),
        }
    )
    return out


def _per_step_interval_metrics(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    alpha: float,
    step_values: Any,
) -> tuple[list[float], list[float], list[float], list[float]]:
    payload = _vectorized_interval_metrics(
        y,
        lo,
        hi,
        alpha=alpha,
        step_values=step_values,
    )
    return (
        list(payload["coverage_by_step"]),
        list(payload["mean_width_by_step"]),
        list(payload["interval_score_by_step"]),
        list(payload["winkler_score_by_step"]),
    )


def _weighted_interval_score_by_step(
    y: np.ndarray,
    median: np.ndarray,
    *,
    wis_intervals: list[tuple[np.ndarray, np.ndarray, float]],
    step_values: Any,
) -> list[float]:
    if not wis_intervals:
        raise ValueError("wis_intervals must be non-empty")

    y_arr = np.asarray(y, dtype=float).reshape(-1)
    median_arr = np.asarray(median, dtype=float).reshape(-1)
    if y_arr.shape != median_arr.shape:
        raise ValueError(f"Shape mismatch: y{y_arr.shape} median{median_arr.shape}")

    uniq, inverse, counts = _step_group_inverse_counts(step_values)
    if not uniq:
        return []
    n_steps = len(uniq)

    total = 0.5 * _mean_by_step_from_inverse(
        np.abs(y_arr - median_arr),
        inverse=inverse,
        counts=counts,
        n_steps=n_steps,
    )
    k = 0

    for lo, hi, alpha in wis_intervals:
        _y_arr, lo_arr, hi_arr = _validated_interval_arrays(y_arr, lo, hi)
        score = _interval_score_vector(y_arr, lo_arr, hi_arr, alpha=alpha)
        total += (float(alpha) / 2.0) * _mean_by_step_from_inverse(
            score,
            inverse=inverse,
            counts=counts,
            n_steps=n_steps,
        )
        k += 1

    return (total / (float(k) + 0.5)).tolist()


def _vectorized_pinball_summary(
    y: Any,
    quantile_forecasts: Any,
    pcts: list[int],
) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    qhat = np.asarray(quantile_forecasts, dtype=float)
    if qhat.ndim != 2:
        raise ValueError(f"Expected quantile_forecasts to be 2D, got shape {qhat.shape}")
    if qhat.shape[0] != y_arr.size:
        raise ValueError(
            f"Row mismatch: y has {y_arr.size} rows but quantile_forecasts has shape {qhat.shape}"
        )
    if qhat.shape[1] != len(pcts):
        raise ValueError(f"Column mismatch: expected {len(pcts)} quantiles, got shape {qhat.shape}")

    qs = np.asarray([float(pct) / 100.0 for pct in pcts], dtype=float)
    residual = y_arr[:, None] - qhat
    pinball_matrix = np.maximum(qs[None, :] * residual, (qs[None, :] - 1.0) * residual)
    pinball_means = np.mean(pinball_matrix, axis=0)

    out: dict[str, Any] = {"quantiles": list(pcts)}
    for idx, pct in enumerate(pcts):
        out[f"pinball_p{pct}"] = float(pinball_means[idx])
    out["pinball_mean"] = float(np.mean(pinball_means))
    out["crps"] = float(crps_from_quantiles(y_arr, qhat, quantiles=tuple(qs.tolist())))
    return out


def _vectorized_point_metrics(
    y: np.ndarray,
    yhat: np.ndarray,
    *,
    step_values: Any | None = None,
    eps: float = 1e-8,
) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=float)
    yhat_arr = np.asarray(yhat, dtype=float)
    if y_arr.shape != yhat_arr.shape:
        raise ValueError(f"Shape mismatch: y_true{y_arr.shape} vs y_pred{yhat_arr.shape}")

    error = y_arr - yhat_arr
    abs_error = np.abs(error)
    sq_error = error**2
    mape_denom = np.where(np.abs(y_arr) < eps, eps, np.abs(y_arr))
    abs_pct_error = abs_error / mape_denom
    smape_denom = np.abs(y_arr) + np.abs(yhat_arr)
    smape_denom = np.where(smape_denom < eps, eps, smape_denom)
    smape_terms = 2.0 * abs_error / smape_denom

    out: dict[str, Any] = {
        "n_points": int(y_arr.size),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(sq_error))),
        "mape": float(np.mean(abs_pct_error)),
        "smape": float(np.mean(smape_terms)),
    }

    if step_values is None:
        return out

    uniq, inverse, counts = _step_group_inverse_counts(step_values)
    if not uniq:
        return out
    n_steps = len(uniq)

    out.update(
        {
            "steps": uniq,
            "mae_by_step": _mean_by_step_from_inverse(
                abs_error, inverse=inverse, counts=counts, n_steps=n_steps
            ).tolist(),
            "rmse_by_step": np.sqrt(
                _mean_by_step_from_inverse(
                    sq_error, inverse=inverse, counts=counts, n_steps=n_steps
                )
            ).tolist(),
            "mape_by_step": _mean_by_step_from_inverse(
                abs_pct_error, inverse=inverse, counts=counts, n_steps=n_steps
            ).tolist(),
            "smape_by_step": _mean_by_step_from_inverse(
                smape_terms, inverse=inverse, counts=counts, n_steps=n_steps
            ).tolist(),
        }
    )
    return out


def evaluate_predictions(
    df: pd.DataFrame,
    *,
    y_col: str = "y",
    yhat_col: str = "yhat",
    step_col: str = "step",
) -> dict[str, Any]:
    """
    Compute standard point-forecast metrics from a tidy predictions table.

    Expected columns (defaults):
      - y: true values
      - yhat: predictions
      - step: horizon step index (1..H), optional but recommended
    """
    if y_col not in df.columns:
        raise KeyError(f"Missing y column: {y_col!r}")
    if yhat_col not in df.columns:
        raise KeyError(f"Missing yhat column: {yhat_col!r}")

    y = df[y_col].to_numpy(dtype=float, copy=False)
    yhat = df[yhat_col].to_numpy(dtype=float, copy=False)
    steps = df[step_col] if step_col in df.columns else None
    return _vectorized_point_metrics(y, yhat, step_values=steps)


def evaluate_quantile_predictions(
    df: pd.DataFrame,
    *,
    y_col: str = "y",
    step_col: str = "step",
    yhat_prefix: str = "yhat_p",
) -> dict[str, Any]:
    """
    Compute basic probabilistic metrics from a tidy predictions table that includes
    quantile columns.

    Column convention:
      - `yhat_p{pct}` for percentile forecasts, e.g. yhat_p10, yhat_p50, yhat_p90

    Returns:
      - pinball loss per quantile + mean across quantiles
      - interval coverage/width/score for symmetric quantile pairs (pX, p(100-X))
    """
    if y_col not in df.columns:
        raise KeyError(f"Missing y column: {y_col!r}")

    y = df[y_col].to_numpy(dtype=float, copy=False)
    q_cols = _quantile_column_map(df, yhat_prefix=yhat_prefix)

    if not q_cols:
        return {"quantiles": [], "pinball_mean": float("nan")}

    pcts = sorted(q_cols)
    qhat_mat = np.column_stack([df[q_cols[pct]].to_numpy(dtype=float, copy=False) for pct in pcts])
    out = _vectorized_pinball_summary(y, qhat_mat, pcts)

    levels = _symmetric_interval_levels(pcts, q_cols)
    out["interval_levels"] = levels

    center_col = None
    if 50 in q_cols:
        center_col = q_cols[50]
    elif "yhat" in df.columns:
        center_col = "yhat"
    elif pcts:
        center_col = q_cols[min(pcts, key=lambda p: abs(p - 50))]

    wis_intervals: list[tuple[np.ndarray, np.ndarray, float]] = []

    for level in levels:
        lo_pct = (100 - level) // 2
        hi_pct = 100 - lo_pct
        lo = df[q_cols[int(lo_pct)]].to_numpy(dtype=float, copy=False)
        hi = df[q_cols[int(hi_pct)]].to_numpy(dtype=float, copy=False)
        alpha = 1.0 - float(level) / 100.0
        interval_payload = _vectorized_interval_metrics(y, lo, hi, alpha=alpha)
        out[f"coverage_{level}"] = float(interval_payload["coverage"])
        out[f"mean_width_{level}"] = float(interval_payload["mean_width"])
        out[f"interval_score_{level}"] = float(interval_payload["interval_score"])
        out[f"winkler_score_{level}"] = float(interval_payload["winkler_score"])
        wis_intervals.append((lo, hi, alpha))

        if step_col in df.columns:
            cov_by_step, width_by_step, score_by_step, winkler_by_step = _per_step_interval_metrics(
                y,
                lo,
                hi,
                alpha=alpha,
                step_values=df[step_col],
            )
            out[f"coverage_{level}_by_step"] = cov_by_step
            out[f"mean_width_{level}_by_step"] = width_by_step
            out[f"interval_score_{level}_by_step"] = score_by_step
            out[f"winkler_score_{level}_by_step"] = winkler_by_step

    if center_col is not None and wis_intervals:
        median = df[center_col].to_numpy(dtype=float, copy=False)
        out["weighted_interval_score"] = float(
            weighted_interval_score(y, median, intervals=wis_intervals)
        )
        if step_col in df.columns:
            out["weighted_interval_score_by_step"] = _weighted_interval_score_by_step(
                y,
                median,
                wis_intervals=wis_intervals,
                step_values=df[step_col],
            )

    return out
