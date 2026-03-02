from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    interval_coverage,
    interval_score,
    mae,
    mape,
    mean_interval_width,
    pinball_loss,
    rmse,
    smape,
)


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

    out: dict[str, Any] = {
        "n_points": int(y.size),
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "mape": mape(y, yhat),
        "smape": smape(y, yhat),
    }

    if step_col in df.columns:
        steps = np.asarray(df[step_col])
        uniq = sorted({int(s) for s in steps.tolist()})
        mae_by_step: list[float] = []
        rmse_by_step: list[float] = []
        mape_by_step: list[float] = []
        smape_by_step: list[float] = []
        for s in uniq:
            mask = steps == s
            ys = y[mask]
            yps = yhat[mask]
            mae_by_step.append(mae(ys, yps))
            rmse_by_step.append(rmse(ys, yps))
            mape_by_step.append(mape(ys, yps))
            smape_by_step.append(smape(ys, yps))

        out.update(
            {
                "steps": uniq,
                "mae_by_step": mae_by_step,
                "rmse_by_step": rmse_by_step,
                "mape_by_step": mape_by_step,
                "smape_by_step": smape_by_step,
            }
        )

    return out


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

    # Detect quantile columns.
    q_cols: dict[int, str] = {}
    for c in df.columns:
        if not str(c).startswith(str(yhat_prefix)):
            continue
        pct_s = str(c)[len(str(yhat_prefix)) :]
        if not pct_s.isdigit():
            continue
        pct = int(pct_s)
        if 0 < pct < 100:
            q_cols[pct] = str(c)

    if not q_cols:
        return {"quantiles": [], "pinball_mean": float("nan")}

    pcts = sorted(q_cols)

    out: dict[str, Any] = {"quantiles": pcts}

    pinballs: list[float] = []
    for pct in pcts:
        q = float(pct) / 100.0
        yp = df[q_cols[pct]].to_numpy(dtype=float, copy=False)
        pb = pinball_loss(y, yp, q=q)
        out[f"pinball_p{pct}"] = float(pb)
        pinballs.append(float(pb))

    out["pinball_mean"] = float(np.mean(np.asarray(pinballs, dtype=float)))

    # Interval metrics for symmetric pairs: pX / p(100-X) -> central level = 100 - 2X
    levels: list[int] = []
    for lo_pct in pcts:
        if lo_pct >= 50:
            continue
        hi_pct = 100 - lo_pct
        if hi_pct not in q_cols:
            continue
        level = 100 - 2 * lo_pct
        levels.append(int(level))

    levels = sorted(set(levels))
    out["interval_levels"] = levels

    for level in levels:
        lo_pct = (100 - level) // 2
        hi_pct = 100 - lo_pct
        lo = df[q_cols[int(lo_pct)]].to_numpy(dtype=float, copy=False)
        hi = df[q_cols[int(hi_pct)]].to_numpy(dtype=float, copy=False)
        alpha = 1.0 - float(level) / 100.0
        out[f"coverage_{level}"] = interval_coverage(y, lo, hi)
        out[f"mean_width_{level}"] = mean_interval_width(lo, hi)
        out[f"interval_score_{level}"] = interval_score(y, lo, hi, alpha=alpha)

        if step_col in df.columns:
            steps = np.asarray(df[step_col])
            uniq = sorted({int(s) for s in steps.tolist()})
            cov_by_step: list[float] = []
            width_by_step: list[float] = []
            score_by_step: list[float] = []
            for s in uniq:
                mask = steps == s
                cov_by_step.append(interval_coverage(y[mask], lo[mask], hi[mask]))
                width_by_step.append(mean_interval_width(lo[mask], hi[mask]))
                score_by_step.append(interval_score(y[mask], lo[mask], hi[mask], alpha=alpha))
            out[f"coverage_{level}_by_step"] = cov_by_step
            out[f"mean_width_{level}_by_step"] = width_by_step
            out[f"interval_score_{level}_by_step"] = score_by_step

    return out
