from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import interval_coverage, interval_score, mean_interval_width, winkler_score


def _validate_levels(levels: tuple[float, ...]) -> tuple[float, ...]:
    if not levels:
        raise ValueError("levels must be non-empty")
    out: list[float] = []
    for lv in levels:
        f = float(lv)
        if not (0.0 < f < 1.0):
            raise ValueError("levels must be in (0, 1)")
        out.append(f)
    return tuple(sorted(set(out)))


@dataclass(frozen=True)
class ConformalIntervals:
    """
    Symmetric conformal intervals using absolute residual quantiles.

    If `per_step=True`, stores a separate radius per horizon step.
    Otherwise uses a single radius pooled across steps.
    """

    levels: tuple[float, ...]
    per_step: bool
    steps: tuple[int, ...]
    radius: dict[float, np.ndarray]  # level -> (len(steps),) array

    def interval_columns(self) -> list[str]:
        cols: list[str] = []
        for lv in self.levels:
            p = int(round(lv * 100))
            cols.append(f"yhat_lo_{p}")
            cols.append(f"yhat_hi_{p}")
        return cols


def fit_conformal_intervals(
    df: pd.DataFrame,
    *,
    y_col: str = "y",
    yhat_col: str = "yhat",
    step_col: str = "step",
    levels: tuple[float, ...] = (0.8, 0.9),
    per_step: bool = True,
) -> ConformalIntervals:
    """
    Fit symmetric conformal intervals from a predictions table.
    """
    if y_col not in df.columns:
        raise KeyError(f"Missing y column: {y_col!r}")
    if yhat_col not in df.columns:
        raise KeyError(f"Missing yhat column: {yhat_col!r}")
    if step_col not in df.columns:
        raise KeyError(f"Missing step column: {step_col!r}")

    levels_final = _validate_levels(tuple(levels))
    steps = sorted({int(s) for s in df[step_col].tolist()})
    if not steps:
        raise ValueError("No steps found to fit conformal intervals.")

    y = df[y_col].to_numpy(dtype=float, copy=False)
    yhat = df[yhat_col].to_numpy(dtype=float, copy=False)
    steps_arr = df[step_col].to_numpy()
    abs_err = np.abs(y - yhat)

    radius: dict[float, np.ndarray] = {}

    if not per_step:
        if abs_err.size == 0:
            raise ValueError("No residuals available to fit conformal intervals.")
        for lv in levels_final:
            q = np.quantile(abs_err, lv, method="higher")
            radius[lv] = np.full((len(steps),), float(q), dtype=float)
        return ConformalIntervals(
            levels=levels_final,
            per_step=False,
            steps=tuple(int(s) for s in steps),
            radius=radius,
        )

    for lv in levels_final:
        qs: list[float] = []
        for s in steps:
            mask = steps_arr == s
            err_s = abs_err[mask]
            if err_s.size == 0:
                raise ValueError(f"No residuals available for step={s} to fit conformal intervals.")
            qs.append(float(np.quantile(err_s, lv, method="higher")))
        radius[lv] = np.asarray(qs, dtype=float)

    return ConformalIntervals(
        levels=levels_final,
        per_step=True,
        steps=tuple(int(s) for s in steps),
        radius=radius,
    )


def apply_conformal_intervals(
    df: pd.DataFrame,
    conformal: ConformalIntervals,
    *,
    yhat_col: str = "yhat",
    step_col: str = "step",
) -> pd.DataFrame:
    """
    Return a copy of `df` with conformal interval columns appended.
    """
    if yhat_col not in df.columns:
        raise KeyError(f"Missing yhat column: {yhat_col!r}")
    if step_col not in df.columns:
        raise KeyError(f"Missing step column: {step_col!r}")

    yhat = df[yhat_col].to_numpy(dtype=float, copy=False)
    steps_arr = df[step_col].to_numpy()

    step_to_idx = {int(s): i for i, s in enumerate(conformal.steps)}
    if not step_to_idx:
        raise ValueError("ConformalIntervals has no steps.")

    out = df.copy()
    for lv in conformal.levels:
        rad = conformal.radius[lv]
        radii = np.empty_like(yhat, dtype=float)
        for i, s in enumerate(steps_arr):
            idx = step_to_idx.get(int(s))
            if idx is None:
                raise ValueError(f"Step {int(s)} not present in conformal calibrator.")
            radii[i] = float(rad[idx])

        p = int(round(lv * 100))
        out[f"yhat_lo_{p}"] = yhat - radii
        out[f"yhat_hi_{p}"] = yhat + radii

    return out


def summarize_conformal_predictions(
    df: pd.DataFrame,
    *,
    y_col: str = "y",
    yhat_col: str = "yhat",
    step_col: str = "step",
    levels: tuple[float, ...] = (0.8, 0.9),
    per_step: bool = True,
) -> dict[str, Any]:
    """
    Fit conformal intervals on a predictions table and summarize calibration/sharpness.
    """
    conf = fit_conformal_intervals(
        df,
        y_col=y_col,
        yhat_col=yhat_col,
        step_col=step_col,
        levels=levels,
        per_step=per_step,
    )
    out_df = apply_conformal_intervals(df, conf, yhat_col=yhat_col, step_col=step_col)
    y = out_df[y_col].to_numpy(dtype=float, copy=False)
    steps = np.asarray(out_df[step_col]) if step_col in out_df.columns else None

    out: dict[str, Any] = {
        "conformal_levels": [float(v) for v in conf.levels],
        "conformal_per_step": bool(conf.per_step),
    }

    uniq_steps = [int(s) for s in conf.steps]
    for lv in conf.levels:
        pct = int(round(lv * 100))
        lo = out_df[f"yhat_lo_{pct}"].to_numpy(dtype=float, copy=False)
        hi = out_df[f"yhat_hi_{pct}"].to_numpy(dtype=float, copy=False)
        alpha = 1.0 - float(lv)

        cov = interval_coverage(y, lo, hi)
        width = mean_interval_width(lo, hi)
        score = interval_score(y, lo, hi, alpha=alpha)
        wink = winkler_score(y, lo, hi, alpha=alpha)
        radius = conf.radius[lv].astype(float)

        out[f"radius_{pct}_by_step"] = radius.tolist()
        out[f"coverage_{pct}"] = float(cov)
        out[f"mean_width_{pct}"] = float(width)
        out[f"sharpness_{pct}"] = float(width)
        out[f"interval_score_{pct}"] = float(score)
        out[f"winkler_score_{pct}"] = float(wink)
        out[f"calibration_gap_{pct}"] = float(cov - float(lv))

        if steps is not None:
            cov_by_step: list[float] = []
            width_by_step: list[float] = []
            score_by_step: list[float] = []
            wink_by_step: list[float] = []
            for s in uniq_steps:
                mask = steps == s
                cov_s = interval_coverage(y[mask], lo[mask], hi[mask])
                width_s = mean_interval_width(lo[mask], hi[mask])
                score_s = interval_score(y[mask], lo[mask], hi[mask], alpha=alpha)
                wink_s = winkler_score(y[mask], lo[mask], hi[mask], alpha=alpha)
                cov_by_step.append(float(cov_s))
                width_by_step.append(float(width_s))
                score_by_step.append(float(score_s))
                wink_by_step.append(float(wink_s))

            out[f"coverage_{pct}_by_step"] = cov_by_step
            out[f"mean_width_{pct}_by_step"] = width_by_step
            out[f"sharpness_{pct}_by_step"] = list(width_by_step)
            out[f"interval_score_{pct}_by_step"] = score_by_step
            out[f"winkler_score_{pct}_by_step"] = wink_by_step
            out[f"calibration_gap_{pct}_by_step"] = [float(v - float(lv)) for v in cov_by_step]

    return out
