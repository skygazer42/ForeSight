from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
