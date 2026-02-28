from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import mae, mape, rmse, smape


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
