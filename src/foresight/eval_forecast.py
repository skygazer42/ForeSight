from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtesting import walk_forward
from .data.format import to_long
from .datasets.loaders import load_dataset
from .datasets.registry import get_dataset_spec
from .metrics import mae, mape, rmse, smape
from .models.registry import make_forecaster


def _require_long_df(long_df: Any) -> pd.DataFrame:
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df must be a pandas DataFrame")
    required = {"unique_id", "ds", "y"}
    missing = required.difference(long_df.columns)
    if missing:
        raise KeyError(f"long_df missing required columns: {sorted(missing)}")
    return long_df


def _parse_levels(levels: Any) -> tuple[float, ...]:
    if levels is None:
        return ()

    if isinstance(levels, list | tuple):
        items = list(levels)
    elif isinstance(levels, str):
        s = levels.strip()
        items = [] if not s else [p.strip() for p in s.split(",") if p.strip()]
    else:
        items = [levels]

    out: list[float] = []
    for it in items:
        f = float(it)
        if f >= 1.0:
            f = f / 100.0  # allow 80 -> 0.8
        if not (0.0 < f < 1.0):
            raise ValueError("conformal_levels must be in (0,1) or percentages like 80,90")
        out.append(f)
    return tuple(sorted(set(out)))


def eval_model_long_df(
    *,
    model: str,
    long_df: Any,
    horizon: int,
    step: int,
    min_train_size: int,
    model_params: dict[str, Any] | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
    conformal_levels: Any = None,
    conformal_per_step: bool = True,
) -> dict[str, Any]:
    """
    Generic evaluation for any registered model on a canonical long-format DataFrame.

    The input must have columns: unique_id, ds, y.
    """
    df = _require_long_df(long_df)
    if df.empty:
        raise ValueError("long_df is empty")

    df = df.sort_values(["unique_id", "ds"], kind="mergesort")
    forecaster = make_forecaster(str(model), **(model_params or {}))

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_true_by_step: list[list[np.ndarray]] = [[] for _ in range(int(horizon))]
    y_pred_by_step: list[list[np.ndarray]] = [[] for _ in range(int(horizon))]

    n_series = 0
    n_series_skipped = 0

    for _uid, g in df.groupby("unique_id", sort=False):
        n_series += 1
        y = g["y"].to_numpy(dtype=float, copy=False)
        try:
            res = walk_forward(
                y,
                horizon=int(horizon),
                step=int(step),
                min_train_size=int(min_train_size),
                max_train_size=max_train_size,
                max_windows=max_windows,
                forecaster=forecaster,
            )
        except ValueError:
            n_series_skipped += 1
            continue

        y_true_all.append(res.y_true.reshape(-1))
        y_pred_all.append(res.y_pred.reshape(-1))

        for i in range(res.horizon):
            y_true_by_step[i].append(res.y_true[:, i])
            y_pred_by_step[i].append(res.y_pred[:, i])

    if not y_true_all:
        raise ValueError("No series had enough data for the requested backtest parameters.")

    yt = np.concatenate(y_true_all, axis=0)
    yp = np.concatenate(y_pred_all, axis=0)

    mae_by_step = [
        mae(np.concatenate(t), np.concatenate(p))
        for t, p in zip(y_true_by_step, y_pred_by_step, strict=True)
    ]
    rmse_by_step = [
        rmse(np.concatenate(t), np.concatenate(p))
        for t, p in zip(y_true_by_step, y_pred_by_step, strict=True)
    ]
    mape_by_step = [
        mape(np.concatenate(t), np.concatenate(p))
        for t, p in zip(y_true_by_step, y_pred_by_step, strict=True)
    ]
    smape_by_step = [
        smape(np.concatenate(t), np.concatenate(p))
        for t, p in zip(y_true_by_step, y_pred_by_step, strict=True)
    ]

    out: dict[str, Any] = {
        "model": str(model),
        "horizon": int(horizon),
        "step": int(step),
        "min_train_size": int(min_train_size),
        "max_windows": None if max_windows is None else int(max_windows),
        "max_train_size": None if max_train_size is None else int(max_train_size),
        "n_series": int(n_series),
        "n_series_skipped": int(n_series_skipped),
        "n_points": int(yt.size),
        "mae": mae(yt, yp),
        "rmse": rmse(yt, yp),
        "mape": mape(yt, yp),
        "smape": smape(yt, yp),
        "mae_by_step": mae_by_step,
        "rmse_by_step": rmse_by_step,
        "mape_by_step": mape_by_step,
        "smape_by_step": smape_by_step,
    }

    levels = _parse_levels(conformal_levels)
    if levels:
        out["conformal_levels"] = list(levels)
        out["conformal_per_step"] = bool(conformal_per_step)

        if conformal_per_step:
            abs_err_by_step = [
                np.abs(np.concatenate(t) - np.concatenate(p))
                for t, p in zip(y_true_by_step, y_pred_by_step, strict=True)
            ]
        else:
            abs_err_pooled = np.abs(yt - yp)
            abs_err_by_step = [abs_err_pooled] * int(horizon)

        for lv in levels:
            pct = int(round(lv * 100))
            radius = np.array(
                [np.quantile(err, lv, method="higher") for err in abs_err_by_step], dtype=float
            )

            cov_by_step: list[float] = []
            for i in range(int(horizon)):
                yt_i = np.concatenate(y_true_by_step[i])
                yp_i = np.concatenate(y_pred_by_step[i])
                lo = yp_i - float(radius[i])
                hi = yp_i + float(radius[i])
                cov_by_step.append(float(np.mean((yt_i >= lo) & (yt_i <= hi))))

            out[f"radius_{pct}_by_step"] = radius.astype(float).tolist()
            out[f"coverage_{pct}_by_step"] = cov_by_step
            out[f"mean_width_{pct}_by_step"] = (2.0 * radius).astype(float).tolist()
            out[f"coverage_{pct}"] = float(np.mean(cov_by_step))
            out[f"mean_width_{pct}"] = float(np.mean(2.0 * radius))

    return out


def eval_model(
    *,
    model: str,
    dataset: str,
    horizon: int,
    step: int,
    min_train_size: int,
    y_col: str | None = None,
    model_params: dict[str, Any] | None = None,
    data_dir: str | Path | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
    conformal_levels: Any = None,
    conformal_per_step: bool = True,
) -> dict[str, Any]:
    """
    Generic evaluation for any registered model on a dataset spec (supports panel datasets).

    Data is converted to a canonical long format (unique_id, ds, y), then evaluated
    per-series using walk-forward backtesting and aggregated across series.
    """
    spec = get_dataset_spec(str(dataset))
    y_col_final = str(y_col) if (y_col is not None and str(y_col).strip()) else spec.default_y

    df = load_dataset(str(dataset), data_dir=data_dir)
    long_df = to_long(
        df,
        time_col=spec.time_col,
        y_col=y_col_final,
        id_cols=tuple(spec.group_cols),
        dropna=True,
    )
    if long_df.empty:
        raise ValueError("Loaded 0 rows after to_long(dropna=True). Check dataset and y_col.")

    payload = eval_model_long_df(
        model=str(model),
        long_df=long_df,
        horizon=int(horizon),
        step=int(step),
        min_train_size=int(min_train_size),
        model_params=model_params,
        max_windows=max_windows,
        max_train_size=max_train_size,
        conformal_levels=conformal_levels,
        conformal_per_step=conformal_per_step,
    )
    payload["dataset"] = str(dataset)
    payload["y_col"] = y_col_final
    return payload
