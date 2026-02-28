from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .backtesting import walk_forward
from .datasets.loaders import load_dataset
from .metrics import mae, mape, rmse, smape
from .models.naive import naive_last, seasonal_naive


def _to_1d_float_series(values: Any) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {y.shape}")
    return y


def eval_naive_last(
    *,
    dataset: str,
    y_col: str,
    horizon: int,
    step: int,
    min_train_size: int,
    max_windows: int | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    df = load_dataset(dataset, data_dir=data_dir)
    if y_col not in df.columns:
        raise KeyError(f"Column {y_col!r} not found in dataset {dataset!r}. Columns: {list(df.columns)}")

    y = _to_1d_float_series(df[y_col].dropna().to_numpy())
    res = walk_forward(
        y,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        max_windows=max_windows,
        forecaster=naive_last,
    )

    y_true = res.y_true.reshape(-1)
    y_pred = res.y_pred.reshape(-1)

    mae_by_step = [mae(res.y_true[:, i], res.y_pred[:, i]) for i in range(res.horizon)]
    rmse_by_step = [rmse(res.y_true[:, i], res.y_pred[:, i]) for i in range(res.horizon)]
    mape_by_step = [mape(res.y_true[:, i], res.y_pred[:, i]) for i in range(res.horizon)]
    smape_by_step = [smape(res.y_true[:, i], res.y_pred[:, i]) for i in range(res.horizon)]

    return {
        "model": "naive-last",
        "dataset": dataset,
        "y_col": y_col,
        "horizon": int(horizon),
        "step": int(step),
        "min_train_size": int(min_train_size),
        "n_obs": int(y.size),
        "n_windows": int(res.y_true.shape[0]),
        "n_points": int(res.y_true.size),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mae_by_step": mae_by_step,
        "rmse_by_step": rmse_by_step,
        "mape_by_step": mape_by_step,
        "smape_by_step": smape_by_step,
    }


def eval_seasonal_naive(
    *,
    dataset: str,
    y_col: str,
    horizon: int,
    step: int,
    min_train_size: int,
    season_length: int,
    max_windows: int | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    df = load_dataset(dataset, data_dir=data_dir)
    if y_col not in df.columns:
        raise KeyError(f"Column {y_col!r} not found in dataset {dataset!r}. Columns: {list(df.columns)}")

    y = _to_1d_float_series(df[y_col].dropna().to_numpy())

    def _forecaster(train: Any, h: int) -> np.ndarray:
        return seasonal_naive(train, h, season_length=season_length)

    res = walk_forward(
        y,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        max_windows=max_windows,
        forecaster=_forecaster,
    )

    y_true = res.y_true.reshape(-1)
    y_pred = res.y_pred.reshape(-1)

    mae_by_step = [mae(res.y_true[:, i], res.y_pred[:, i]) for i in range(res.horizon)]
    rmse_by_step = [rmse(res.y_true[:, i], res.y_pred[:, i]) for i in range(res.horizon)]
    mape_by_step = [mape(res.y_true[:, i], res.y_pred[:, i]) for i in range(res.horizon)]
    smape_by_step = [smape(res.y_true[:, i], res.y_pred[:, i]) for i in range(res.horizon)]

    return {
        "model": "seasonal-naive",
        "dataset": dataset,
        "y_col": y_col,
        "horizon": int(horizon),
        "step": int(step),
        "min_train_size": int(min_train_size),
        "season_length": int(season_length),
        "n_obs": int(y.size),
        "n_windows": int(res.y_true.shape[0]),
        "n_points": int(res.y_true.size),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mae_by_step": mae_by_step,
        "rmse_by_step": rmse_by_step,
        "mape_by_step": mape_by_step,
        "smape_by_step": smape_by_step,
    }
