from __future__ import annotations

from typing import Any

import numpy as np

from .backtesting import walk_forward
from .datasets.loaders import load_dataset
from .metrics import mae, mape, rmse, smape
from .models.naive import naive_last


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
) -> dict[str, Any]:
    df = load_dataset(dataset)
    if y_col not in df.columns:
        raise KeyError(f"Column {y_col!r} not found in dataset {dataset!r}. Columns: {list(df.columns)}")

    y = _to_1d_float_series(df[y_col].dropna().to_numpy())
    res = walk_forward(
        y,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        forecaster=naive_last,
    )

    y_true = res.y_true.reshape(-1)
    y_pred = res.y_pred.reshape(-1)

    return {
        "model": "naive-last",
        "dataset": dataset,
        "y_col": y_col,
        "horizon": int(horizon),
        "step": int(step),
        "min_train_size": int(min_train_size),
        "n_windows": int(res.y_true.shape[0]),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }

