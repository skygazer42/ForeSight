from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

Forecaster = Callable[[Any, int], np.ndarray]


@dataclass(frozen=True)
class WalkForwardResult:
    y_true: np.ndarray  # shape: (n_windows, horizon)
    y_pred: np.ndarray  # shape: (n_windows, horizon)
    train_ends: np.ndarray  # shape: (n_windows,) indices into original series
    horizon: int
    step: int
    min_train_size: int


def walk_forward(
    y: Any,
    *,
    horizon: int,
    step: int = 1,
    min_train_size: int,
    max_windows: int | None = None,
    forecaster: Forecaster,
) -> WalkForwardResult:
    """
    Simple rolling-origin evaluation for 1D series.

    For each window ending at `train_end` (exclusive), fit/forecast using `forecaster(train, horizon)`.
    """
    series = np.asarray(y, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"walk_forward expects 1D series, got shape {series.shape}")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if step <= 0:
        raise ValueError("step must be >= 1")
    if min_train_size <= 0:
        raise ValueError("min_train_size must be >= 1")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be >= 1")

    max_train_end = series.size - horizon
    if max_train_end < min_train_size:
        raise ValueError(
            f"Not enough data: need at least min_train_size+horizon={min_train_size + horizon}, got {series.size}"
        )

    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    train_ends: list[int] = []

    for train_end in range(min_train_size, max_train_end + 1, step):
        train = series[:train_end]
        true = series[train_end : train_end + horizon]
        pred = np.asarray(forecaster(train, horizon), dtype=float)
        if pred.shape != (horizon,):
            raise ValueError(f"forecaster must return shape ({horizon},), got {pred.shape}")

        y_true_list.append(true)
        y_pred_list.append(pred)
        train_ends.append(train_end)
        if max_windows is not None and len(y_true_list) >= max_windows:
            break

    y_true_arr = np.stack(y_true_list, axis=0)
    y_pred_arr = np.stack(y_pred_list, axis=0)
    train_ends_arr = np.asarray(train_ends, dtype=int)

    return WalkForwardResult(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        train_ends=train_ends_arr,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
    )
