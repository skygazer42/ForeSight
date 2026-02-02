from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr


def mae(y_true: Any, y_pred: Any) -> float:
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: Any, y_pred: Any) -> float:
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: Any, y_pred: Any, *, eps: float = 1e-8) -> float:
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    denom = np.where(np.abs(yt) < eps, eps, np.abs(yt))
    return float(np.mean(np.abs((yt - yp) / denom)))


def smape(y_true: Any, y_pred: Any, *, eps: float = 1e-8) -> float:
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_pred)
    denom = np.abs(yt) + np.abs(yp)
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(2.0 * np.abs(yt - yp) / denom))

