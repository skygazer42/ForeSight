from __future__ import annotations

from typing import Any

import numpy as np

from ..base import BaseForecaster, BaseGlobalForecaster
from ..models.registry import (
    get_model_spec as _get_model_spec,
    make_forecaster,
    make_forecaster_object,
    make_global_forecaster,
    make_global_forecaster_object,
    make_multivariate_forecaster,
)
from ..models.specs import ForecasterFn, GlobalForecasterFn, ModelSpec, MultivariateForecasterFn


def get_model_spec(model: str) -> ModelSpec:
    return _get_model_spec(str(model))


def make_local_forecaster_runner(
    model: str,
    model_params: dict[str, Any] | None = None,
) -> ForecasterFn:
    return make_forecaster(str(model), **dict(model_params or {}))


def make_local_forecaster_object_runner(
    model: str,
    model_params: dict[str, Any] | None = None,
) -> BaseForecaster:
    return make_forecaster_object(str(model), **dict(model_params or {}))


def make_global_forecaster_runner(
    model: str,
    model_params: dict[str, Any] | None = None,
) -> GlobalForecasterFn:
    return make_global_forecaster(str(model), **dict(model_params or {}))


def make_global_forecaster_object_runner(
    model: str,
    model_params: dict[str, Any] | None = None,
) -> BaseGlobalForecaster:
    return make_global_forecaster_object(str(model), **dict(model_params or {}))


def make_multivariate_forecaster_runner(
    model: str,
    model_params: dict[str, Any] | None = None,
) -> MultivariateForecasterFn:
    return make_multivariate_forecaster(str(model), **dict(model_params or {}))


def call_local_xreg_forecaster(
    *,
    model: str,
    train_y: np.ndarray,
    horizon: int,
    train_exog: np.ndarray,
    future_exog: np.ndarray,
    model_params: dict[str, Any] | None = None,
) -> np.ndarray:
    forecaster = make_local_forecaster_runner(str(model), model_params)
    try:
        out = forecaster(
            train_y,
            int(horizon),
            train_exog=train_exog,
            future_exog=future_exog,
        )
    except TypeError as e:
        raise ValueError(
            f"Model {model!r} advertises x_cols support but its local callable does not accept "
            "`train_exog` / `future_exog`."
        ) from e

    yhat = np.asarray(out, dtype=float)
    if yhat.shape != (int(horizon),):
        raise ValueError(f"forecaster must return shape ({int(horizon)},), got {yhat.shape}")
    return yhat


__all__ = [
    "call_local_xreg_forecaster",
    "get_model_spec",
    "make_global_forecaster_object_runner",
    "make_global_forecaster_runner",
    "make_local_forecaster_object_runner",
    "make_local_forecaster_runner",
    "make_multivariate_forecaster_runner",
]
