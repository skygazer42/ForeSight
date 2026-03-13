from __future__ import annotations

from ..base import (
    BaseForecaster,
    BaseGlobalForecaster,
    RegistryForecaster,
    RegistryGlobalForecaster,
)
from . import resolution as _resolution
from .resolution import get_model_spec, list_models
from .runtime import (
    make_forecaster,
    make_forecaster_object,
    make_global_forecaster,
    make_global_forecaster_object,
    make_multivariate_forecaster,
)
from .specs import (
    ForecasterFn,
    GlobalForecasterFn,
    LocalForecasterFn,
    ModelFactory,
    ModelSpec,
    MultivariateForecasterFn,
)

_REGISTRY = _resolution._REGISTRY

__all__ = [
    "BaseForecaster",
    "BaseGlobalForecaster",
    "ForecasterFn",
    "GlobalForecasterFn",
    "LocalForecasterFn",
    "ModelFactory",
    "ModelSpec",
    "MultivariateForecasterFn",
    "RegistryForecaster",
    "RegistryGlobalForecaster",
    "get_model_spec",
    "list_models",
    "make_forecaster",
    "make_forecaster_object",
    "make_global_forecaster",
    "make_global_forecaster_object",
    "make_multivariate_forecaster",
]
