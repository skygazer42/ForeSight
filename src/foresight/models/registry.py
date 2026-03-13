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
from .statsmodels_wrap import (
    arima_forecast,
    auto_arima_forecast,
    autoreg_forecast,
    ets_forecast,
    fourier_arima_forecast,
    fourier_auto_arima_forecast,
    fourier_autoreg_forecast,
    fourier_ets_forecast,
    fourier_sarimax_forecast,
    fourier_uc_forecast,
    mstl_arima_forecast,
    mstl_auto_arima_forecast,
    mstl_autoreg_forecast,
    mstl_ets_forecast,
    mstl_sarimax_forecast,
    mstl_uc_forecast,
    sarimax_forecast,
    stl_arima_forecast,
    stl_auto_arima_forecast,
    stl_autoreg_forecast,
    stl_ets_forecast,
    stl_sarimax_forecast,
    stl_uc_forecast,
    tbats_lite_auto_arima_forecast,
    tbats_lite_autoreg_forecast,
    tbats_lite_ets_forecast,
    tbats_lite_forecast,
    tbats_lite_sarimax_forecast,
    tbats_lite_uc_forecast,
    unobserved_components_forecast,
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
