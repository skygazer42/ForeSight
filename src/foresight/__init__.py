from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "__version__",
    "BaseForecaster",
    "BaseGlobalForecaster",
    "bootstrap_intervals",
    "build_hierarchy_spec",
    "check_hierarchical_consistency",
    "eval_hierarchical_forecast_df",
    "eval_model",
    "eval_model_long_df",
    "eval_multivariate_model_df",
    "forecast_model",
    "forecast_model_long_df",
    "infer_series_frequency",
    "load_forecaster",
    "load_forecaster_artifact",
    "make_forecaster",
    "make_forecaster_object",
    "make_global_forecaster",
    "make_global_forecaster_object",
    "make_multivariate_forecaster",
    "prepare_long_df",
    "reconcile_hierarchical_forecasts",
    "save_forecaster",
    "to_long",
    "tune_model",
    "tune_model_long_df",
    "validate_long_df",
]

__version__ = "0.2.9"


def __getattr__(name: str) -> Any:
    if name in {"BaseForecaster", "BaseGlobalForecaster"}:
        module = import_module(".base", __name__)
        return getattr(module, name)
    if name in {
        "build_hierarchy_spec",
        "infer_series_frequency",
        "prepare_long_df",
        "to_long",
        "validate_long_df",
    }:
        module = import_module(".data", __name__)
        return getattr(module, name)
    if name in {"bootstrap_intervals"}:
        module = import_module(".intervals", __name__)
        return getattr(module, name)
    if name in {
        "eval_hierarchical_forecast_df",
        "eval_model",
        "eval_model_long_df",
        "eval_multivariate_model_df",
    }:
        module = import_module(".eval_forecast", __name__)
        return getattr(module, name)
    if name in {"forecast_model", "forecast_model_long_df"}:
        module = import_module(".forecast", __name__)
        return getattr(module, name)
    if name in {
        "make_forecaster",
        "make_forecaster_object",
        "make_global_forecaster",
        "make_global_forecaster_object",
        "make_multivariate_forecaster",
    }:
        module = import_module(".models.registry", __name__)
        return getattr(module, name)
    if name in {"load_forecaster", "load_forecaster_artifact", "save_forecaster"}:
        module = import_module(".serialization", __name__)
        return getattr(module, name)
    if name in {"check_hierarchical_consistency", "reconcile_hierarchical_forecasts"}:
        module = import_module(".hierarchical", __name__)
        return getattr(module, name)
    if name in {"tune_model", "tune_model_long_df"}:
        module = import_module(".tuning", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
