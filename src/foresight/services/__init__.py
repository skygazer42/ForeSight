from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "detect_anomalies",
    "detect_anomalies_long_df",
    "eval_hierarchical_forecast_df",
    "eval_model",
    "eval_model_long_df",
    "eval_multivariate_model_df",
    "forecast_long_df",
    "forecast_model",
    "forecast_model_long_df",
]


def __getattr__(name: str) -> Any:
    if name in {"detect_anomalies", "detect_anomalies_long_df"}:
        module = import_module(".detection", __name__)
        return getattr(module, name)
    if name in {
        "eval_hierarchical_forecast_df",
        "eval_model",
        "eval_model_long_df",
        "eval_multivariate_model_df",
    }:
        module = import_module(".evaluation", __name__)
        return getattr(module, name)
    if name in {"forecast_long_df", "forecast_model", "forecast_model_long_df"}:
        module = import_module(".forecasting", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
