from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "__version__",
    "align_long_df",
    "BaseForecaster",
    "BaseGlobalForecaster",
    "bootstrap_intervals",
    "build_hierarchy_spec",
    "check_hierarchical_consistency",
    "clip_long_df_outliers",
    "eval_hierarchical_forecast_df",
    "eval_model",
    "eval_model_long_df",
    "eval_multivariate_model_df",
    "enrich_long_df_calendar",
    "fit_long_df_scaler",
    "forecast_model",
    "forecast_model_long_df",
    "infer_series_frequency",
    "inverse_transform_long_df_with_scaler",
    "load_forecaster",
    "load_forecaster_artifact",
    "make_local_xreg_forecast_bundle",
    "make_panel_sequence_blocks",
    "make_panel_sequence_tensors",
    "make_panel_window_arrays",
    "make_panel_window_frame",
    "make_panel_window_predict_arrays",
    "make_panel_window_predict_frame",
    "make_supervised_arrays",
    "make_supervised_frame",
    "make_supervised_predict_arrays",
    "make_supervised_predict_frame",
    "make_forecaster",
    "make_forecaster_object",
    "make_global_forecaster",
    "make_global_forecaster_object",
    "make_multivariate_forecaster",
    "prepare_long_df",
    "reconcile_hierarchical_forecasts",
    "save_forecaster",
    "split_supervised_frame",
    "split_supervised_arrays",
    "split_panel_window_arrays",
    "split_panel_window_frame",
    "split_panel_sequence_blocks",
    "split_panel_sequence_tensors",
    "split_long_df",
    "to_long",
    "transform_long_df_with_scaler",
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
        "align_long_df",
        "build_hierarchy_spec",
        "clip_long_df_outliers",
        "enrich_long_df_calendar",
        "fit_long_df_scaler",
        "infer_series_frequency",
        "inverse_transform_long_df_with_scaler",
        "make_local_xreg_forecast_bundle",
        "make_panel_sequence_blocks",
        "make_panel_sequence_tensors",
        "make_panel_window_arrays",
        "make_panel_window_frame",
        "make_panel_window_predict_arrays",
        "make_panel_window_predict_frame",
        "make_supervised_arrays",
        "make_supervised_frame",
        "make_supervised_predict_arrays",
        "make_supervised_predict_frame",
        "prepare_long_df",
        "split_supervised_frame",
        "split_supervised_arrays",
        "split_panel_window_arrays",
        "split_panel_window_frame",
        "split_panel_sequence_blocks",
        "split_panel_sequence_tensors",
        "split_long_df",
        "to_long",
        "transform_long_df_with_scaler",
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
