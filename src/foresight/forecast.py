from __future__ import annotations

from typing import Any

import pandas as pd

from .services import forecasting as _forecasting

_require_long_df = _forecasting._require_long_df
_require_future_df = _forecasting._require_future_df
_merge_history_and_future_df = _forecasting._merge_history_and_future_df
_require_observed_history_only = _forecasting._require_observed_history_only
_normalize_model_params = _forecasting._normalize_model_params
_normalize_x_cols = _forecasting._normalize_x_cols
_normalize_covariate_roles = _forecasting._normalize_covariate_roles
_require_x_cols_if_needed = _forecasting._require_x_cols_if_needed
_parse_interval_levels = _forecasting._parse_interval_levels
_interval_level_label = _forecasting._interval_level_label
_resolve_interval_min_train_size = _forecasting._resolve_interval_min_train_size
_interval_column_names = _forecasting._interval_column_names
_parse_quantiles = _forecasting._parse_quantiles
_required_quantiles_for_interval_levels = _forecasting._required_quantiles_for_interval_levels
_merge_quantiles_for_interval_levels = _forecasting._merge_quantiles_for_interval_levels
_add_interval_columns_from_quantile_predictions = (
    _forecasting._add_interval_columns_from_quantile_predictions
)
_local_interval_columns = _forecasting._local_interval_columns
_local_xreg_interval_payload = _forecasting._local_xreg_interval_payload
_call_local_xreg_forecaster = _forecasting._call_local_xreg_forecaster
_as_datetime_index = _forecasting._as_datetime_index
_infer_future_ds = _forecasting._infer_future_ds
_future_frame_for_group = _forecasting._future_frame_for_group
_prepare_local_xreg_forecast_group = _forecasting._prepare_local_xreg_forecast_group
_prepare_global_forecast_input = _forecasting._prepare_global_forecast_input
_finalize_forecast_frame = _forecasting._finalize_forecast_frame


def forecast_model_long_df(
    *,
    model: str,
    long_df: Any,
    future_df: Any | None = None,
    horizon: int,
    model_params: dict[str, Any] | None = None,
    interval_levels: Any = None,
    interval_min_train_size: int | None = None,
    interval_samples: int = 1000,
    interval_seed: int | None = None,
) -> pd.DataFrame:
    return _forecasting.forecast_model_long_df(
        model=model,
        long_df=long_df,
        future_df=future_df,
        horizon=horizon,
        model_params=model_params,
        interval_levels=interval_levels,
        interval_min_train_size=interval_min_train_size,
        interval_samples=interval_samples,
        interval_seed=interval_seed,
    )


def forecast_model(
    *,
    model: str,
    y: Any,
    horizon: int,
    ds: Any | None = None,
    unique_id: str = "series=0",
    model_params: dict[str, Any] | None = None,
    interval_levels: Any = None,
    interval_min_train_size: int | None = None,
    interval_samples: int = 1000,
    interval_seed: int | None = None,
) -> pd.DataFrame:
    return _forecasting.forecast_model(
        model=model,
        y=y,
        horizon=horizon,
        ds=ds,
        unique_id=unique_id,
        model_params=model_params,
        interval_levels=interval_levels,
        interval_min_train_size=interval_min_train_size,
        interval_samples=interval_samples,
        interval_seed=interval_seed,
    )


__all__ = [
    "forecast_model",
    "forecast_model_long_df",
]
