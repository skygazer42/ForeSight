from __future__ import annotations

from pathlib import Path
from typing import Any

from .services import evaluation as _evaluation

_require_long_df = _evaluation._require_long_df
_parse_levels = _evaluation._parse_levels
_normalize_x_cols = _evaluation._normalize_x_cols
_normalize_covariate_roles = _evaluation._normalize_covariate_roles
_require_x_cols_if_needed = _evaluation._require_x_cols_if_needed
_call_local_xreg_forecaster = _evaluation._call_local_xreg_forecaster
_require_multivariate_df = _evaluation._require_multivariate_df
_walk_forward_multivariate = _evaluation._walk_forward_multivariate


def eval_multivariate_model_df(
    *,
    model: str,
    df: Any,
    target_cols: Any,
    horizon: int,
    step: int,
    min_train_size: int,
    ds_col: str | None = "ds",
    model_params: dict[str, Any] | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
) -> dict[str, Any]:
    return _evaluation.eval_multivariate_model_df(
        model=model,
        df=df,
        target_cols=target_cols,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        ds_col=ds_col,
        model_params=model_params,
        max_windows=max_windows,
        max_train_size=max_train_size,
    )


def eval_hierarchical_forecast_df(
    *,
    forecast_df: Any,
    hierarchy: Any,
    method: str,
    history_df: Any = None,
    yhat_col: str = "yhat",
    exog_agg: Any = None,
) -> dict[str, Any]:
    return _evaluation.eval_hierarchical_forecast_df(
        forecast_df=forecast_df,
        hierarchy=hierarchy,
        method=method,
        history_df=history_df,
        yhat_col=yhat_col,
        exog_agg=exog_agg,
    )


def eval_model_long_df(
    *,
    model: str,
    long_df: Any,
    horizon: int,
    step: int,
    min_train_size: int,
    model_params: dict[str, Any] | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
    conformal_levels: Any = None,
    conformal_per_step: bool = True,
) -> dict[str, Any]:
    return _evaluation.eval_model_long_df(
        model=model,
        long_df=long_df,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        model_params=model_params,
        max_windows=max_windows,
        max_train_size=max_train_size,
        conformal_levels=conformal_levels,
        conformal_per_step=conformal_per_step,
    )


def eval_model(
    *,
    model: str,
    dataset: str,
    horizon: int,
    step: int,
    min_train_size: int,
    y_col: str | None = None,
    model_params: dict[str, Any] | None = None,
    data_dir: str | Path | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
    conformal_levels: Any = None,
    conformal_per_step: bool = True,
) -> dict[str, Any]:
    return _evaluation.eval_model(
        model=model,
        dataset=dataset,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        y_col=y_col,
        model_params=model_params,
        data_dir=data_dir,
        max_windows=max_windows,
        max_train_size=max_train_size,
        conformal_levels=conformal_levels,
        conformal_per_step=conformal_per_step,
    )


__all__ = [
    "_call_local_xreg_forecaster",
    "_normalize_covariate_roles",
    "_normalize_x_cols",
    "_parse_levels",
    "_require_long_df",
    "_require_multivariate_df",
    "_require_x_cols_if_needed",
    "_walk_forward_multivariate",
    "eval_hierarchical_forecast_df",
    "eval_model",
    "eval_model_long_df",
    "eval_multivariate_model_df",
]
