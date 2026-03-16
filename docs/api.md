# Python API Reference

This page documents the stable root-package entry points exported by `foresight.__all__`. Import these names directly from `foresight`.

```python
from foresight import (
    eval_model,
    forecast_model,
    load_forecaster_artifact,
    make_forecaster_object,
    save_forecaster,
)
```

## Core classes

| symbol | source | purpose |
| --- | --- | --- |
| `BaseForecaster` | `foresight.base` | Stateful local forecaster base class with fit/predict helpers. |
| `BaseGlobalForecaster` | `foresight.base` | Stateful panel/global forecaster base class for long-format data. |

## Forecasting

| symbol | source | purpose |
| --- | --- | --- |
| `forecast_model` | `foresight.forecast` | Run a one-off forecast for a single series and return a forecast dataframe. |
| `forecast_model_long_df` | `foresight.forecast` | Run a one-off forecast for long-format panel/global inputs, optionally with a separate future_df for known future covariates. |
| `make_forecaster` | `foresight.models.registry` | Create a stateless local forecasting callable from the registry. |
| `make_forecaster_object` | `foresight.models.registry` | Create a stateful local forecaster object with fit/predict/save support. |
| `make_global_forecaster` | `foresight.models.registry` | Create a stateless global/panel forecasting callable from the registry. |
| `make_global_forecaster_object` | `foresight.models.registry` | Create a stateful global forecaster object for panel workflows. |
| `make_multivariate_forecaster` | `foresight.models.registry` | Create a multivariate forecaster callable for wide matrix forecasting. |

## Evaluation

| symbol | source | purpose |
| --- | --- | --- |
| `eval_model` | `foresight.eval_forecast` | Walk-forward evaluation for a single univariate series or packaged dataset. |
| `eval_model_long_df` | `foresight.eval_forecast` | Walk-forward evaluation for long-format panel/global forecasting data. |
| `eval_multivariate_model_df` | `foresight.eval_forecast` | Evaluate multivariate forecasters on wide data frames. |

## Artifacts

| symbol | source | purpose |
| --- | --- | --- |
| `load_forecaster` | `foresight.serialization` | Load a persisted forecaster object from disk. |
| `load_forecaster_artifact` | `foresight.serialization` | Inspect the structured artifact payload before reconstructing an object. |
| `save_forecaster` | `foresight.serialization` | Persist a fitted forecaster and its schema/version metadata to disk. |

## Data preparation

| symbol | source | purpose |
| --- | --- | --- |
| `align_long_df` | `foresight.data` | Regularize per-series timestamps to a target frequency, with optional resampling aggregation. |
| `clip_long_df_outliers` | `foresight.data` | Clip per-series numeric outliers in long-format data without dropping rows. |
| `enrich_long_df_calendar` | `foresight.data` | Append deterministic calendar and cyclical time features onto long-format panel data. |
| `fit_long_df_scaler` | `foresight.data` | Fit reversible per-series or global scaling statistics for long-format numeric columns. |
| `infer_series_frequency` | `foresight.data` | Infer a sensible pandas-compatible series frequency from timestamps. |
| `inverse_transform_long_df_with_scaler` | `foresight.data` | Reverse fitted long-format scaling statistics to restore original numeric units. |
| `make_supervised_frame` | `foresight.data` | Build sklearn-style supervised training tables from long or wide time-series inputs. |
| `prepare_long_df` | `foresight.data` | Normalize and validate long-format panel data before forecasting/evaluation, with separate missing-value policies for target, historic covariates, and future covariates. |
| `split_long_df` | `foresight.data` | Chronologically split each long-format series into train, validation, and test partitions. |
| `to_long` | `foresight.data` | Convert wide or column-mapped inputs into ForeSight long format with role-aware historic_x_cols / future_x_cols support. |
| `transform_long_df_with_scaler` | `foresight.data` | Apply fitted scaling statistics to long-format numeric columns for training or evaluation workflows. |
| `validate_long_df` | `foresight.data` | Check that long-format inputs satisfy required schema and null rules. |

## Hierarchical forecasting

| symbol | source | purpose |
| --- | --- | --- |
| `build_hierarchy_spec` | `foresight.data` | Build a hierarchy specification from raw identifier columns. |
| `check_hierarchical_consistency` | `foresight.hierarchical` | Validate whether hierarchical forecasts reconcile cleanly. |
| `eval_hierarchical_forecast_df` | `foresight.eval_forecast` | Score reconciled hierarchical forecasts against held-out history, including bottom-up exogenous aggregation when requested. |
| `reconcile_hierarchical_forecasts` | `foresight.hierarchical` | Reconcile hierarchical forecasts with top-down or bottom-up methods, with optional bottom-up exogenous aggregation via exog_agg. |

## Intervals and tuning

| symbol | source | purpose |
| --- | --- | --- |
| `bootstrap_intervals` | `foresight.intervals` | Construct bootstrap forecast intervals from historical residual behavior. |
| `tune_model` | `foresight.tuning` | Grid-search a local forecasting model against backtest metrics. |
| `tune_model_long_df` | `foresight.tuning` | Grid-search a panel/global model on long-format data. |

## Package metadata

| symbol | source | purpose |
| --- | --- | --- |
| `__version__` | `foresight.__init__` | Installed ForeSight package version. |

## Notable data contracts

- `to_long(...)` accepts `historic_x_cols`, `future_x_cols`, and legacy `x_cols` (aliasing future covariates).
- `prepare_long_df(...)` supports separate `historic_x_missing` / `future_x_missing` policies after role-aware conversion.
- `forecast_model_long_df(...)` accepts `future_df=...` so known-future covariates can arrive in a separate dataframe from observed history.
- Lag-based regression models accept either contiguous `lags=n` or explicit `target_lags=(1, 7, 14)`; the sklearn `*-step-lag-global` family also supports `historic_x_lags` / `future_x_lags` when `x_cols` are supplied.
- `reconcile_hierarchical_forecasts(...)` supports `exog_agg={"promo": "sum", "temp": "mean"}` for bottom-up exogenous aggregation.

## Root package export list

- `__version__`
- `align_long_df`
- `BaseForecaster`
- `BaseGlobalForecaster`
- `bootstrap_intervals`
- `build_hierarchy_spec`
- `check_hierarchical_consistency`
- `clip_long_df_outliers`
- `eval_hierarchical_forecast_df`
- `eval_model`
- `eval_model_long_df`
- `eval_multivariate_model_df`
- `enrich_long_df_calendar`
- `fit_long_df_scaler`
- `forecast_model`
- `forecast_model_long_df`
- `infer_series_frequency`
- `inverse_transform_long_df_with_scaler`
- `load_forecaster`
- `load_forecaster_artifact`
- `make_supervised_frame`
- `make_forecaster`
- `make_forecaster_object`
- `make_global_forecaster`
- `make_global_forecaster_object`
- `make_multivariate_forecaster`
- `prepare_long_df`
- `reconcile_hierarchical_forecasts`
- `save_forecaster`
- `split_long_df`
- `to_long`
- `transform_long_df_with_scaler`
- `tune_model`
- `tune_model_long_df`
- `validate_long_df`
