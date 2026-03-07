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
| `forecast_model_long_df` | `foresight.forecast` | Run a one-off forecast for long-format panel/global inputs. |
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
| `infer_series_frequency` | `foresight.data` | Infer a sensible pandas-compatible series frequency from timestamps. |
| `prepare_long_df` | `foresight.data` | Normalize and validate long-format panel data before forecasting/evaluation. |
| `to_long` | `foresight.data` | Convert wide or column-mapped inputs into ForeSight long format. |
| `validate_long_df` | `foresight.data` | Check that long-format inputs satisfy required schema and null rules. |

## Hierarchical forecasting

| symbol | source | purpose |
| --- | --- | --- |
| `build_hierarchy_spec` | `foresight.data` | Build a hierarchy specification from raw identifier columns. |
| `check_hierarchical_consistency` | `foresight.hierarchical` | Validate whether hierarchical forecasts reconcile cleanly. |
| `eval_hierarchical_forecast_df` | `foresight.eval_forecast` | Score reconciled hierarchical forecasts against held-out history. |
| `reconcile_hierarchical_forecasts` | `foresight.hierarchical` | Reconcile hierarchical forecasts with top-down or bottom-up methods. |

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

## Root package export list

- `__version__`
- `BaseForecaster`
- `BaseGlobalForecaster`
- `bootstrap_intervals`
- `build_hierarchy_spec`
- `check_hierarchical_consistency`
- `eval_hierarchical_forecast_df`
- `eval_model`
- `eval_model_long_df`
- `eval_multivariate_model_df`
- `forecast_model`
- `forecast_model_long_df`
- `infer_series_frequency`
- `load_forecaster`
- `load_forecaster_artifact`
- `make_forecaster`
- `make_forecaster_object`
- `make_global_forecaster`
- `make_global_forecaster_object`
- `make_multivariate_forecaster`
- `prepare_long_df`
- `reconcile_hierarchical_forecasts`
- `save_forecaster`
- `to_long`
- `tune_model`
- `tune_model_long_df`
- `validate_long_df`
