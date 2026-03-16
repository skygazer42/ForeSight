# Balanced TS Workflows and Local Models Design

## Goal

Extend ForeSight with a balanced wave of:

- higher-level time-series data workflows that turn raw panel data into stable training inputs
- a small set of lightweight, trainable local sklearn lag models that fill gaps relative to the existing global ML catalog

The wave should reuse existing primitives wherever possible, avoid touching the forecast and evaluation service paths, and stay compatible with the package's current long-format and CLI conventions.

## Scope

This wave adds four data workflow helpers:

- `align_long_df()` for per-series frequency alignment and optional resampling
- `clip_long_df_outliers()` for per-series outlier clipping on numeric columns
- `enrich_long_df_calendar()` for deterministic calendar and cyclical time features derived from `ds`
- `make_supervised_frame()` for turning long or wide time series into supervised training tables

This wave also adds four local ML forecasters:

- `bayesian-ridge-lag`
- `ard-lag`
- `omp-lag`
- `passive-aggressive-lag`

Each model stays in the existing local lag-family style and uses scikit-learn as an optional dependency.

## Architecture

The new workflow layer lives above `to_long()` and `prepare_long_df()` rather than replacing them. Raw data still enters ForeSight through the existing formatting and validation helpers. Once a canonical long DataFrame exists, new workflow functions provide a higher-level path for:

1. regularizing panel timestamps
2. clipping unstable targets before fitting
3. enriching rows with reusable time features
4. building a supervised feature/target table for downstream model training

This keeps the new behavior inside `src/foresight/data/` and `src/foresight/features/`, where the package already expects deterministic preprocessing logic. The forecast and evaluation services do not need new orchestration paths for this wave.

The local model additions continue to flow through the existing model stack:

`catalog/ml.py -> runtime.py -> regression.py`

That preserves a single registration and construction path for all local ML lag models.

## Data Workflow Design

### `align_long_df()`

Purpose:

- align each `unique_id` series to an explicit or inferred frequency
- optionally aggregate observations when multiple rows land in the same resampled bucket
- return canonical long format with `unique_id / ds / y / ...`

Behavior:

- requires a valid long DataFrame
- accepts `freq`, `agg`, and optional explicit target/covariate columns
- fails fast on duplicate timestamps unless resampling is explicitly requested
- applies per-column aggregation only to numeric data, and raises on unsupported columns or strategies
- sorts by `unique_id`, then `ds`

This function is intentionally narrower than a full pipeline engine. It only normalizes time spacing and basic aggregation rules.

### `clip_long_df_outliers()`

Purpose:

- clip unstable spikes before training
- support robust, deterministic clipping without introducing a new dependency

Behavior:

- default column set is `("y",)`
- clips independently per `unique_id`
- supports `method="iqr"` and `method="zscore"`
- returns a DataFrame with the same rows and column order
- preserves non-target columns unchanged

The API is clipping-only for this wave. Row dropping, masking, and anomaly labeling are out of scope.

### `enrich_long_df_calendar()`

Purpose:

- expose reusable calendar features from the existing dependency-free time feature generator
- append stable numeric feature columns directly onto long-format data

Behavior:

- uses `features.time.build_time_features()`
- adds columns with a configurable prefix such as `cal_`
- fails on naming conflicts unless explicitly allowed later in a separate wave
- preserves row count and row order

This gives users an immediate bridge from prepared long tables to trainable feature tables without inventing a second feature DSL.

### `make_supervised_frame()`

Purpose:

- turn time series into sklearn-ready supervised data
- expose one stable output schema for local experimentation and workflow chaining

Supported inputs:

- long format with `unique_id / ds / y`
- wide format with `ds` plus one or more target columns

Output schema:

- metadata columns: `unique_id`, `ds`, `target_t`
- feature columns: prefixed `feat_*`
- target columns:
  - single-step mode: `y_target`
  - direct multi-step mode: `y_t+1`, `y_t+2`, ..., `y_t+h`

Feature generation reuses existing ForeSight primitives:

- lag windows from `features.lag`
- roll stats and diffs from `features.tabular`
- seasonal lag features from `features.lag`
- optional time features from `features.time`

The helper does not train models. It only produces stable supervised tables.

## Local Model Design

The new local models are lightweight sklearn forecasters that mirror the package's existing local lag model style:

- direct multi-horizon training
- lag-based feature generation
- optional lag-derived, seasonal, and Fourier features
- `MultiOutputRegressor` where the base estimator is not naturally multi-output

Planned models:

### `bayesian-ridge-lag`

- direct multi-horizon Bayesian ridge regression
- good default for small data with shrinkage and posterior-style regularization

### `ard-lag`

- direct multi-horizon ARD regression
- useful for sparse autoregressive feature selection without adding a new dependency family

### `omp-lag`

- direct multi-horizon orthogonal matching pursuit
- offers a sparse linear baseline distinct from Lasso and ElasticNet

### `passive-aggressive-lag`

- direct multi-horizon passive-aggressive regression
- adds an online-leaning large-margin option to the local lag family

All four require only the existing `ml` extra and should register under `catalog/ml.py` with parameter help aligned to the current ML catalog style.

## Error Handling

The wave follows existing ForeSight error semantics:

- invalid types raise `TypeError`
- missing columns raise `KeyError`
- invalid configuration values raise `ValueError`
- missing optional dependencies raise `ImportError`

Specific guardrails:

- `align_long_df()` rejects invalid frequencies, unsupported aggregation strategies, and ambiguous duplicate timestamps
- `clip_long_df_outliers()` rejects unknown methods, non-positive thresholds, and non-numeric clipping targets
- `enrich_long_df_calendar()` rejects column-name collisions
- `make_supervised_frame()` rejects invalid horizon/lag specs, insufficient history, and nonexistent covariate columns
- new local models reject invalid solver parameters before calling sklearn

No silent fallback behavior should be added in this wave beyond the package's current optional frequency inference behavior.

## Testing Strategy

Coverage is split into four layers:

1. unit tests for each new workflow helper
2. CLI tests for each new `foresight data` subcommand
3. model registry and optional-dependency tests for the four local models
4. one end-to-end smoke path that chains `prepare -> align -> clip -> calendar -> supervised`

The new tests should follow the style of:

- `tests/test_data_prep.py`
- `tests/test_features_tabular.py`
- `tests/test_cli_data.py`
- `tests/test_models_optional_deps_ml.py`

## Out of Scope

This wave does not:

- add a general pipeline object API
- change `forecast_model()` or `eval_model()` service flows
- add holiday calendars or external calendar dependencies
- add anomaly detection labels or row-removal workflows
- add new global, torch, xgboost, lightgbm, or catboost families
- add historic-covariate support to `forecast_model_long_df()` or `eval_model_long_df()`

## Expected Outcome

After this wave, ForeSight should support a tighter workflow from raw long-format data to trainable tables, while also exposing a few new lightweight local trainable models that make the local ML family feel more complete without expanding the dependency surface.
