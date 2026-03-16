# Local Count Lag Models Design

## Goal

Extend ForeSight with a compact second wave of trainable local time-series models by filling a clear parity gap in the sklearn lag family:

- `poisson-lag`
- `gamma-lag`
- `tweedie-lag`

These models should reuse the existing local lag-model stack, stay within the current optional `ml` dependency surface, and expose behavior that is materially different from the already available Gaussian-style local regressors.

## Why This Wave

The current catalog already contains:

- local Gaussian and robust sklearn lag models such as `ridge-lag`, `huber-lag`, and `quantile-lag`
- global panel count-family models such as `poisson-step-lag-global`, `gamma-step-lag-global`, and `tweedie-step-lag-global`
- XGBoost local count-family models such as `xgb-poisson-lag`, `xgb-gamma-lag`, and `xgb-tweedie-lag`

What is still missing is the lightweight local sklearn version of that count-family batch. That makes the local catalog feel uneven: users can reach for distribution-aware models globally or through XGBoost, but not through the simplest sklearn-only local path.

This wave fills that gap without introducing a new abstraction. It adds a small but high-value batch that is easy to train, easy to test, and useful for examples involving non-negative or strictly positive targets such as demand, clicks, traffic, and claims.

## Scope

This wave adds three local lag forecasters:

- `poisson-lag`: direct multi-horizon `PoissonRegressor` on lag features
- `gamma-lag`: direct multi-horizon `GammaRegressor` on lag features
- `tweedie-lag`: direct multi-horizon `TweedieRegressor` on lag features

All three should support the same local feature options already used by `huber-lag` and `quantile-lag`:

- lag-derived features via `roll_windows`, `roll_stats`, and `diff_lags`
- seasonal lag features via `seasonal_lags` and `seasonal_diff_lags`
- Fourier features via `fourier_periods` and `fourier_orders`

Out of scope for this wave:

- new service-layer orchestration
- new CLI commands
- new global models
- recursive local count-family variants
- interval estimation or probabilistic output schemas

## Architecture

The implementation should follow the existing local model registration path:

`catalog/ml.py -> runtime.py -> regression.py`

`regression.py` owns the actual sklearn training and prediction logic. The new functions should mirror the style of `huber_lag_direct_forecast()` and `quantile_lag_direct_forecast()`:

- validate `horizon`, `lags`, and estimator-specific parameters
- build lagged training matrices with `_make_lagged_xy_multi()`
- enrich features through `_augment_lag_matrix()` and `_augment_lag_feat_row()`
- wrap the base estimator in `MultiOutputRegressor`
- predict a direct multi-step horizon from the most recent lag row

`runtime.py` should expose one factory per model, coercing inputs to stable Python types before dispatching to the regression implementation.

`catalog/ml.py` should register the three model keys with descriptions, defaults, parameter help, `requires=("ml",)`, and local interface semantics.

## Model-Specific Rules

### `poisson-lag`

- Base estimator: `sklearn.linear_model.PoissonRegressor`
- Training targets must be non-negative
- Parameters: `lags`, `alpha`, `max_iter`, plus shared lag-feature options

### `gamma-lag`

- Base estimator: `sklearn.linear_model.GammaRegressor`
- Training targets must be strictly positive
- Parameters: `lags`, `alpha`, `max_iter`, plus shared lag-feature options

### `tweedie-lag`

- Base estimator: `sklearn.linear_model.TweedieRegressor`
- `power` must be `<= 0` or `>= 1`
- Target validation depends on `power`:
  - `power == 1` allows non-negative targets
  - `power > 1` requires strictly positive targets
- Parameters: `lags`, `power`, `alpha`, `max_iter`, plus shared lag-feature options

The Tweedie target checks should match the semantics already used by the global sklearn implementation so users do not get conflicting behavior between local and global APIs.

## Error Handling

The wave should stay consistent with current ForeSight conventions:

- missing sklearn dependency raises `ImportError`
- invalid lag or horizon raises `ValueError`
- invalid regularization or iteration parameters raises `ValueError`
- invalid target-domain assumptions raise `ValueError` with model-specific messages

No silent clipping or target shifting should be introduced. If data violates the estimator assumptions, the model should fail fast.

## Testing Strategy

Coverage should stay compact but specific:

1. registry and optional dependency tests in `tests/test_models_optional_deps_ml.py`
2. local feature smoke coverage in `tests/test_models_lag_derived_features.py`
3. API-level argument coercion coverage in `tests/test_forecaster_api.py`

The tests should prove:

- each model is registered as an optional `ml` model
- feature-rich local calls produce finite horizon outputs when sklearn is installed
- legacy string coercion through `make_forecaster()` works for `alpha`, `max_iter`, and `power`
- invalid data-domain assumptions fail with the correct error messages

## Expected Outcome

After this wave, ForeSight will have a more coherent local sklearn catalog: lightweight local distribution-aware models will cover non-negative and positive-target forecasting tasks without requiring XGBoost or a global panel workflow.
