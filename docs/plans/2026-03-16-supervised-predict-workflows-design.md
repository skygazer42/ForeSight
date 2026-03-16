# Supervised Predict Workflows Design

## Goal

Extend ForeSight's dataframe-first supervised workflow family with public prediction-time helpers:

- `make_supervised_predict_frame`
- `make_supervised_predict_arrays`

This wave should make it possible to materialize the exact direct-forecast feature rows that local lag-style models consume at a chosen cutoff, without rebuilding private feature logic outside the package.

## Why This Wave

The supervised workflow family is now strong on the training side:

- `make_supervised_frame()` builds inspectable training rows
- `make_supervised_arrays()` exposes dense training arrays
- `split_supervised_frame()` and `split_supervised_arrays()` handle chronological holdouts after materialization

That still leaves one obvious gap: users can train direct lag models from public workflow outputs, but they cannot build the matching prediction-time feature row from the same workflow layer.

Today, users who want to:

- inspect the exact feature row fed to a trained direct lag regressor at forecast time
- export a stable prediction design matrix for custom sklearn-style models
- debug lag, seasonal, Fourier, or exogenous feature leakage around a cutoff
- keep training and prediction feature construction in one public workflow family

still need to reconstruct the prediction-side row themselves.

That is weak API symmetry. If the package exposes the training-side supervised row builder publicly, it should also expose the prediction-side mirror.

## Alternatives Considered

### Option A: Do nothing and leave prediction row construction to users

This preserves the status quo, but forces every downstream training loop to rebuild feature ordering and cutoff logic manually.

### Option B: Return one row per future step

This looks superficially similar to the panel-window prediction helpers, but it is the wrong abstraction for the supervised family. Multi-step supervised training rows are direct examples, not recursive step-by-step examples. A per-step prediction frame would imply future lag values that are not available without model predictions.

### Option C: Return one direct prediction row per eligible series and cutoff

This is the recommended option. It matches the supervised training representation, avoids recursive leakage, and keeps feature parity with `make_supervised_frame(..., horizon=h)`.

## Scope

This wave adds two helpers in `src/foresight/data/workflows.py`:

### `make_supervised_predict_frame`

Build one prediction-time supervised feature row per eligible `unique_id` for a requested `cutoff` and `horizon`.

### `make_supervised_predict_arrays`

Convert that dataframe into dense numeric arrays plus metadata for direct sklearn-style prediction loops.

This wave does not add recursive forecasting orchestration, model scoring, or separate split helpers for prediction bundles.

## Architecture

The implementation should stay in the existing workflow layer:

- core logic in `src/foresight/data/workflows.py`
- exports in `src/foresight/data/__init__.py`
- root exports in `src/foresight/__init__.py`

The architectural rule is:

`long_df + cutoff + horizon -> one direct prediction row per series`

The prediction helper should mirror the training helper's feature naming and ordering conventions:

- lag features from observed history through `cutoff`
- lag-derived rolling and diff features from the same lag row
- seasonal lag and seasonal diff features aligned to the next target start
- optional Fourier features aligned to the next target start
- optional time features aligned to the next target start
- optional `x_cols` taken from the next target start row, exactly as `make_supervised_frame()` does for training examples

This means prediction rows stay compatible with models trained from the existing supervised frame outputs.

## API Design

Proposed signatures:

```python
def make_supervised_predict_frame(
    long_df: Any,
    *,
    cutoff: Any,
    horizon: int = 1,
    lags: Any = 5,
    x_cols: tuple[str, ...] = (),
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    add_time_features: bool = False,
) -> pd.DataFrame:
    ...


def make_supervised_predict_arrays(
    long_df: Any,
    *,
    cutoff: Any,
    horizon: int = 1,
    lags: Any = 5,
    x_cols: tuple[str, ...] = (),
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    add_time_features: bool = False,
    dtype: Any = np.float64,
) -> dict[str, Any]:
    ...
```

## Output Semantics

`make_supervised_predict_frame()` should return a dataframe with leading metadata columns:

- `unique_id`
- `cutoff_ds`
- `target_start_ds`
- `target_end_ds`

Feature columns should follow the exact same deterministic order as the feature columns produced by `make_supervised_frame(..., horizon=h)` for the matching training row.

`make_supervised_predict_arrays()` should return:

- `X`
- `feature_names`
- `index`
- `metadata`

Where `index` contains:

- `unique_id`
- `cutoff_ds`
- `target_start_ds`
- `target_end_ds`

## Missing-Value Semantics

This workflow should explicitly support forecast-style future rows where `y` is missing after the cutoff.

That means:

- historical `y` values needed by lag-based features must be finite through the cutoff history
- future `y` values after the cutoff are allowed to be null because prediction features do not consume them
- requested `x_cols` must be finite on the first future target row only, because that is the same exogenous row the training workflow uses for each direct example
- there must be at least `horizon` future rows after the cutoff so the helper can emit stable target-start and target-end metadata

No hidden imputation or recursive rollout behavior should be introduced here.

## Validation and Error Handling

This wave should fail fast on malformed inputs:

- `TypeError` when `long_df` is not a pandas DataFrame
- `KeyError` when required columns are missing
- `KeyError` when requested `x_cols` are not present
- `ValueError` when no eligible prediction rows can be built for the requested cutoff
- `ValueError` when a series has duplicate timestamps
- `ValueError` when required historical `y` values are non-finite
- `ValueError` when required first-target `x_cols` values are non-finite

## Testing Strategy

Coverage should stay concentrated in:

- `tests/test_data_workflows.py`
- `tests/test_root_import.py`
- `tests/test_docs_rnn_generated.py` indirectly through regenerated API metadata

The tests should prove:

- predict-frame feature values match the training frame at the same cutoff when future targets are observed
- prediction arrays match the predict frame exactly
- future rows with missing `y` still work when required first-target covariates are present
- root exports and generated docs stay in sync

## Expected Outcome

After this wave, the supervised workflow family becomes symmetric across training and prediction:

- build training rows and arrays
- split training rows and arrays
- build direct prediction rows and arrays for a cutoff

That gives users one stable public path for both sides of the direct supervised data flow, instead of forcing them to rebuild prediction-time feature logic by hand.
