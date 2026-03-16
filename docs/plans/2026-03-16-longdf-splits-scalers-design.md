# LongDF Splits and Scalers Design

## Goal

Extend ForeSight's data workflow layer with a compact training-oriented batch that closes the gap between raw panel preparation and model fitting:

- `split_long_df`
- `fit_long_df_scaler`
- `transform_long_df_with_scaler`
- `inverse_transform_long_df_with_scaler`

This wave should make it straightforward to build repeatable training pipelines such as:

`prepare -> align -> clip -> split -> fit scaler -> transform -> make_supervised_frame -> train -> inverse transform`

## Why This Wave

The package already covers several useful preprocessing steps:

- canonical long-format conversion and validation
- frequency alignment
- outlier clipping
- deterministic calendar enrichment
- supervised feature table generation

What is still missing is the workflow layer that typically sits between preprocessing and training:

- deterministic train/validation/test splitting for long-format panel data
- reproducible scaling that can be fit once on training data and applied consistently to holdout data and predictions

Without these helpers, users have to rebuild common time-series training patterns outside the package. That weakens the value of the existing data workflow layer, especially now that `make_supervised_frame()` already exists as a model-ready downstream target.

## Scope

This wave adds four helpers in `src/foresight/data/workflows.py`:

### `split_long_df`

Split a long-format panel chronologically into `train`, optional `valid`, and optional `test` partitions per `unique_id`.

Supported behavior:

- explicit `valid_size` / `test_size`
- fraction-based `valid_frac` / `test_frac`
- optional temporal `gap`
- minimum training length guard via `min_train_size`
- deterministic sorting by `unique_id`, then `ds`

Return type:

- `dict[str, pd.DataFrame]` with keys `train`, `valid`, and `test`

### `fit_long_df_scaler`

Fit scaling statistics for selected numeric columns.

Supported methods:

- `standard`
- `minmax`
- `maxabs`

Supported scopes:

- `per_series`
- `global`

Return type:

- a stable stats DataFrame describing fitted scaling parameters per `(scope, unique_id, column)`

### `transform_long_df_with_scaler`

Apply a fitted scaler stats table to another long-format DataFrame using the same columns and scope logic.

### `inverse_transform_long_df_with_scaler`

Reverse the scaling for selected numeric columns, so post-training predictions or evaluation frames can be returned to the original target scale.

## Architecture

The implementation stays entirely inside the existing data layer:

- logic in `src/foresight/data/workflows.py`
- exports in `src/foresight/data/__init__.py`
- optional root-package exports in `src/foresight/__init__.py`

No model runtime or service orchestration changes are needed. These helpers are pure data transformations. That keeps the wave low-risk and makes the new code useful from both Python and future CLI or service entry points.

The scaler design intentionally uses a plain DataFrame for fitted statistics instead of a custom estimator object. That keeps serialization and inspection simple, aligns with the package's dataframe-centric workflow style, and avoids introducing a new artifact abstraction just for preprocessing.

## Split Design

`split_long_df` should operate per series, not globally over the combined frame. Each `unique_id` gets split independently according to chronological order.

Rules:

- if both size- and fraction-based arguments are provided for the same partition, raise `ValueError`
- fractions must be in `[0, 1)`
- sizes must be non-negative integers
- at least one of `valid_size`, `test_size`, `valid_frac`, or `test_frac` must be positive
- `gap` removes rows between train and holdout partitions for each series
- every series must retain at least `min_train_size` rows in `train`

Partition order per series:

`train -> gap -> valid -> gap -> test`

If `valid` is omitted, the split becomes:

`train -> gap -> test`

This matches common time-series holdout practice and avoids leakage from rows adjacent to the evaluation horizon.

## Scaler Design

The scaler stats DataFrame should include enough information for both forward and inverse transforms without guesswork.

Proposed columns:

- `scope`
- `unique_id`
- `column`
- `method`
- `center`
- `scale`
- `data_min`
- `data_max`

Behavior:

- stats are fit on finite non-null observations only
- nulls remain null during forward and inverse transforms
- constant-series edge cases remain stable:
  - `standard`: use `scale=1.0`
  - `maxabs`: use `scale=1.0` when max abs is zero
  - `minmax`: map constant series to zero by using `scale=1.0` after subtracting `data_min`

For `global` scope, the stats row uses `unique_id="__global__"` so the table is still joinable and explicit.

## Error Handling

This wave should fail fast on configuration mistakes:

- bad schema or missing columns: `KeyError`
- invalid types: `TypeError`
- invalid method/scope/fraction/size arguments: `ValueError`
- non-numeric requested scaling columns: `ValueError`
- attempting to transform with missing stats rows for requested columns or series: `ValueError`

No silent fallback from `per_series` to `global` should be added. If the provided scaler table does not match the requested transform, the function should raise.

## Testing Strategy

Coverage should stay concentrated in:

- `tests/test_data_workflows.py`
- `tests/test_root_import.py`
- `tests/test_docs_rnn_generated.py` indirectly through updated API metadata

The tests should prove:

- split helpers preserve chronology and per-series boundaries
- gap and min-train constraints behave correctly
- scaler fit/transform/inverse cycles round-trip numeric data
- `per_series` and `global` scopes behave differently and correctly
- root exports and generated API docs stay in sync

## Expected Outcome

After this wave, ForeSight will support a cleaner end-to-end training workflow for time-series experiments: users can partition long-format data safely, fit reproducible scaling statistics, transform training and holdout sets consistently, and invert predictions back to the original scale without rebuilding these utilities externally.
