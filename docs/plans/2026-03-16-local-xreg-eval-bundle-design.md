# Local XReg Eval Bundle Design

## Goal

Extend ForeSight's public data workflow layer with a walk-forward evaluation helper for local models that require known future covariates:

- `make_local_xreg_eval_bundle`

This wave should expose the exact rolling-origin window inputs that the current local `x_cols` evaluation path already builds internally.

## Why This Wave

The local xreg forecast side now has a public workflow helper:

- `make_local_xreg_forecast_bundle()`

That helper exposes the one-shot forecast-time arrays that local future-covariate models consume. But the evaluation side still lives entirely inside `eval_model_long_df()`:

- rolling-origin windows are generated internally
- per-window `train_y`, `train_exog`, `future_exog`, and held-out targets are not inspectable
- users who want to reproduce exactly the same evaluation windows still need to rebuild split logic by hand

That leaves the local xreg workflow family incomplete. Forecast-time data is public, but evaluation-time data is still hidden inside the service layer.

## Alternatives Considered

### Option A: Expose nested per-series groups, each with a `windows` list

This mirrors the forecast bundle's per-series grouping, but it adds one extra nesting layer around the real execution unit. In evaluation, the model is called per rolling-origin window, not per series aggregate.

### Option B: Expose a flat `windows` list

This is the recommended option. It matches the actual execution unit in `eval_model_long_df()`, makes downstream inspection simple, and keeps the schema shallow.

### Option C: Add a generic local backtesting bundle for all local models

That could be useful later, but it broadens scope immediately. The local xreg path already has distinct semantics because it carries `train_exog` and `future_exog`, so it is better to expose that current contract directly first.

## Scope

This wave adds one helper in `src/foresight/data/workflows.py`:

### `make_local_xreg_eval_bundle`

Build a walk-forward local xreg evaluation bundle from a fully observed long-format dataframe and the standard rolling-origin parameters:

- `horizon`
- `step`
- `min_train_size`
- optional `max_train_size`
- optional `max_windows`

This wave does not add model scoring, metrics, conformal intervals, or historic covariate support. It only exposes the evaluation windows.

## Architecture

The implementation should stay in the existing workflow layer:

- core logic in `src/foresight/data/workflows.py`
- exports in `src/foresight/data/__init__.py`
- root exports in `src/foresight/__init__.py`

The helper should reuse the same conceptual rules already enforced by `eval_model_long_df()`:

- canonical long-format validation
- full observed history only; no missing `y`
- finite `x_cols` values for requested covariates
- rolling-origin split generation through the existing split semantics
- no support for `historic_x_cols`

The architectural rule is:

`fully observed long_df + rolling-origin parameters -> flat local xreg eval window bundle`

## API Design

Proposed signature:

```python
def make_local_xreg_eval_bundle(
    long_df: Any,
    *,
    horizon: int,
    step: int,
    min_train_size: int,
    x_cols: Any = (),
    future_x_cols: Any = (),
    historic_x_cols: Any = (),
    max_windows: int | None = None,
    max_train_size: int | None = None,
    dtype: Any = np.float64,
) -> dict[str, Any]:
    ...
```

Covariate-role semantics:

- `x_cols` stays as a compatibility alias for future covariates
- `future_x_cols` is accepted and merged with `x_cols`
- `historic_x_cols` is rejected explicitly because the current local xreg evaluation path does not support it

## Bundle Schema

The helper should return:

```python
{
    "windows": [
        {
            "unique_id": ...,
            "window": ...,
            "cutoff_ds": ...,
            "target_start_ds": ...,
            "target_end_ds": ...,
            "x_cols": (...,),
            "train_y": np.ndarray,
            "actual_y": np.ndarray,
            "train_exog": np.ndarray,
            "future_exog": np.ndarray,
            "train_index": pd.DataFrame,
            "test_index": pd.DataFrame,
        },
        ...
    ],
    "metadata": {
        "horizon": ...,
        "step": ...,
        "min_train_size": ...,
        "max_train_size": ...,
        "max_windows": ...,
        "x_cols": (...,),
        "n_series": ...,
        "n_series_skipped": ...,
        "n_windows": ...,
        "series_ids": (...,),
    },
}
```

Per-window semantics:

- `train_y` shape: `(train_length,)`
- `actual_y` shape: `(horizon,)`
- `train_exog` shape: `(train_length, n_x)`
- `future_exog` shape: `(horizon, n_x)`
- `train_index` columns: `unique_id`, `ds`
- `test_index` columns: `unique_id`, `ds`, `step`
- `window` is 1-based within each `unique_id`

## Missing-Value Semantics

This workflow should mirror `eval_model_long_df()` exactly:

- missing `y` values are not allowed
- missing `x_cols` values are not allowed
- future covariates are taken from the held-out test horizon inside the fully observed series

This is different from the forecast bundle on purpose. Forecast-time data allows future rows with missing `y`; evaluation windows do not.

## Validation and Error Handling

This wave should fail fast on malformed inputs:

- `TypeError` when `long_df` is not a pandas DataFrame
- `KeyError` when required columns are missing
- `ValueError` when `long_df` contains duplicate `unique_id/ds` rows
- `ValueError` when `historic_x_cols` is requested
- `ValueError` when future covariates are not supplied
- `ValueError` when `y` contains missing values
- `ValueError` when requested `x_cols` contain missing values

Series that do not have enough observations for `min_train_size + horizon` should be skipped rather than raising, because that is what the current evaluation service already does.

## Testing Strategy

Coverage should stay concentrated in:

- `tests/test_data_workflows.py`
- `tests/test_root_import.py`
- `tests/test_docs_rnn_generated.py` indirectly through regenerated API metadata

The tests should prove:

- rolling-origin windows carry the same train/test boundaries as `rolling_origin_splits()`
- `max_train_size` changes the training window start as expected
- `future_x_cols` aliasing matches `x_cols`
- `historic_x_cols` is rejected clearly
- root exports and generated docs stay in sync

## Expected Outcome

After this wave, ForeSight's local xreg public workflow layer will cover both of the main data preparation surfaces:

- one-shot forecast bundles
- walk-forward evaluation bundles

That gives users an inspectable public path for the same per-window arrays that `eval_model_long_df()` already consumes internally, without forcing them to reverse-engineer rolling-origin split behavior.
