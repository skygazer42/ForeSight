# Local XReg Forecast Bundle Design

## Goal

Extend ForeSight's public data workflow layer with a forecast-time helper for local models that require known future covariates:

- `make_local_xreg_forecast_bundle`

This wave should expose the exact per-series arrays that the current local `x_cols` forecast path already builds internally:

- `train_y`
- `train_exog`
- `future_exog`

## Why This Wave

ForeSight already supports local models with future covariates in the forecast and evaluation services. The relevant service path is stable and capability-driven:

- `forecast_model_long_df(..., model_params={"x_cols": (...)})`
- `eval_model_long_df(..., model_params={"x_cols": (...)})`

But that data flow is still trapped inside the service layer.

Today, users who want to:

- inspect the exact arrays sent to a local xreg forecaster
- run custom local estimators outside the registry using the same covariate preparation rules
- debug observed versus future covariate alignment around a cutoff
- reuse the package's validated `future_df` merge semantics without invoking a model

still need to reconstruct the service logic by hand.

That is a public API gap. The package already exposes training and prediction workflows for supervised and panel-window families. Local xreg forecast inputs should also have a public, inspectable workflow surface.

## Alternatives Considered

### Option A: Add a `frame + arrays` pair

This would mimic the newer workflow families on paper, but it is the wrong shape here. Local xreg forecast inputs are inherently ragged by series because `train_y` and `train_exog` lengths depend on observed history. There is no natural single global feature matrix to expose.

### Option B: Add one forecast bundle helper

This is the recommended option. It mirrors the real service-layer model call shape exactly, keeps the API small, and avoids inventing a fake tabular representation that users would immediately need to regroup.

### Option C: Add only evaluation-window helpers

That would be useful later, but it skips the simpler, more foundational public gap: exposing one forecast-time local xreg bundle for a prepared history and future horizon.

## Scope

This wave adds one helper in `src/foresight/data/workflows.py`:

### `make_local_xreg_forecast_bundle`

Build a forecast-time bundle of per-series local xreg arrays and index metadata from:

- a merged `long_df` with trailing future rows where `y` is missing, or
- an observed-history `long_df` plus a separate `future_df`

This wave does not add evaluation split helpers, model scoring, or historic covariate support. It only exposes the current local future-covariate forecast input contract.

## Architecture

The implementation should stay in the existing workflow layer:

- core logic in `src/foresight/data/workflows.py`
- exports in `src/foresight/data/__init__.py`
- root exports in `src/foresight/__init__.py`

The helper should reuse the same conceptual rules already enforced by `forecast_model_long_df()`:

- canonical long-format validation
- optional `future_df` normalization and merge
- per-series chronological split into observed history and future horizon rows
- finite observed and future covariates for requested `x_cols`
- no support for `historic_x_cols`

The architectural rule is:

`observed long_df (+ optional future_df) -> per-series local xreg forecast bundle`

## API Design

Proposed signature:

```python
def make_local_xreg_forecast_bundle(
    long_df: Any,
    *,
    horizon: int,
    x_cols: Any = (),
    future_x_cols: Any = (),
    historic_x_cols: Any = (),
    future_df: Any | None = None,
    dtype: Any = np.float64,
) -> dict[str, Any]:
    ...
```

Covariate-role semantics:

- `x_cols` stays as a compatibility alias for future covariates
- `future_x_cols` is accepted and merged with `x_cols`
- `historic_x_cols` is rejected explicitly because the current local xreg path does not support it

This keeps the helper aligned with the broader package covariate vocabulary without pretending local historic covariates already work.

## Bundle Schema

The helper should return:

```python
{
    "groups": [
        {
            "unique_id": ...,
            "cutoff_ds": ...,
            "x_cols": (...,),
            "train_y": np.ndarray,
            "train_exog": np.ndarray,
            "future_exog": np.ndarray,
            "train_index": pd.DataFrame,
            "future_index": pd.DataFrame,
        },
        ...
    ],
    "metadata": {
        "horizon": ...,
        "x_cols": (...,),
        "n_series": ...,
        "uses_future_df": ...,
        "series_ids": (...,),
    },
}
```

Per-group semantics:

- `train_y` shape: `(n_observed,)`
- `train_exog` shape: `(n_observed, n_x)`
- `future_exog` shape: `(horizon, n_x)`
- `train_index` columns: `unique_id`, `ds`
- `future_index` columns: `unique_id`, `ds`, `step`

The bundle should preserve per-series order and expose enough metadata to map arrays back to timestamps cleanly.

## Missing-Value Semantics

This workflow should explicitly support forecast-style future rows where `y` is missing after the observed history.

That means:

- each series must have at least one observed `y`
- once `y` first becomes missing within a series, all later `y` values must remain missing
- if `future_df` is supplied, it must not contain observed `y` values
- requested future covariates must be finite in both observed history and future horizon rows

This wave should not silently impute covariates or allow interleaved observed and missing targets.

## Validation and Error Handling

This wave should fail fast on malformed inputs:

- `TypeError` when `long_df` or `future_df` is not a pandas DataFrame
- `KeyError` when required columns are missing
- `ValueError` when `long_df` contains duplicate `unique_id/ds` rows
- `ValueError` when `future_df` overlaps `long_df`
- `ValueError` when `future_df` contains observed `y`
- `ValueError` when `historic_x_cols` is requested
- `ValueError` when a series has no observed history
- `ValueError` when missing `y` values appear before the end of observed history
- `ValueError` when future rows are fewer than `horizon`
- `ValueError` when required observed or future covariate values are missing

## Testing Strategy

Coverage should stay concentrated in:

- `tests/test_data_workflows.py`
- `tests/test_root_import.py`
- `tests/test_docs_rnn_generated.py` indirectly through regenerated API metadata

The tests should prove:

- merged `long_df` with trailing missing `y` produces the expected per-series arrays and index metadata
- `future_df` input produces the same bundle as the merged `long_df` path
- `x_cols` and `future_x_cols` compatibility is preserved
- `historic_x_cols` is rejected clearly
- root exports and generated docs stay in sync

## Expected Outcome

After this wave, ForeSight will expose a stable public path for the local future-covariate forecast data flow, instead of leaving it entirely inside the service layer.

Users will be able to build and inspect the exact per-series arrays that local xreg models consume, which makes the package stronger for debugging, custom model integration, and reproducible workflow chaining.
