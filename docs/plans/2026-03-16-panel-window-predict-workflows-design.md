# Panel Window Predict Workflows Design

## Goal

Extend ForeSight's panel-window workflow family with public helpers that build prediction-time feature inputs for global step-lag style models:

- `make_panel_window_predict_frame`
- `make_panel_window_predict_arrays`

This wave should make it possible to inspect and export the exact future-step feature rows that global regression models consume at forecast time, without depending on private model-layer helpers.

## Why This Wave

The panel-window family is now strong on the training side:

- `make_panel_window_frame()` materializes training rows
- `make_panel_window_arrays()` exports dense training arrays
- `split_panel_window_frame()` and `split_panel_window_arrays()` handle post-materialization chronology

That still leaves one practical gap: prediction-time feature construction is only available implicitly inside `src/foresight/models/global_regression.py`.

Today, users who want to:

- inspect the features fed to a global step-lag model at a given cutoff
- cache prediction design matrices before model scoring
- debug future covariate leakage or missing-value issues
- feed custom external estimators with the same forecast-time features as the built-in models

still need to reconstruct the model-internal prediction logic themselves.

That is weak API design. The package already exposes the training-side window builder publicly, so the prediction-side mirror should also be public.

## Alternatives Considered

### Option A: Do nothing and leave prediction feature building inside the model layer

This preserves the status quo, but keeps an important workflow hidden and forces users to reverse-engineer private helpers.

### Option B: Return one packed row per series with horizon encoded inside the feature vector

This would be compact, but it would not match the current global step-lag model interface. The internal model code expands one row per forecast step and appends a `step` feature, so a packed representation would create unnecessary divergence.

### Option C: Return one row per future step, aligned with the training panel-window frame

This is the recommended option. It preserves symmetry with `make_panel_window_frame()`, matches the internal step-lag prediction path, and keeps feature auditing simple.

## Scope

This wave adds two helpers in `src/foresight/data/workflows.py`:

### `make_panel_window_predict_frame`

Build a dataframe of prediction-time feature rows, one row per `(unique_id, target_ds, step)` for the requested `cutoff` and `horizon`.

### `make_panel_window_predict_arrays`

Convert that dataframe into dense numeric arrays plus metadata for direct use in sklearn-style or custom prediction loops.

This wave does not add target values, split helpers, or model scoring logic. It only exposes the prediction-side feature workflow.

## Architecture

The implementation should stay in the existing workflow layer:

- core logic in `src/foresight/data/workflows.py`
- exports in `src/foresight/data/__init__.py`
- root exports in `src/foresight/__init__.py`

The architectural rule is:

`long_df -> panel-window training rows` and `long_df + cutoff -> panel-window prediction rows`

The prediction helper should mirror the training helper's feature ordering and naming conventions:

- target lag features from the history before cutoff
- seasonal lag features from the same history
- historic exogenous lag features relative to the prediction origin
- future exogenous lag features relative to each future target step
- optional time features aligned to `target_ds`

Prediction rows should expand by horizon in exactly the same way the global step-lag model internals do: one feature row per future step, with `step` carried as metadata rather than packed into a hidden structure.

## API Design

Proposed signatures:

```python
def make_panel_window_predict_frame(
    long_df: Any,
    *,
    cutoff: Any,
    horizon: int,
    lags: Any = 24,
    target_lags: Any = (),
    seasonal_lags: Any = (),
    historic_x_lags: Any = (),
    future_x_lags: Any = (),
    x_cols: tuple[str, ...] = (),
    add_time_features: bool = False,
) -> pd.DataFrame:
    ...


def make_panel_window_predict_arrays(
    long_df: Any,
    *,
    cutoff: Any,
    horizon: int,
    lags: Any = 24,
    target_lags: Any = (),
    seasonal_lags: Any = (),
    historic_x_lags: Any = (),
    future_x_lags: Any = (),
    x_cols: tuple[str, ...] = (),
    add_time_features: bool = False,
    dtype: Any = np.float64,
) -> dict[str, Any]:
    ...
```

## Output Semantics

`make_panel_window_predict_frame()` should return a dataframe with leading metadata columns:

- `unique_id`
- `cutoff_ds`
- `target_ds`
- `step`

Feature columns should follow the exact same deterministic order as `make_panel_window_frame()`, except that there is no `y` target column.

`make_panel_window_predict_arrays()` should return:

- `X`
- `feature_names`
- `index`
- `metadata`

Where `index` contains:

- `unique_id`
- `cutoff_ds`
- `target_ds`
- `step`

## Missing-Value Semantics

This workflow should explicitly support forecast-style future rows where `y` is missing after cutoff.

That means:

- historical `y` values needed by lag features must be finite through the cutoff history
- future `y` values after cutoff are allowed to be null because prediction features do not consume them
- required `x_cols` values at the indices referenced by historic or future exogenous lags must be finite

The helper should fail clearly when the necessary future rows or future covariate values are unavailable.

## Validation and Error Handling

This wave should fail fast on malformed inputs:

- `TypeError` when `long_df` is not a pandas DataFrame
- `KeyError` when required columns are missing
- `KeyError` when requested `x_cols` are not present
- `ValueError` when no eligible prediction rows can be built for the requested cutoff
- `ValueError` when a series has duplicate timestamps
- `ValueError` when required history `y` values are non-finite
- `ValueError` when required covariate values are non-finite

No hidden imputation or model-layer fallback behavior should be introduced here.

## Testing Strategy

Coverage should stay concentrated in:

- `tests/test_data_workflows.py`
- `tests/test_root_import.py`
- `tests/test_docs_rnn_generated.py` indirectly through regenerated API metadata

The tests should prove:

- predict-frame feature values match the training frame at the same cutoff when future targets are observed
- prediction arrays match the predict frame exactly
- future rows with missing `y` still work when required future covariates are present
- root exports and generated docs stay in sync

## Expected Outcome

After this wave, the panel-window workflow family becomes symmetric across training and prediction:

- build training rows and arrays
- split training rows and arrays
- build prediction rows and arrays for a cutoff

That gives users a stable, inspectable public path for both sides of the global step-lag data flow, instead of forcing them to rely on private model internals at forecast time.
