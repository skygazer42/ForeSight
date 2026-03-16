# Panel Window Workflows Design

## Goal

Extend ForeSight's training-oriented data workflow layer with panel window builders that sit between cleaned long-format data and model fitting:

- `make_panel_window_frame`
- `make_panel_window_arrays`

This wave should make it straightforward to build repeatable global-model training datasets such as:

`prepare -> align -> clip -> enrich -> split -> scale -> make_panel_window_* -> train`

## Why This Wave

The package already has useful low-level and mid-level workflow steps:

- canonical long-format preparation and validation
- deterministic alignment and clipping
- calendar enrichment
- train/valid/test partitioning
- reversible scaling
- supervised table generation

What is still missing is a workflow layer that explicitly builds panel training windows with the same lag semantics already used by the global regression helpers. Right now, users can reach model-ready tables through `make_supervised_frame()`, but there is no public workflow helper focused on:

- panel-style sliding windows across multiple `unique_id` series
- direct export to dense `X` / `y` arrays for estimators
- stable metadata about window origin, forecast step, and feature layout

Adding this layer closes the gap between dataframe-centric preprocessing and array-oriented model training without forcing users into the internal model factories.

## Scope

This wave adds two helpers in `src/foresight/data/workflows.py`:

### `make_panel_window_frame`

Build a long-form training dataset where each output row represents one supervised target step from one panel window.

### `make_panel_window_arrays`

Build the same dataset as `make_panel_window_frame`, then convert it into dense numeric arrays plus metadata that can be consumed directly by sklearn-style estimators or custom training loops.

Both helpers should accept long-format input only and should reuse the package's existing lag-role semantics:

- `lags`
- `target_lags`
- `seasonal_lags`
- `historic_x_lags`
- `future_x_lags`
- `x_cols`

This wave does not replace `make_supervised_frame()`, does not introduce a new DSL, and does not change forecast-service behavior.

## Architecture

The implementation should stay in the existing data workflow layer:

- core logic in `src/foresight/data/workflows.py`
- exports in `src/foresight/data/__init__.py`
- root exports in `src/foresight/__init__.py`

The two public helpers should share one internal window-building path. The frame helper becomes the canonical public representation, and the arrays helper is only a conversion layer on top of that frame. This keeps the behavior inspectable and avoids maintaining two independent implementations with subtle schema drift.

The design intentionally mirrors the current global step-lag training semantics in `src/foresight/models/global_regression.py`:

- `target_lags` default to `lags`
- `future_x_lags` default to `(0,)` when `x_cols` are provided
- horizon is represented explicitly through a `step` column
- time-based features should be aligned to the target timestamp, not the cutoff timestamp

That alignment lets users move between the data workflows and global model helpers without learning two different feature conventions.

## API Design

Proposed signatures:

```python
def make_panel_window_frame(
    long_df: Any,
    *,
    horizon: int = 1,
    lags: Any = 24,
    target_lags: Any = (),
    seasonal_lags: Any = (),
    historic_x_lags: Any = (),
    future_x_lags: Any = (),
    x_cols: tuple[str, ...] = (),
    add_time_features: bool = False,
) -> pd.DataFrame:
    ...


def make_panel_window_arrays(
    long_df: Any,
    *,
    horizon: int = 1,
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

`make_panel_window_frame()` should return a stable training table with leading metadata columns:

- `unique_id`
- `cutoff_ds`
- `target_ds`
- `step`
- `y`

Feature columns should then be appended in deterministic order:

- `y_lag_{k}` from normalized `target_lags`
- `y_seasonal_lag_{k}` from normalized `seasonal_lags`
- `historic_x__{col}_lag_{k}` from `historic_x_lags`, indexed from the cutoff point
- `future_x__{col}_lag_{k}` from `future_x_lags`, indexed from the target point and allowing `0`
- optional time-feature columns derived from `target_ds`

`make_panel_window_arrays()` should call the frame helper and return:

- `X`: 2D numeric feature matrix
- `y`: 1D numeric target vector
- `feature_names`: ordered tuple of feature columns used in `X`
- `index`: DataFrame containing `unique_id`, `cutoff_ds`, `target_ds`, and `step`
- `metadata`: dict with normalized configuration and shape information

## Metadata and Semantics

The arrays `metadata` dict should make downstream training code self-describing. At minimum it should include:

- `horizon`
- normalized `target_lags`
- normalized `seasonal_lags`
- normalized `historic_x_lags`
- normalized `future_x_lags`
- normalized `x_cols`
- `add_time_features`
- `n_series`
- `n_windows`
- `n_rows`
- `n_features`

`n_windows` refers to the number of distinct `(unique_id, cutoff_ds)` window origins. `n_rows` refers to the flattened row count after expanding each window over horizon steps.

The semantic mapping should stay consistent with the model layer:

- if `target_lags` is empty, resolve it from `lags`
- if `x_cols` is empty, both historic and future exogenous blocks are omitted
- if `x_cols` is non-empty and `future_x_lags` is empty, default to `(0,)`
- `historic_x_lags` are relative to the cutoff timestamp
- `future_x_lags` are relative to each target timestamp

This keeps the workflow intuitive for direct model training while still exposing a fully inspectable dataframe.

## Error Handling

This wave should fail fast on malformed or ambiguous input:

- `TypeError` when `long_df` is not a pandas DataFrame
- `KeyError` when required columns `unique_id`, `ds`, or `y` are missing
- `KeyError` when requested `x_cols` are not present
- `ValueError` for invalid `horizon` or lag specifications
- `ValueError` when `x_cols` includes reserved column names
- `ValueError` when any series has duplicate timestamps, with guidance to run `align_long_df()` first
- `ValueError` when no series has enough history to build any window
- `ValueError` when generated numeric features or targets contain non-finite values

No silent deduplication, leakage-prone fallback behavior, or automatic imputation should be introduced here.

## Testing Strategy

Coverage should stay concentrated in:

- `tests/test_data_workflows.py`
- `tests/test_root_import.py`
- `tests/test_docs_rnn_generated.py` indirectly through regenerated API metadata

The tests should prove:

- frame schema and row counts for multi-series, multi-step inputs
- deterministic feature naming and ordering
- correct mapping of `future_x_lags=(0, 1, ...)`
- `seasonal_lags` are emitted as separate feature columns
- `make_panel_window_arrays()` stays consistent with the frame builder
- duplicate timestamps and short-history inputs fail clearly
- root exports and generated docs stay in sync

## Expected Outcome

After this wave, ForeSight will expose a compact, inspectable workflow layer for building panel training datasets from long-format time series. Users will be able to move from cleaned panel data to either:

- a model-ready dataframe with explicit metadata columns, or
- dense `X` / `y` arrays with feature names and configuration metadata

without re-implementing the step-lag semantics that already exist inside the package's global model helpers.
