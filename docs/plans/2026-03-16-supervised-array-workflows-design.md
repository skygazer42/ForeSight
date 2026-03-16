# Supervised Array Workflows Design

## Goal

Extend ForeSight's dataframe-first supervised workflow layer with dense array helpers that sit directly on top of `make_supervised_frame()`:

- `make_supervised_arrays`
- `split_supervised_arrays`

This wave should make it straightforward to build repeatable sklearn-style and custom tabular training bundles such as:

`prepare -> enrich -> make_supervised_frame -> make_supervised_arrays -> split_supervised_arrays -> train`

## Why This Wave

ForeSight already exposes a strong tabular-supervised entry point through `make_supervised_frame()`. That function is valuable because it keeps feature engineering inspectable and works across:

- long-format panel data
- wide multivariate data converted internally to long format
- single-step and multi-step direct targets
- lag, rolling, seasonal, Fourier, and time-derived features

But there is still one obvious ergonomic gap: users who want to train estimators usually need dense `X` / `y` arrays plus stable metadata, not only a dataframe.

Today they still need to hand-roll:

- feature column selection
- target column selection
- row index preservation
- conversion to 1D vs 2D targets
- chronological train/valid/test splits over already-built supervised examples

That is repetitive, easy to get subtly wrong, and inconsistent with the richer workflow support already available for panel windows and sequence bundles.

This wave turns the supervised dataframe layer into a fuller training workflow without changing the existing frame API.

## Scope

This wave adds two helpers in `src/foresight/data/workflows.py`:

### `make_supervised_arrays`

Build a dense supervised training bundle by first constructing `make_supervised_frame(...)` and then deterministically exposing feature arrays, target arrays, row index, and metadata.

### `split_supervised_arrays`

Chronologically split a supervised array bundle into train, validation, and test partitions independently within each `unique_id`.

This wave does not replace `make_supervised_frame()`. The dataframe remains the canonical inspectable public representation, while the arrays helper becomes an adapter for direct model training.

## Architecture

The implementation should stay in the existing workflow layer:

- core logic in `src/foresight/data/workflows.py`
- exports in `src/foresight/data/__init__.py`
- root exports in `src/foresight/__init__.py`

The critical architectural rule is:

`data -> make_supervised_frame -> make_supervised_arrays`

`make_supervised_arrays()` should not rebuild the feature engineering logic independently. Instead it should call `make_supervised_frame()` and then perform a deterministic conversion:

- metadata columns become `index`
- feature columns become `X`
- target columns become `y`

Likewise, `split_supervised_arrays()` should operate on already-built supervised example rows. Unlike panel-window workflows, each row here is already one complete training example, so the split unit is the supervised row itself rather than a multi-row window origin.

## API Design

Proposed signatures:

```python
def make_supervised_arrays(
    data: Any,
    *,
    input_format: str = "auto",
    ds_col: str = "ds",
    target_cols: tuple[str, ...] = (),
    lags: Any = 5,
    horizon: int = 1,
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


def split_supervised_arrays(
    bundle: Any,
    *,
    valid_size: int | None = None,
    test_size: int | None = None,
    valid_frac: float | None = None,
    test_frac: float | None = None,
    gap: int = 0,
    min_train_size: int = 1,
) -> dict[str, dict[str, Any]]:
    ...
```

The split parameter set should mirror the existing workflow split helpers so the package keeps one predictable chronology model.

## Bundle Schema

`make_supervised_arrays()` should return:

- `X`: 2D numeric feature matrix
- `y`: 1D array for single-step targets, 2D array for multi-step direct targets
- `feature_names`: ordered tuple of feature column names
- `target_names`: ordered tuple of target column names
- `index`: DataFrame with:
  - `unique_id`
  - `ds`
  - `target_t`
- `metadata`: dict with normalized configuration and shape information

The feature ordering must match the frame output exactly. The target ordering must match the frame target columns exactly:

- `("y_target",)` for horizon 1
- `("y_t+1", ..., "y_t+h")` for direct multi-step outputs

## Metadata and Semantics

The metadata should keep downstream training code self-describing. At minimum it should include:

- `input_format`
- `horizon`
- `n_series`
- `n_rows`
- `n_features`
- `n_targets`
- `feature_names`
- `target_names`

For `split_supervised_arrays()`, each partition metadata should carry forward the original configuration and update:

- `n_rows`
- `partition`

Split semantics:

- sort rows independently within each `unique_id` by `ds`, then `target_t`
- allocate trailing rows to `test`, preceding rows to `valid`, and earlier rows to `train`
- apply `gap` in supervised-row units
- require at least `min_train_size` rows per series after holdouts and gap

## Validation and Error Handling

This wave should fail fast on malformed inputs:

- `TypeError` when `bundle` is not the expected dict structure
- `TypeError` when `index` is not a DataFrame
- `TypeError` when required index columns are missing
- `ValueError` when `X`, `y`, and `index` row counts disagree
- `ValueError` when `feature_names` or `target_names` do not match array shapes
- the same split-size validation errors already enforced by the current workflow helpers

No silent reshaping, no target-column guessing beyond the known frame conventions, and no split behavior that mixes per-series chronology should be introduced.

## Testing Strategy

Coverage should stay concentrated in:

- `tests/test_data_workflows.py`
- `tests/test_root_import.py`
- `tests/test_docs_rnn_generated.py` indirectly through regenerated API metadata

The tests should prove:

- dense arrays match the supervised frame for both features and targets
- multi-step direct targets become a 2D `y` matrix with stable target names
- split behavior preserves per-series chronological order
- malformed bundles fail clearly
- root exports and generated docs stay in sync

## Expected Outcome

After this wave, ForeSight's general supervised workflow layer will support both of the common downstream representations:

- inspectable dataframes via `make_supervised_frame()`
- direct training bundles via `make_supervised_arrays()`

Users will also be able to split already-built supervised examples chronologically without reconstructing the frame or re-implementing array extraction logic outside the package.
