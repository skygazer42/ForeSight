# Packed Sequence Tensor Workflows Design

## Goal

Extend ForeSight's training-oriented data workflow layer with packed sequence tensor builders that bridge long-format panel data and sequence-model training loops:

- `make_panel_sequence_tensors`
- `split_panel_sequence_tensors`

This wave should make it straightforward to build repeatable global sequence-model datasets such as:

`prepare -> align -> clip -> enrich -> split -> scale -> make_panel_sequence_tensors -> train`

## Why This Wave

The package now has a solid dataframe-first workflow stack:

- canonical long-format validation and preparation
- deterministic alignment, clipping, and calendar enrichment
- chronological train/valid/test splitting
- reversible long-format scaling
- panel window frames and dense tabular arrays

What is still missing is the sequence-oriented workflow layer for global Torch-style models. The internal global model code in `src/foresight/models/torch_global.py` already constructs packed sequence tensors for training and prediction, but that logic is not available as a public workflow.

That leaves two gaps:

- users cannot build stable sequence-model training bundles without reimplementing internal logic
- the data workflow layer currently stops at tabular training shapes, even though the package now contains many global sequence models

This wave closes that gap by turning the existing internal packed-sequence conventions into public, inspectable, numpy/pandas-first workflow helpers.

## Scope

This wave adds two helpers in `src/foresight/data/workflows.py`:

### `make_panel_sequence_tensors`

Build a sequence-model training bundle from long-format panel data using a fixed `cutoff`, `context_length`, and `horizon`.

### `split_panel_sequence_tensors`

Split the training-window portion of a packed sequence bundle into chronological train/valid/test subsets without rebuilding raw long-format data.

This wave keeps the workflow layer framework-neutral:

- no `torch.Tensor`
- no `TensorDataset`
- no `DataLoader`

Public outputs should stay in numpy/pandas containers so they remain useful across Torch, JAX, NumPy, or custom loops.

## Architecture

The implementation should stay entirely in the existing data workflow layer:

- core logic in `src/foresight/data/workflows.py`
- exports in `src/foresight/data/__init__.py`
- root exports in `src/foresight/__init__.py`

The design should mirror the stable internal conventions already present in `src/foresight/models/torch_global.py`, especially `_build_panel_dataset(...)`:

- training samples are packed as `X.shape == (n_samples, context_length + horizon, input_dim)`
- targets are packed as `y.shape == (n_samples, horizon)`
- each sample carries a series id
- each series has one prediction sample aligned to the requested cutoff

Rather than exposing model-internal helpers directly, the workflow layer should reimplement the minimum required behavior in a clean public form. That keeps data workflows inspectable, avoids a hard dependency from `foresight.data` back into `foresight.models`, and reduces the risk of circular architecture.

## API Design

Proposed signatures:

```python
def make_panel_sequence_tensors(
    long_df: Any,
    *,
    cutoff: Any,
    horizon: int,
    context_length: int = 96,
    x_cols: tuple[str, ...] = (),
    normalize: bool = True,
    max_train_size: int | None = None,
    sample_step: int = 1,
    add_time_features: bool = True,
    dtype: Any = np.float64,
) -> dict[str, Any]:
    ...


def split_panel_sequence_tensors(
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

`make_panel_sequence_tensors()` should return a dictionary with three top-level keys:

- `train`
- `predict`
- `metadata`

`split_panel_sequence_tensors()` should return:

- `train`
- `valid`
- `test`

Each partition should preserve the same sub-structure as the original `train` block and should carry forward metadata required for downstream training.

## Bundle Structure

The `train` block should contain:

- `X`: packed sequence features with shape `(n_samples, context_length + horizon, input_dim)`
- `y`: packed horizon targets with shape `(n_samples, horizon)`
- `series_id`: integer-encoded series ids with shape `(n_samples,)`
- `window_index`: DataFrame with at least:
  - `unique_id`
  - `cutoff_ds`
  - `target_start_ds`
  - `target_end_ds`

The `predict` block should contain one sequence sample per eligible series:

- `X`
- `series_id`
- `index`: DataFrame with at least:
  - `unique_id`
  - `cutoff_ds`
  - `target_start_ds`
  - `target_end_ds`
- `target_mean`
- `target_std`

When `normalize=False`, `target_mean` and `target_std` should still be present for schema stability, using `0.0` and `1.0`.

The `metadata` block should include:

- `cutoff`
- `context_length`
- `horizon`
- `x_cols`
- `normalize`
- `sample_step`
- `max_train_size`
- `add_time_features`
- `channel_names`
- `time_feature_names`
- `n_series`
- `n_train_windows`
- `n_predict_windows`
- `input_dim`

## Packed Channel Semantics

The packed feature tensor should use a stable channel order in the last dimension:

1. target `y` channel
2. requested `x_cols`
3. generated time feature channels

This order should be reflected exactly in `metadata["channel_names"]`.

Within the sequence axis:

- the first `context_length` steps represent observed context
- the last `horizon` steps represent forecast positions

The target `y` channel should follow the current internal global-model convention:

- past context rows contain observed target values
- future forecast rows contain zeros

Exogenous and time-feature channels should contain their real aligned values for both context and horizon rows whenever available.

## Normalization and Splitting Semantics

Normalization should stay aligned with the current global Torch implementation:

- only the target series `y` is normalized
- normalization is computed per series, using only the cutoff-aligned training history available to that series
- `x_cols` and time features remain in their original scale

This keeps the bundle useful for both the existing global models and outside training loops.

`split_panel_sequence_tensors()` should operate on the already-built `train` windows rather than on raw long-format data. Partitioning should be based on `window_index`, using the same per-series chronological logic already established by `split_long_df()`:

- split independently per `unique_id`
- maintain chronological order by `cutoff_ds`, then `target_start_ds`
- support sizes or fractions, optional `gap`, and `min_train_size`

The `predict` block should not be repartitioned. It should remain attached to the original cutoff-aligned forecasting bundle.

## Error Handling

This wave should fail fast on malformed or ambiguous input:

- `TypeError` when `long_df` is not a pandas DataFrame
- `KeyError` when required columns are missing
- `KeyError` when requested `x_cols` are missing
- `ValueError` for invalid `horizon`, `context_length`, `sample_step`, or `max_train_size`
- `ValueError` when duplicate timestamps appear within a series, with guidance to run `align_long_df()` first
- `ValueError` when `y` or requested covariates contain non-finite values in needed regions
- `ValueError` when no training windows can be built
- `ValueError` when no prediction windows can be built for the requested cutoff
- `TypeError` when `split_panel_sequence_tensors()` receives a malformed bundle

No PyTorch-specific errors or classes should leak into the public data workflow API.

## Testing Strategy

Coverage should stay concentrated in:

- `tests/test_data_workflows.py`
- `tests/test_root_import.py`
- `tests/test_docs_rnn_generated.py` indirectly through regenerated API metadata

The tests should prove:

- packed training shapes and stable channel ordering
- `predict` block has one sample per eligible series
- `normalize=True` returns usable `target_mean` / `target_std`
- `sample_step` and `max_train_size` alter the number of windows as expected
- `split_panel_sequence_tensors()` preserves chronology and per-series boundaries
- duplicate timestamps and insufficient-history inputs fail clearly
- root exports and generated docs stay in sync

## Expected Outcome

After this wave, ForeSight will expose a reusable sequence-model workflow layer above raw long-format data and below global Torch-style models. Users will be able to build packed sequence bundles with explicit metadata, train any compatible sequence model on them, and split training windows chronologically without rebuilding the entire dataset outside the package.
