# Structured Sequence Block Workflows Design

## Goal

Extend ForeSight's sequence-oriented data workflow layer with explicit encoder-decoder style block builders that sit on top of packed sequence tensors:

- `make_panel_sequence_blocks`
- `split_panel_sequence_blocks`

This wave should make it straightforward to build repeatable training bundles for structured sequence models such as TimeXer, encoder-decoder Transformers, Seq2Seq, and DeepAR-style workflows without forcing callers to slice packed channel tensors manually.

## Why This Wave

The package now exposes two useful sequence-data layers:

- packed sequence tensors for global sequence models
- chronological splitting of packed training windows

That closes the gap between long-format panel data and internal Torch-style training loops, but it still leaves one practical problem for external training code and future model adapters: packed tensors are efficient, not expressive.

Today, callers still need to remember the exact layout:

- context rows vs. horizon rows
- target channel vs. exogenous channels vs. time channels
- how future target values are masked

That is tolerable for a single internal helper, but weak as a reusable public workflow. Many sequence model families naturally think in structured blocks:

- `past_y`
- `future_y_seed`
- `past_x`
- `future_x`
- `past_time`
- `future_time`

This wave adds that explicit layer while keeping it grounded in the already-validated packed-sequence behavior.

## Scope

This wave adds two helpers in `src/foresight/data/workflows.py`:

### `make_panel_sequence_blocks`

Build a structured sequence-model bundle from long-format panel data by first constructing packed tensors and then deterministically exposing them as named encoder-decoder blocks.

### `split_panel_sequence_blocks`

Chronologically split the training-window portion of a structured block bundle into train/valid/test subsets while preserving the block schema.

This wave does not replace `make_panel_sequence_tensors()` or `split_panel_sequence_tensors()`. Instead, it builds on them and acts as the more expressive public view for model families that want explicit blocks rather than packed channels.

## Architecture

The implementation should stay in the existing data workflow layer:

- core logic in `src/foresight/data/workflows.py`
- exports in `src/foresight/data/__init__.py`
- root exports in `src/foresight/__init__.py`

The critical architectural rule is:

`long_df -> packed bundle -> structured block bundle`

`make_panel_sequence_blocks()` should internally call `make_panel_sequence_tensors()` and perform a deterministic transformation of the packed arrays into structured blocks. Likewise, `split_panel_sequence_blocks()` should be defined in terms of `split_panel_sequence_tensors()` or equivalent shared split semantics so chronology rules do not diverge.

This keeps one authoritative source for:

- cutoff handling
- per-series target normalization
- sample stepping
- max-train-size truncation
- time feature generation
- series id assignment

The structured layer is therefore a view adapter, not a second independent sequence builder.

## API Design

Proposed signatures:

```python
def make_panel_sequence_blocks(
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


def split_panel_sequence_blocks(
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

The public parameter set should intentionally mirror the packed workflow so the two helpers remain interchangeable views over the same underlying samples.

## Structured Bundle Schema

`make_panel_sequence_blocks()` should return a dictionary with:

- `train`
- `predict`
- `metadata`

The `train` block should contain:

- `past_y`: `(n_samples, context_length, 1)`
- `future_y_seed`: `(n_samples, horizon, 1)`
- `past_x`: `(n_samples, context_length, x_dim)`
- `future_x`: `(n_samples, horizon, x_dim)`
- `past_time`: `(n_samples, context_length, time_dim)`
- `future_time`: `(n_samples, horizon, time_dim)`
- `series_id`: `(n_samples,)`
- `target_y`: `(n_samples, horizon)`
- `window_index`: DataFrame with:
  - `unique_id`
  - `cutoff_ds`
  - `target_start_ds`
  - `target_end_ds`

The `predict` block should expose the same structural inputs except for `target_y`:

- `past_y`
- `future_y_seed`
- `past_x`
- `future_x`
- `past_time`
- `future_time`
- `series_id`
- `index`
- `target_mean`
- `target_std`

## Stable Zero-Width Block Semantics

The structured schema should remain stable even when optional inputs are absent.

If `x_cols` is empty:

- `past_x.shape == (n, context_length, 0)`
- `future_x.shape == (n, horizon, 0)`

If `add_time_features=False`:

- `past_time.shape == (n, context_length, 0)`
- `future_time.shape == (n, horizon, 0)`

The keys should not disappear. Returning zero-width arrays keeps the public schema stable and avoids forcing downstream model code into repeated conditional branching.

## Metadata and Semantics

The `metadata` block should include at least:

- `cutoff`
- `context_length`
- `horizon`
- `x_cols`
- `normalize`
- `sample_step`
- `max_train_size`
- `add_time_features`
- `time_feature_names`
- `x_dim`
- `time_dim`
- `n_series`
- `n_train_windows`
- `n_predict_windows`
- `block_layout`

`block_layout` should explicitly document the public meaning of each block and its canonical shape convention. This is especially important because the structured workflow is intended to remove guesswork from packed channel slicing.

Semantically:

- `past_y` contains the context target channel only
- `future_y_seed` contains the forecast-position target channel only and should remain zero-filled, matching the packed workflow
- `past_x` and `future_x` contain only user-requested covariates
- `past_time` and `future_time` contain only generated time features
- `target_y` remains normalized or unnormalized exactly as in the packed training bundle

## Splitting Design

`split_panel_sequence_blocks()` should split only the `train` windows, not the `predict` bundle. The return structure should be:

- `train`
- `valid`
- `test`

Each partition should preserve the same keys as the original `train` block:

- `past_y`
- `future_y_seed`
- `past_x`
- `future_x`
- `past_time`
- `future_time`
- `series_id`
- `target_y`
- `window_index`
- `metadata`

The split semantics should stay aligned with `split_panel_sequence_tensors()`:

- independent chronology per `unique_id`
- order by `cutoff_ds`, then `target_start_ds`
- support for `valid/test` sizes or fractions
- optional `gap`
- `min_train_size` guard

## Error Handling

This wave should fail fast on malformed input:

- `TypeError` when `bundle` is not a dictionary with the expected top-level keys
- `TypeError` when train/predict sections are missing required arrays or index frames
- `ValueError` when train blocks have incompatible row counts
- `ValueError` when packed channels cannot be reconciled with metadata dimensions
- the same split validation errors already enforced by packed sequence splitting

No silent best-effort repair should be added. If the packed bundle is malformed, the structured adapter should raise clearly.

## Testing Strategy

Coverage should stay concentrated in:

- `tests/test_data_workflows.py`
- `tests/test_root_import.py`
- `tests/test_docs_rnn_generated.py` indirectly through generated API metadata

The tests should prove:

- structured blocks match the packed bundle values exactly
- zero-width `x` and `time` blocks are still present and correctly shaped
- predict bundles preserve `target_mean` / `target_std`
- structured splitting preserves chronology and per-series boundaries
- root exports and generated docs stay in sync

## Expected Outcome

After this wave, ForeSight will expose a sequence-data workflow layer with two complementary public forms:

- packed sequence tensors for generic high-throughput model inputs
- structured sequence blocks for encoder-decoder style model code

That makes the package materially easier to use for additional trainable TS model families without duplicating the underlying sample-generation logic.
