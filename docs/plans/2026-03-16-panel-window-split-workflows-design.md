# Panel Window Split Workflows Design

## Goal

Extend ForeSight's training-oriented panel-window workflow layer with chronological split helpers that operate on already-built windows instead of forcing users to go back to raw long-format data:

- `split_panel_window_frame`
- `split_panel_window_arrays`

This wave should make it straightforward to build repeatable sklearn-style and custom training pipelines such as:

`prepare -> align -> clip -> enrich -> split_long_df -> scale -> make_panel_window_* -> split_panel_window_* -> train`

## Why This Wave

The package now exposes a much stronger training-data stack than it did originally:

- long-format preparation and validation
- deterministic long-format train/valid/test splitting
- reversible long-format scaling
- panel window frame and array builders
- packed sequence tensor builders and chronological splits
- structured sequence block builders and chronological splits

That leaves one obvious asymmetry: the `make_panel_window_*` family can build trainable examples, but it still cannot split those examples after materialization.

For many workflows that is inconvenient:

- users often want to build one inspectable panel-window dataset and split it later
- downstream estimators may consume dense arrays, not raw long DataFrames
- re-running `make_panel_window_*` over pre-split raw data can be more awkward than splitting existing windows
- the package already established window-level split semantics for packed and structured sequence bundles

This wave closes that gap and rounds out the panel-window family so it matches the rest of the training-oriented workflow surface.

## Scope

This wave adds two helpers in `src/foresight/data/workflows.py`:

### `split_panel_window_frame`

Chronologically split a dataframe returned by `make_panel_window_frame()` into per-series `train`, `valid`, and `test` partitions based on distinct `(unique_id, cutoff_ds)` window origins.

### `split_panel_window_arrays`

Chronologically split a bundle returned by `make_panel_window_arrays()` while preserving `X`, `y`, `index`, `feature_names`, and metadata for each partition.

This wave does not change `make_panel_window_frame()` or `make_panel_window_arrays()` behavior beyond whatever metadata is minimally required to support stable splitting.

## Architecture

The implementation should stay in the existing workflow layer:

- core logic in `src/foresight/data/workflows.py`
- exports in `src/foresight/data/__init__.py`
- root exports in `src/foresight/__init__.py`

The critical architectural rule is:

`window origins drive the split, horizon-expanded rows follow the origin`

That means the split unit is not an individual dataframe row or array row. It is the distinct forecasting window defined by:

- `unique_id`
- `cutoff_ds`

Every target step that belongs to the same window origin must remain in the same partition. Otherwise horizon rows from one logical training example could leak across train/valid/test boundaries.

The simplest way to achieve this is:

1. extract unique window origins from the frame or array bundle
2. partition those origins independently within each `unique_id`
3. map the selected origins back to the full horizon-expanded rows

This mirrors the semantics already used by `split_panel_sequence_tensors()` and `split_panel_sequence_blocks()`, which also split on window-level chronology rather than individual sequence positions.

## API Design

Proposed signatures:

```python
def split_panel_window_frame(
    frame: Any,
    *,
    valid_size: int | None = None,
    test_size: int | None = None,
    valid_frac: float | None = None,
    test_frac: float | None = None,
    gap: int = 0,
    min_train_size: int = 1,
) -> dict[str, pd.DataFrame]:
    ...


def split_panel_window_arrays(
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

The parameter set intentionally mirrors the existing chronological split helpers:

- `split_long_df`
- `split_panel_sequence_tensors`
- `split_panel_sequence_blocks`

That keeps the package predictable and avoids inventing a new split vocabulary for panel windows.

## Structured Outputs

`split_panel_window_frame()` should return:

- `train`
- `valid`
- `test`

Each partition is a dataframe with the same columns and row ordering conventions as the input frame.

`split_panel_window_arrays()` should return the same partition names, where each partition contains:

- `X`
- `y`
- `index`
- `feature_names`
- `metadata`

Each partition should preserve all rows associated with a selected window origin. Since every window expands into `horizon` target rows, the partition row counts should remain multiples of the per-window horizon for each included origin.

## Metadata and Semantics

The split behavior should be defined in terms of unique window origins:

- order window origins by `cutoff_ds` within each `unique_id`
- allocate the last origins to `test`, the preceding origins to `valid`, and the earlier origins to `train`
- apply `gap` in window-origin units, not expanded target-row units
- require at least `min_train_size` origins per series after applying holdouts and gap

For array partitions, metadata should remain self-describing. At minimum each partition metadata should carry forward the original normalized configuration and update:

- `n_windows`
- `n_rows`
- `partition`

The original feature ordering and feature names must remain unchanged.

## Validation and Error Handling

This wave should fail fast on malformed inputs:

- `TypeError` when `frame` is not a pandas DataFrame
- `TypeError` when `bundle` is not the expected dict structure
- `TypeError` when required frame or index columns are missing
- `ValueError` when row counts do not align with `X`, `y`, or `index`
- the same split-size validation errors already enforced by the existing workflow helpers

No silent regrouping, no per-row splitting, and no fallback behavior that might leak horizon steps across partitions should be introduced.

## Testing Strategy

Coverage should stay concentrated in:

- `tests/test_data_workflows.py`
- `tests/test_root_import.py`
- `tests/test_docs_rnn_generated.py` indirectly through regenerated API metadata

The tests should prove:

- frame partitions preserve per-series chronological cutoff ordering
- all rows from one `(unique_id, cutoff_ds)` window stay together
- array partitions align exactly with frame-derived partitions
- malformed array bundles fail clearly
- root exports and generated docs stay in sync

## Expected Outcome

After this wave, ForeSight's panel-window workflow family will become symmetrical with the sequence workflow family:

- build panel-window frames or arrays
- split them chronologically without rebuilding raw long data
- train sklearn-style or custom models against stable partitioned artifacts

That makes the package's training-data surface more coherent and reduces the amount of external data plumbing users need to write around the existing workflow APIs.
