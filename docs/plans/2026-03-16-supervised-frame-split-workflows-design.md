# Supervised Frame Split Workflow Design

## Goal

Extend ForeSight's general supervised workflow family with an inspectable chronological split helper for the dataframe returned by `make_supervised_frame()`:

- `split_supervised_frame`

This wave should let users keep the dataframe-first representation all the way through feature engineering and only split after the supervised examples have already been materialized.

## Why This Wave

The supervised workflow family is now much stronger than it was originally:

- `make_supervised_frame()` exposes an inspectable dataframe for lag/rolling/seasonal/Fourier features
- `make_supervised_arrays()` exposes dense `X` / `y` / `index` bundles for direct training
- `split_supervised_arrays()` supports post-materialization chronological train/validation/test partitioning

That still leaves one asymmetry:

- the canonical inspectable representation is the dataframe
- but only the array representation can currently be split after materialization

That pushes users into an unnecessary choice:

- either keep the frame and manually write split logic, or
- convert to arrays earlier than they want just to get access to the split helper

This wave closes that gap and makes the supervised family consistent with the package's other workflow layers, where both the inspectable and training-oriented representations can be split chronologically.

## Alternatives Considered

### Option A: Tell users to split raw long data first

This is already possible with `split_long_df()`, but it is not always the right ergonomics. Once users have a fully materialized supervised frame, they often want to inspect and split those exact examples without reconstructing them from the raw long data.

### Option B: Tell users to convert to arrays and use `split_supervised_arrays()`

This works, but it unnecessarily hides the inspectable dataframe and forces an array conversion step when the user may still want to audit feature columns and target rows.

### Option C: Add `split_supervised_frame()` on the materialized frame

This is the recommended option. It preserves the dataframe-first philosophy, avoids duplicate external split code, and can share chronology semantics with `split_supervised_arrays()`.

## Scope

This wave adds one helper in `src/foresight/data/workflows.py`:

### `split_supervised_frame`

Chronologically split a dataframe returned by `make_supervised_frame()` into `train`, `valid`, and `test` partitions independently within each `unique_id`.

This wave should also refactor the internal split path for `split_supervised_arrays()` if that is the cleanest way to keep the two helpers aligned.

## Architecture

The implementation should stay in the existing workflow layer:

- core logic in `src/foresight/data/workflows.py`
- exports in `src/foresight/data/__init__.py`
- root exports in `src/foresight/__init__.py`

The key architectural rule is:

`supervised rows are the split unit`

Unlike panel-window workflows, each row in the supervised frame already represents one complete training example. There is no multi-row window group to keep together. That means split semantics should operate per row:

- sort by `ds`, then `target_t` within each `unique_id`
- allocate trailing rows to `test`
- allocate preceding rows to `valid`
- leave the earlier rows in `train`
- apply `gap` in supervised-row units

To avoid drift, the cleanest implementation is to extract a shared internal helper that computes row positions for per-series chronological supervised splits. Both:

- `split_supervised_frame()`
- `split_supervised_arrays()`

should consume that shared helper.

## API Design

Proposed signature:

```python
def split_supervised_frame(
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
```

The parameter set intentionally mirrors the existing split helpers:

- `split_long_df`
- `split_panel_window_frame`
- `split_supervised_arrays`

## Output Semantics

`split_supervised_frame()` should return:

- `train`
- `valid`
- `test`

Each partition should preserve:

- all original columns
- chronological ordering within each `unique_id`
- exact feature and target values from the input frame

The helper should not add, remove, or rename columns.

## Validation and Error Handling

This wave should fail fast on malformed inputs:

- `TypeError` when `frame` is not a pandas DataFrame
- `TypeError` when required columns `unique_id`, `ds`, or `target_t` are missing
- `ValueError` when the frame is empty
- the same split-size validation errors already enforced by `split_supervised_arrays()`

No silent coercion of missing chronology columns and no global cross-series splitting should be introduced.

## Testing Strategy

Coverage should stay concentrated in:

- `tests/test_data_workflows.py`
- `tests/test_root_import.py`
- `tests/test_docs_rnn_generated.py` indirectly through regenerated API metadata

The tests should prove:

- per-series chronological row splitting for the supervised frame
- stable agreement between frame partitions and array partitions derived from the same source data
- root exports and generated docs stay in sync

## Expected Outcome

After this wave, the supervised workflow family becomes symmetrical:

- build an inspectable frame
- split the frame chronologically
- optionally convert to arrays either before or after inspection

That reduces external data plumbing and keeps ForeSight's dataframe-first workflow style coherent across both sequence-oriented and tabular-supervised training paths.
