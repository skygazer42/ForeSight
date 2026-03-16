# Panel Window Split Workflows Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add panel-window split helpers that chronologically partition materialized frame and array workflows by window origin while preserving horizon-expanded training rows.

**Architecture:** Keep the public implementation in `src/foresight/data/workflows.py`. Define split behavior in terms of distinct `(unique_id, cutoff_ds)` window origins so frame and array partitions stay aligned. Export the new helpers from `src/foresight/data/__init__.py` and `src/foresight/__init__.py`, then sync generated API metadata.

**Tech Stack:** Python 3.10, pandas, numpy, pytest

---

### Task 1: Add Panel Window Split Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

Add tests that prove:

```python
def test_split_panel_window_frame_respects_per_series_window_order() -> None:
    frame = make_panel_window_frame(...)
    parts = split_panel_window_frame(frame, valid_size=1, test_size=1)
    assert set(parts) == {"train", "valid", "test"}


def test_split_panel_window_arrays_matches_frame_partitions() -> None:
    frame = make_panel_window_frame(...)
    arrays = make_panel_window_arrays(...)
    frame_parts = split_panel_window_frame(frame, valid_size=1, test_size=1)
    array_parts = split_panel_window_arrays(arrays, valid_size=1, test_size=1)
    assert array_parts["train"]["X"].shape[0] == len(frame_parts["train"])
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because `split_panel_window_frame` and `split_panel_window_arrays` do not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run tests to verify they fail for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing helper imports or missing functions

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add panel window split coverage"
```

### Task 2: Add Root Export Tests

**Files:**
- Modify: `tests/test_root_import.py`

**Step 1: Write the failing test**

```python
def test_root_import_exports_panel_window_split_workflows() -> None:
    assert hasattr(foresight, "split_panel_window_frame")
    assert hasattr(foresight, "split_panel_window_arrays")
```

**Step 2: Run tests to verify it fails**

Run: `pytest tests/test_root_import.py -q`
Expected: FAIL because root exports are missing

**Step 3: Write the minimal implementation**

Only update tests in this step.

**Step 4: Run tests to verify it fails for the right reason**

Run: `pytest tests/test_root_import.py -q`
Expected: FAIL on missing exports

**Step 5: Commit**

```bash
git add tests/test_root_import.py
git commit -m "test: cover panel window split workflow exports"
```

### Task 3: Implement Panel Window Split Helpers

**Files:**
- Modify: `src/foresight/data/workflows.py`
- Modify: `src/foresight/data/__init__.py`
- Modify: `src/foresight/__init__.py`
- Modify: `tests/test_data_workflows.py`
- Modify: `tests/test_root_import.py`

**Step 1: Write the minimal implementation**

Add:

```python
def split_panel_window_frame(...): ...
def split_panel_window_arrays(...): ...
```

Implementation requirements:

- partition by distinct `(unique_id, cutoff_ds)` window origins
- preserve all horizon-expanded rows for each selected origin
- keep frame and array split semantics aligned
- reuse the existing split-size validation helpers instead of inventing a new policy
- preserve feature ordering and array metadata

**Step 2: Run focused tests to verify they pass**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 3: Refactor only if needed**

If frame and array paths duplicate origin-partition logic, extract a small internal helper for partitioning window origins without changing public behavior.

**Step 4: Run focused tests again**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tests/test_data_workflows.py tests/test_root_import.py
git commit -m "feat: add panel window split workflows"
```

### Task 4: Sync API Metadata and Verify

**Files:**
- Modify: `tools/generate_model_capability_docs.py`
- Regenerate: `docs/api.md`
- Regenerate: `docs/models.md`

**Step 1: Update API metadata**

Add metadata rows for:

- `split_panel_window_frame`
- `split_panel_window_arrays`

**Step 2: Regenerate docs**

Run: `python tools/generate_model_capability_docs.py`
Expected: `Wrote: docs/models.md` and `Wrote: docs/api.md`

**Step 3: Run verification**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py tests/test_docs_rnn_generated.py -q`
Expected: PASS

**Step 4: Run broader lint and regression checks**

Run: `ruff check src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tools/generate_model_capability_docs.py tests/test_data_workflows.py tests/test_root_import.py`

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py tests/test_cli_data.py tests/test_docs_rnn_generated.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tools/generate_model_capability_docs.py docs/api.md docs/models.md
git commit -m "docs: sync panel window split workflow metadata"
```
