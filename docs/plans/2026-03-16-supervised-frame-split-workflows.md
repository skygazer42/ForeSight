# Supervised Frame Split Workflow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `split_supervised_frame()` so materialized supervised dataframes can be split chronologically without converting to arrays first.

**Architecture:** Keep the public implementation in `src/foresight/data/workflows.py`. Extract a shared per-series supervised-row split helper so `split_supervised_frame()` and `split_supervised_arrays()` stay aligned. Export the new helper from `src/foresight/data/__init__.py` and `src/foresight/__init__.py`, then sync generated API metadata.

**Tech Stack:** Python 3.10, pandas, numpy, pytest

---

### Task 1: Add Supervised Frame Split Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

Add tests that prove:

```python
def test_split_supervised_frame_respects_per_series_order() -> None:
    frame = make_supervised_frame(...)
    parts = split_supervised_frame(frame, valid_size=1, test_size=1)
    assert set(parts) == {"train", "valid", "test"}


def test_split_supervised_frame_matches_array_partitions() -> None:
    frame = make_supervised_frame(...)
    bundle = make_supervised_arrays(...)
    frame_parts = split_supervised_frame(frame, valid_size=1, test_size=1)
    array_parts = split_supervised_arrays(bundle, valid_size=1, test_size=1)
    assert frame_parts["train"]["ds"].tolist() == array_parts["train"]["index"]["ds"].tolist()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because `split_supervised_frame` does not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run tests to verify they fail for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing helper import or missing function

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add supervised frame split coverage"
```

### Task 2: Add Root Export Test

**Files:**
- Modify: `tests/test_root_import.py`

**Step 1: Write the failing test**

```python
def test_root_import_exports_supervised_frame_split_workflow() -> None:
    assert hasattr(foresight, "split_supervised_frame")
```

**Step 2: Run tests to verify it fails**

Run: `pytest tests/test_root_import.py -q`
Expected: FAIL because root export is missing

**Step 3: Write the minimal implementation**

Only update tests in this step.

**Step 4: Run tests to verify it fails for the right reason**

Run: `pytest tests/test_root_import.py -q`
Expected: FAIL on missing export

**Step 5: Commit**

```bash
git add tests/test_root_import.py
git commit -m "test: cover supervised frame split export"
```

### Task 3: Implement Supervised Frame Split Helper

**Files:**
- Modify: `src/foresight/data/workflows.py`
- Modify: `src/foresight/data/__init__.py`
- Modify: `src/foresight/__init__.py`
- Modify: `tests/test_data_workflows.py`
- Modify: `tests/test_root_import.py`

**Step 1: Write the minimal implementation**

Add:

```python
def split_supervised_frame(...): ...
```

Implementation requirements:

- validate the frame shape and required columns
- split chronologically per `unique_id` by `ds`, then `target_t`
- preserve all original columns and row ordering
- keep `split_supervised_arrays()` aligned by sharing the same row-position logic

**Step 2: Run focused tests to verify they pass**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 3: Refactor only if needed**

If frame and array split paths duplicate chronology logic, extract a small internal helper for per-series supervised row partitioning without changing public behavior.

**Step 4: Run focused tests again**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tests/test_data_workflows.py tests/test_root_import.py
git commit -m "feat: add supervised frame split workflow"
```

### Task 4: Sync API Metadata and Verify

**Files:**
- Modify: `tools/generate_model_capability_docs.py`
- Regenerate: `docs/api.md`
- Regenerate: `docs/models.md`

**Step 1: Update API metadata**

Add metadata row for:

- `split_supervised_frame`

**Step 2: Regenerate docs**

Run: `python tools/generate_model_capability_docs.py`
Expected: `Wrote: docs/models.md` and `Wrote: docs/api.md`

**Step 3: Run verification**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py tests/test_docs_rnn_generated.py -q`
Expected: PASS

**Step 4: Run broader lint and regression checks**

Run: `ruff check src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tools/generate_model_capability_docs.py tests/test_data_workflows.py tests/test_root_import.py`

Run: `pytest -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tools/generate_model_capability_docs.py docs/api.md docs/models.md
git commit -m "docs: sync supervised frame split workflow metadata"
```
