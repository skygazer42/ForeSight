# Supervised Array Workflows Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add dense supervised array helpers and chronological split support on top of `make_supervised_frame()` without duplicating feature-engineering logic.

**Architecture:** Keep the public implementation in `src/foresight/data/workflows.py`. Define `make_supervised_arrays()` as a deterministic adapter over `make_supervised_frame()`, and define `split_supervised_arrays()` in terms of per-series chronological row splitting over the materialized supervised examples. Export the new helpers from `src/foresight/data/__init__.py` and `src/foresight/__init__.py`, then sync generated API metadata.

**Tech Stack:** Python 3.10, pandas, numpy, pytest

---

### Task 1: Add Supervised Array Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

Add tests that prove:

```python
def test_make_supervised_arrays_matches_frame_output() -> None:
    frame = make_supervised_frame(...)
    bundle = make_supervised_arrays(...)
    assert np.allclose(bundle["X"], frame.loc[:, feature_cols].to_numpy(dtype=float))


def test_make_supervised_arrays_multistep_target_is_2d() -> None:
    bundle = make_supervised_arrays(..., horizon=2)
    assert bundle["y"].shape[1] == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because `make_supervised_arrays` does not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run tests to verify they fail for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing import or missing helper

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add supervised array workflow coverage"
```

### Task 2: Add Split and Export Tests

**Files:**
- Modify: `tests/test_data_workflows.py`
- Modify: `tests/test_root_import.py`

**Step 1: Write the failing tests**

Add tests that prove:

```python
def test_split_supervised_arrays_respects_per_series_order() -> None:
    bundle = make_supervised_arrays(...)
    parts = split_supervised_arrays(bundle, valid_size=1, test_size=1)
    assert set(parts) == {"train", "valid", "test"}


def test_root_import_exports_supervised_array_workflows() -> None:
    assert hasattr(foresight, "make_supervised_arrays")
    assert hasattr(foresight, "split_supervised_arrays")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: FAIL because split helper and exports are missing

**Step 3: Write the minimal implementation**

Only update tests in this step.

**Step 4: Run tests to verify they fail for the right reason**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: FAIL on missing helper or missing export

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py tests/test_root_import.py
git commit -m "test: add supervised array split and export coverage"
```

### Task 3: Implement Supervised Array Helpers

**Files:**
- Modify: `src/foresight/data/workflows.py`
- Modify: `src/foresight/data/__init__.py`
- Modify: `src/foresight/__init__.py`
- Modify: `tests/test_data_workflows.py`
- Modify: `tests/test_root_import.py`

**Step 1: Write the minimal implementation**

Add:

```python
def make_supervised_arrays(...): ...
def split_supervised_arrays(...): ...
```

Implementation requirements:

- derive arrays directly from `make_supervised_frame(...)`
- preserve exact feature ordering and target ordering from the frame
- expose 1D `y` for single-step and 2D `y` for multi-step direct targets
- preserve `index` alignment during split
- keep split semantics per-series and chronological

**Step 2: Run focused tests to verify they pass**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 3: Refactor only if needed**

If frame-to-array conversion and split validation duplicate shape checks, extract a small internal helper without changing public behavior.

**Step 4: Run focused tests again**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tests/test_data_workflows.py tests/test_root_import.py
git commit -m "feat: add supervised array workflows"
```

### Task 4: Sync API Metadata and Verify

**Files:**
- Modify: `tools/generate_model_capability_docs.py`
- Regenerate: `docs/api.md`
- Regenerate: `docs/models.md`

**Step 1: Update API metadata**

Add metadata rows for:

- `make_supervised_arrays`
- `split_supervised_arrays`

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
git commit -m "docs: sync supervised array workflow metadata"
```
