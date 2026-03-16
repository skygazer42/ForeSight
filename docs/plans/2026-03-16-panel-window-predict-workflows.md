# Panel Window Predict Workflows Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add public panel-window prediction helpers that expose forecast-time feature rows and dense arrays for global step-lag style models.

**Architecture:** Keep the public implementation in `src/foresight/data/workflows.py`. Reuse the existing panel-window feature ordering and naming conventions so prediction rows align with the current training window outputs. Export the new helpers from `src/foresight/data/__init__.py` and `src/foresight/__init__.py`, then sync generated API metadata.

**Tech Stack:** Python 3.10, pandas, numpy, pytest

---

### Task 1: Add Prediction Workflow Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

Add tests that prove:

```python
def test_make_panel_window_predict_frame_matches_training_features_at_cutoff() -> None:
    train_frame = make_panel_window_frame(...)
    pred_frame = make_panel_window_predict_frame(..., cutoff=...)
    assert pred_frame.loc[:, feature_cols].equals(expected.loc[:, feature_cols])


def test_make_panel_window_predict_arrays_matches_predict_frame() -> None:
    pred_frame = make_panel_window_predict_frame(...)
    pred_arrays = make_panel_window_predict_arrays(...)
    assert np.allclose(pred_arrays["X"], pred_frame.loc[:, feature_cols].to_numpy(dtype=float))
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because the new helpers do not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run tests to verify they fail for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing import or missing function

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add panel window predict workflow coverage"
```

### Task 2: Add Missing-Y and Export Tests

**Files:**
- Modify: `tests/test_data_workflows.py`
- Modify: `tests/test_root_import.py`

**Step 1: Write the failing tests**

Add tests that prove:

```python
def test_make_panel_window_predict_frame_allows_missing_future_y() -> None:
    pred_frame = make_panel_window_predict_frame(long_df_with_future_nan_y, ...)
    assert len(pred_frame) == horizon


def test_root_import_exports_panel_window_predict_workflows() -> None:
    assert hasattr(foresight, "make_panel_window_predict_frame")
    assert hasattr(foresight, "make_panel_window_predict_arrays")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: FAIL because the helpers and exports are missing

**Step 3: Write the minimal implementation**

Only update tests in this step.

**Step 4: Run tests to verify they fail for the right reason**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: FAIL on missing helper or export

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py tests/test_root_import.py
git commit -m "test: cover panel window predict exports"
```

### Task 3: Implement Prediction Helpers

**Files:**
- Modify: `src/foresight/data/workflows.py`
- Modify: `src/foresight/data/__init__.py`
- Modify: `src/foresight/__init__.py`
- Modify: `tests/test_data_workflows.py`
- Modify: `tests/test_root_import.py`

**Step 1: Write the minimal implementation**

Add:

```python
def make_panel_window_predict_frame(...): ...
def make_panel_window_predict_arrays(...): ...
```

Implementation requirements:

- preserve the same feature naming and ordering as `make_panel_window_frame()`
- emit one row per future step
- support missing future `y` after cutoff
- require finite history `y` and required future covariates
- expose stable index and metadata in the arrays bundle

**Step 2: Run focused tests to verify they pass**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 3: Refactor only if needed**

If training and prediction row builders duplicate too much feature emission logic, extract a small shared internal helper without changing public behavior.

**Step 4: Run focused tests again**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tests/test_data_workflows.py tests/test_root_import.py
git commit -m "feat: add panel window predict workflows"
```

### Task 4: Sync API Metadata and Verify

**Files:**
- Modify: `tools/generate_model_capability_docs.py`
- Regenerate: `docs/api.md`
- Regenerate: `docs/models.md`

**Step 1: Update API metadata**

Add metadata rows for:

- `make_panel_window_predict_frame`
- `make_panel_window_predict_arrays`

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
git commit -m "docs: sync panel window predict workflow metadata"
```
