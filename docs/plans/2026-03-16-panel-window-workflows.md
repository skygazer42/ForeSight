# Panel Window Workflows Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add panel-window workflow helpers that turn long-format time series into inspectable training frames and dense `X` / `y` arrays using ForeSight's existing lag-role semantics.

**Architecture:** Keep the implementation inside `src/foresight/data/workflows.py` and share a single internal window-building path between the frame and arrays APIs. Export the new helpers from `src/foresight/data/__init__.py` and `src/foresight/__init__.py`, then update generated API metadata so docs and root imports stay aligned.

**Tech Stack:** Python 3.10, pandas, numpy, pytest

---

### Task 1: Add Panel Window Frame Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

```python
def test_make_panel_window_frame_multi_series_multi_step_schema() -> None:
    out = make_panel_window_frame(
        long_df,
        horizon=2,
        target_lags=(1, 2),
        seasonal_lags=(3,),
        historic_x_lags=(1,),
        future_x_lags=(0, 1),
        x_cols=("promo",),
        add_time_features=False,
    )
    assert list(out.columns[:5]) == ["unique_id", "cutoff_ds", "target_ds", "step", "y"]
    assert "y_lag_1" in out.columns
    assert "y_seasonal_lag_3" in out.columns
    assert "historic_x__promo_lag_1" in out.columns
    assert "future_x__promo_lag_0" in out.columns


def test_make_panel_window_frame_future_x_lag_zero_tracks_target_step() -> None:
    out = make_panel_window_frame(
        long_df,
        horizon=2,
        target_lags=(1,),
        future_x_lags=(0, 1),
        x_cols=("promo",),
    )
    assert ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because `make_panel_window_frame` does not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing import or missing function

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add panel window frame coverage"
```

### Task 2: Add Panel Window Arrays and Error Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

```python
def test_make_panel_window_arrays_matches_frame_output() -> None:
    frame = make_panel_window_frame(long_df, horizon=2, x_cols=("promo",))
    arrays = make_panel_window_arrays(long_df, horizon=2, x_cols=("promo",))
    assert arrays["X"].shape[0] == len(frame)
    assert arrays["y"].shape[0] == len(frame)
    assert tuple(arrays["feature_names"]) == tuple(frame.columns[5:])


def test_make_panel_window_frame_rejects_duplicate_timestamps() -> None:
    with pytest.raises(ValueError, match="align_long_df"):
        make_panel_window_frame(long_df_with_duplicates, horizon=1)


def test_make_panel_window_frame_raises_when_no_windows_can_be_built() -> None:
    with pytest.raises(ValueError, match="enough history"):
        make_panel_window_frame(short_long_df, horizon=2, target_lags=(1, 2, 3))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because arrays helper and validations do not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing helpers or missing validations

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add panel window arrays and validation coverage"
```

### Task 3: Add Root Export Tests

**Files:**
- Modify: `tests/test_root_import.py`

**Step 1: Write the failing tests**

```python
def test_root_import_exports_panel_window_workflows() -> None:
    assert hasattr(foresight, "make_panel_window_frame")
    assert hasattr(foresight, "make_panel_window_arrays")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_root_import.py -q`
Expected: FAIL because the new exports are missing

**Step 3: Write the minimal implementation**

Only update tests in this step.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_root_import.py -q`
Expected: FAIL because root exports are not wired yet

**Step 5: Commit**

```bash
git add tests/test_root_import.py
git commit -m "test: cover panel window root exports"
```

### Task 4: Implement Shared Panel Window Builders

**Files:**
- Modify: `src/foresight/data/workflows.py`
- Modify: `src/foresight/data/__init__.py`
- Modify: `src/foresight/__init__.py`

**Step 1: Write the minimal implementation**

Add shared internals and public APIs:

```python
def make_panel_window_frame(...): ...
def make_panel_window_arrays(...): ...
```

Implementation requirements:

- long-format validation with duplicate timestamp checks per series
- normalized lag-role semantics aligned with `global_regression`
- stable metadata columns: `unique_id`, `cutoff_ds`, `target_ds`, `step`, `y`
- deterministic feature ordering for target, seasonal, historic exogenous, future exogenous, and optional time features
- arrays helper built directly from the frame helper output

**Step 2: Run focused tests to verify they pass**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 3: Refactor only if needed**

If feature assembly becomes repetitive, extract a tiny shared row-builder or lag normalizer without changing behavior.

**Step 4: Run focused tests again**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tests/test_data_workflows.py tests/test_root_import.py
git commit -m "feat: add panel window workflows"
```

### Task 5: Sync API Metadata and Verify

**Files:**
- Modify: `tools/generate_model_capability_docs.py`
- Regenerate: `docs/api.md`
- Regenerate: `docs/models.md`

**Step 1: Update API metadata**

Add metadata rows for:

- `make_panel_window_frame`
- `make_panel_window_arrays`

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
git commit -m "docs: sync panel window workflow metadata"
```
