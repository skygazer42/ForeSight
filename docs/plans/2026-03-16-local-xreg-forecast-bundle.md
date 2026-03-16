# Local XReg Forecast Bundle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a public workflow helper that exposes per-series local xreg forecast arrays and index metadata from observed history plus known future covariates.

**Architecture:** Keep the helper in `src/foresight/data/workflows.py` and reuse the same forecast-time semantics already enforced in the local xreg service path. Export it from `src/foresight/data/__init__.py` and `src/foresight/__init__.py`, then sync generated API metadata so the public API stays self-describing.

**Tech Stack:** Python 3.10, pandas, numpy, pytest

---

### Task 1: Add Forecast Bundle Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

Add tests that prove:

```python
def test_make_local_xreg_forecast_bundle_merged_long_df_shapes_and_values() -> None:
    bundle = make_local_xreg_forecast_bundle(long_df, horizon=3, x_cols=("promo",))
    assert bundle["metadata"]["n_series"] == 1
    assert bundle["groups"][0]["future_exog"].shape == (3, 1)


def test_make_local_xreg_forecast_bundle_future_df_matches_merged_long_df() -> None:
    merged = make_local_xreg_forecast_bundle(merged_long_df, horizon=3, x_cols=("promo",))
    split = make_local_xreg_forecast_bundle(history_df, future_df=future_df, horizon=3, x_cols=("promo",))
    assert ...
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because the new helper does not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run tests to verify they fail for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing import or missing function

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add local xreg forecast bundle coverage"
```

### Task 2: Add Alias / Export Tests

**Files:**
- Modify: `tests/test_data_workflows.py`
- Modify: `tests/test_root_import.py`

**Step 1: Write the failing tests**

Add tests that prove:

```python
def test_make_local_xreg_forecast_bundle_supports_future_x_cols_alias() -> None:
    bundle = make_local_xreg_forecast_bundle(..., future_x_cols=("promo",))
    assert bundle["metadata"]["x_cols"] == ("promo",)


def test_make_local_xreg_forecast_bundle_rejects_historic_x_cols() -> None:
    with pytest.raises(ValueError, match="historic_x_cols are not yet supported"):
        make_local_xreg_forecast_bundle(..., historic_x_cols=("promo_hist",), future_x_cols=("promo",))
```

Also extend root API export coverage for the new helper.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: FAIL because the helper and export are missing

**Step 3: Write the minimal implementation**

Only update tests in this step.

**Step 4: Run tests to verify they fail for the right reason**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: FAIL on missing helper or export

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py tests/test_root_import.py
git commit -m "test: cover local xreg forecast bundle exports"
```

### Task 3: Implement Forecast Bundle Helper

**Files:**
- Modify: `src/foresight/data/workflows.py`
- Modify: `src/foresight/data/__init__.py`
- Modify: `src/foresight/__init__.py`
- Modify: `tests/test_data_workflows.py`
- Modify: `tests/test_root_import.py`

**Step 1: Write the minimal implementation**

Add:

```python
def make_local_xreg_forecast_bundle(...): ...
```

Implementation requirements:

- support merged `long_df` and optional `future_df`
- merge `x_cols` and `future_x_cols`
- reject `historic_x_cols`
- emit per-series `train_y`, `train_exog`, `future_exog`, `train_index`, and `future_index`
- expose stable top-level metadata

**Step 2: Run focused tests to verify they pass**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 3: Refactor only if needed**

If future-frame normalization or duplicate validation would otherwise duplicate service logic too much, extract small internal workflow helpers without changing public behavior.

**Step 4: Run focused tests again**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tests/test_data_workflows.py tests/test_root_import.py
git commit -m "feat: add local xreg forecast bundle"
```

### Task 4: Sync API Metadata and Verify

**Files:**
- Modify: `tools/generate_model_capability_docs.py`
- Regenerate: `docs/api.md`
- Regenerate: `docs/models.md`

**Step 1: Update API metadata**

Add metadata for:

- `make_local_xreg_forecast_bundle`

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
git commit -m "docs: sync local xreg forecast bundle metadata"
```
