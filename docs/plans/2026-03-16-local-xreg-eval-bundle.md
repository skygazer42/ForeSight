# Local XReg Eval Bundle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a public workflow helper that exposes walk-forward local xreg evaluation windows and their array payloads.

**Architecture:** Keep the helper in `src/foresight/data/workflows.py` and reuse the same rolling-origin semantics already enforced in `eval_model_long_df()`. Export it from `src/foresight/data/__init__.py` and `src/foresight/__init__.py`, then sync generated API metadata so the public API stays aligned with the docs.

**Tech Stack:** Python 3.10, pandas, numpy, pytest

---

### Task 1: Add Eval Bundle Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

Add tests that prove:

```python
def test_make_local_xreg_eval_bundle_matches_rolling_origin_windows() -> None:
    bundle = make_local_xreg_eval_bundle(...)
    assert bundle["metadata"]["n_windows"] == 2
    assert bundle["windows"][0]["cutoff_ds"] == ...


def test_make_local_xreg_eval_bundle_max_train_size_rolls_train_window() -> None:
    bundle = make_local_xreg_eval_bundle(..., max_train_size=4)
    assert bundle["windows"][1]["train_index"]["ds"].iloc[0] == ...
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
git commit -m "test: add local xreg eval bundle coverage"
```

### Task 2: Add Alias / Export Tests

**Files:**
- Modify: `tests/test_data_workflows.py`
- Modify: `tests/test_root_import.py`

**Step 1: Write the failing tests**

Add tests that prove:

```python
def test_make_local_xreg_eval_bundle_supports_future_x_cols_alias() -> None:
    bundle = make_local_xreg_eval_bundle(..., future_x_cols=("promo",))
    assert bundle["metadata"]["x_cols"] == ("promo",)


def test_make_local_xreg_eval_bundle_rejects_historic_x_cols() -> None:
    with pytest.raises(ValueError, match="historic_x_cols are not yet supported"):
        make_local_xreg_eval_bundle(..., historic_x_cols=("promo_hist",), future_x_cols=("promo",))
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
git commit -m "test: cover local xreg eval bundle exports"
```

### Task 3: Implement Eval Bundle Helper

**Files:**
- Modify: `src/foresight/data/workflows.py`
- Modify: `src/foresight/data/__init__.py`
- Modify: `src/foresight/__init__.py`
- Modify: `tests/test_data_workflows.py`
- Modify: `tests/test_root_import.py`

**Step 1: Write the minimal implementation**

Add:

```python
def make_local_xreg_eval_bundle(...): ...
```

Implementation requirements:

- support fully observed `long_df` only
- merge `x_cols` and `future_x_cols`
- reject `historic_x_cols`
- generate rolling-origin windows with `rolling_origin_splits()`
- emit per-window `train_y`, `actual_y`, `train_exog`, `future_exog`, `train_index`, and `test_index`
- expose stable top-level metadata including skip counts and total window count

**Step 2: Run focused tests to verify they pass**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 3: Refactor only if needed**

If eval-window extraction duplicates too much array validation or covariate-role normalization logic from the local xreg forecast bundle, extract small internal workflow helpers without changing public behavior.

**Step 4: Run focused tests again**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tests/test_data_workflows.py tests/test_root_import.py
git commit -m "feat: add local xreg eval bundle"
```

### Task 4: Sync API Metadata and Verify

**Files:**
- Modify: `tools/generate_model_capability_docs.py`
- Regenerate: `docs/api.md`
- Regenerate: `docs/models.md`

**Step 1: Update API metadata**

Add metadata for:

- `make_local_xreg_eval_bundle`

**Step 2: Regenerate docs**

Run: `python tools/generate_model_capability_docs.py`
Expected: `Wrote: docs/models.md` and `Wrote: docs/api.md`

**Step 3: Run verification**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py tests/test_docs_rnn_generated.py tests/test_eval_local_xreg.py -q`
Expected: PASS

**Step 4: Run broader lint and regression checks**

Run: `ruff check src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tools/generate_model_capability_docs.py tests/test_data_workflows.py tests/test_root_import.py`

Run: `pytest -q tests/test_data_workflows.py tests/test_root_import.py tests/test_docs_rnn_generated.py tests/test_forecast_api.py tests/test_eval_local_xreg.py`

Expected: PASS

**Step 5: Commit**

```bash
git add tools/generate_model_capability_docs.py docs/api.md docs/models.md
git commit -m "docs: sync local xreg eval bundle metadata"
```
