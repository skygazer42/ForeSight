# LongDF Splits and Scalers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add long-format panel split and reversible scaler helpers that complete the current training-oriented data workflow layer.

**Architecture:** Keep all new behavior in `src/foresight/data/workflows.py` and reuse the package's existing dataframe-first conventions. Export the helpers from `src/foresight/data/__init__.py` and `src/foresight/__init__.py`, then update generated API metadata so root exports and docs stay in sync.

**Tech Stack:** Python 3.10, pandas, numpy, pytest

---

### Task 1: Add Split Helper Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

```python
def test_split_long_df_sizes_respects_per_series_order() -> None:
    parts = split_long_df(long_df, valid_size=1, test_size=1)
    assert set(parts) == {"train", "valid", "test"}


def test_split_long_df_gap_reserves_rows_between_train_and_test() -> None:
    parts = split_long_df(long_df, test_size=1, gap=1)
    assert ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because `split_long_df` does not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing import or missing function

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add long df split workflow coverage"
```

### Task 2: Add Scaler Round-Trip Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

```python
def test_per_series_scaler_round_trip_restores_original_values() -> None:
    scaler = fit_long_df_scaler(long_df, method="standard", scope="per_series")
    scaled = transform_long_df_with_scaler(long_df, scaler)
    restored = inverse_transform_long_df_with_scaler(scaled, scaler)
    assert ...


def test_global_scaler_uses_single_stats_row_per_column() -> None:
    scaler = fit_long_df_scaler(long_df, method="maxabs", scope="global")
    assert ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because scaler helpers do not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing scaler helpers

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add long df scaler workflow coverage"
```

### Task 3: Add Root Export and Docs Sync Tests

**Files:**
- Modify: `tests/test_root_import.py`

**Step 1: Write the failing tests**

```python
def test_root_import_exports_longdf_split_and_scaler_helpers() -> None:
    assert hasattr(foresight, "split_long_df")
    assert hasattr(foresight, "fit_long_df_scaler")
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
git commit -m "test: cover root workflow exports"
```

### Task 4: Implement Split and Scaler Helpers

**Files:**
- Modify: `src/foresight/data/workflows.py`
- Modify: `src/foresight/data/__init__.py`
- Modify: `src/foresight/__init__.py`

**Step 1: Write the minimal implementation**

Add:

```python
def split_long_df(...): ...
def fit_long_df_scaler(...): ...
def transform_long_df_with_scaler(...): ...
def inverse_transform_long_df_with_scaler(...): ...
```

Implementation requirements:

- per-series chronological splitting with `valid/test` sizes or fractions
- optional `gap`
- reversible stats-table-based scaling
- `standard`, `minmax`, and `maxabs` methods
- `per_series` and `global` scopes

**Step 2: Run focused tests to verify they pass**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 3: Refactor only if needed**

If the scaler logic duplicates forward/inverse arithmetic, extract tiny helpers without changing behavior.

**Step 4: Run focused tests again**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tests/test_data_workflows.py tests/test_root_import.py
git commit -m "feat: add long df split and scaler workflows"
```

### Task 5: Sync API Metadata and Verify

**Files:**
- Modify: `tools/generate_model_capability_docs.py`
- Regenerate: `docs/api.md`
- Regenerate: `docs/models.md`

**Step 1: Update API metadata**

Add metadata rows for:

- `split_long_df`
- `fit_long_df_scaler`
- `transform_long_df_with_scaler`
- `inverse_transform_long_df_with_scaler`

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
git commit -m "docs: sync workflow export metadata"
```
