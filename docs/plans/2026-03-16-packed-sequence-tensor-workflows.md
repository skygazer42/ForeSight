# Packed Sequence Tensor Workflows Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add packed sequence tensor workflow helpers that turn long-format panel data into reusable sequence-model training bundles and chronological window splits.

**Architecture:** Keep the public workflow logic in `src/foresight/data/workflows.py` and mirror the stable packed-sequence semantics already used internally by `src/foresight/models/torch_global.py` without importing model-layer helpers. Export the new helpers from `src/foresight/data/__init__.py` and `src/foresight/__init__.py`, then sync generated API metadata so docs and root imports remain aligned.

**Tech Stack:** Python 3.10, pandas, numpy, pytest

---

### Task 1: Add Packed Sequence Bundle Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

```python
def test_make_panel_sequence_tensors_shapes_and_channel_order() -> None:
    bundle = make_panel_sequence_tensors(
        long_df,
        cutoff=pd.Timestamp("2020-01-06"),
        horizon=2,
        context_length=3,
        x_cols=("promo",),
        normalize=False,
        add_time_features=True,
    )
    assert bundle["train"]["X"].shape == (.., 5, ..)
    assert bundle["metadata"]["channel_names"][0] == "y"


def test_make_panel_sequence_tensors_predict_block_has_one_row_per_series() -> None:
    bundle = make_panel_sequence_tensors(...)
    assert bundle["predict"]["X"].shape[0] == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because `make_panel_sequence_tensors` does not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing import or missing function

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add packed sequence tensor bundle coverage"
```

### Task 2: Add Split and Validation Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

```python
def test_split_panel_sequence_tensors_respects_per_series_order() -> None:
    parts = split_panel_sequence_tensors(bundle, valid_size=1, test_size=1)
    assert set(parts) == {"train", "valid", "test"}


def test_make_panel_sequence_tensors_returns_target_norm_stats() -> None:
    bundle = make_panel_sequence_tensors(long_df, ..., normalize=True)
    assert "target_mean" in bundle["predict"]
    assert "target_std" in bundle["predict"]


def test_make_panel_sequence_tensors_rejects_duplicate_timestamps() -> None:
    with pytest.raises(ValueError, match="align_long_df"):
        make_panel_sequence_tensors(long_df_with_duplicates, ...)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because split helper and validations do not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing helpers or missing validations

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add packed sequence split and validation coverage"
```

### Task 3: Add Root Export Tests

**Files:**
- Modify: `tests/test_root_import.py`

**Step 1: Write the failing tests**

```python
def test_root_import_exports_packed_sequence_tensor_workflows() -> None:
    assert hasattr(foresight, "make_panel_sequence_tensors")
    assert hasattr(foresight, "split_panel_sequence_tensors")
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
git commit -m "test: cover packed sequence workflow exports"
```

### Task 4: Implement Packed Sequence Workflow Helpers

**Files:**
- Modify: `src/foresight/data/workflows.py`
- Modify: `src/foresight/data/__init__.py`
- Modify: `src/foresight/__init__.py`

**Step 1: Write the minimal implementation**

Add shared internals and public APIs:

```python
def make_panel_sequence_tensors(...): ...
def split_panel_sequence_tensors(...): ...
```

Implementation requirements:

- long-format validation with duplicate timestamp checks per series
- packed training tensor output `(n_samples, context_length + horizon, input_dim)`
- packed prediction tensor output with one row per eligible series
- per-series target normalization only
- stable `channel_names`, `time_feature_names`, and index frames
- chronological bundle splitting based on `window_index`

**Step 2: Run focused tests to verify they pass**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 3: Refactor only if needed**

If training and prediction packing duplicate too much logic, extract a tiny internal sequence-packing helper without changing public behavior.

**Step 4: Run focused tests again**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tests/test_data_workflows.py tests/test_root_import.py
git commit -m "feat: add packed sequence tensor workflows"
```

### Task 5: Sync API Metadata and Verify

**Files:**
- Modify: `tools/generate_model_capability_docs.py`
- Regenerate: `docs/api.md`
- Regenerate: `docs/models.md`

**Step 1: Update API metadata**

Add metadata rows for:

- `make_panel_sequence_tensors`
- `split_panel_sequence_tensors`

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
git commit -m "docs: sync packed sequence workflow metadata"
```
