# Structured Sequence Block Workflows Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add structured sequence block workflow helpers that expose packed panel sequence bundles as explicit encoder-decoder style inputs and preserve chronological splitting support.

**Architecture:** Keep the public implementation in `src/foresight/data/workflows.py` and define the structured helpers as deterministic adapters over the existing packed sequence tensor workflows. Export the new helpers from `src/foresight/data/__init__.py` and `src/foresight/__init__.py`, then sync generated API metadata so docs and root imports remain aligned.

**Tech Stack:** Python 3.10, pandas, numpy, pytest

---

### Task 1: Add Structured Sequence Block Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

```python
def test_make_panel_sequence_blocks_matches_packed_sequence_values() -> None:
    packed = make_panel_sequence_tensors(...)
    blocks = make_panel_sequence_blocks(...)
    assert blocks["train"]["past_y"].shape == (...)
    assert np.allclose(blocks["train"]["target_y"], packed["train"]["y"])


def test_make_panel_sequence_blocks_predict_contains_norm_stats() -> None:
    blocks = make_panel_sequence_blocks(...)
    assert "target_mean" in blocks["predict"]
    assert "target_std" in blocks["predict"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because `make_panel_sequence_blocks` does not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing import or missing function

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add structured sequence block coverage"
```

### Task 2: Add Zero-Width and Split Tests

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

```python
def test_make_panel_sequence_blocks_keeps_zero_width_optional_blocks() -> None:
    blocks = make_panel_sequence_blocks(..., x_cols=(), add_time_features=False)
    assert blocks["train"]["past_x"].shape[-1] == 0
    assert blocks["train"]["past_time"].shape[-1] == 0


def test_split_panel_sequence_blocks_respects_per_series_order() -> None:
    parts = split_panel_sequence_blocks(bundle, valid_size=1, test_size=1)
    assert set(parts) == {"train", "valid", "test"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL because structured split helper and zero-width behavior do not exist yet

**Step 3: Write the minimal implementation**

Only add tests in this step.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL on missing helper or missing block fields

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add structured sequence block split coverage"
```

### Task 3: Add Root Export Tests

**Files:**
- Modify: `tests/test_root_import.py`

**Step 1: Write the failing tests**

```python
def test_root_import_exports_structured_sequence_block_workflows() -> None:
    assert hasattr(foresight, "make_panel_sequence_blocks")
    assert hasattr(foresight, "split_panel_sequence_blocks")
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
git commit -m "test: cover structured sequence block exports"
```

### Task 4: Implement Structured Sequence Block Adapters

**Files:**
- Modify: `src/foresight/data/workflows.py`
- Modify: `src/foresight/data/__init__.py`
- Modify: `src/foresight/__init__.py`

**Step 1: Write the minimal implementation**

Add:

```python
def make_panel_sequence_blocks(...): ...
def split_panel_sequence_blocks(...): ...
```

Implementation requirements:

- derive structured blocks from `make_panel_sequence_tensors(...)`
- preserve packed semantics for normalization, sample stepping, and cutoff handling
- expose stable zero-width `x/time` blocks instead of omitting keys
- define split behavior in terms of the packed split semantics or shared chronology logic

**Step 2: Run focused tests to verify they pass**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 3: Refactor only if needed**

If block slicing is repetitive, extract a small helper for splitting past/future slices by channel dimensions without changing public behavior.

**Step 4: Run focused tests again**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/data/__init__.py src/foresight/__init__.py tests/test_data_workflows.py tests/test_root_import.py
git commit -m "feat: add structured sequence block workflows"
```

### Task 5: Sync API Metadata and Verify

**Files:**
- Modify: `tools/generate_model_capability_docs.py`
- Regenerate: `docs/api.md`
- Regenerate: `docs/models.md`

**Step 1: Update API metadata**

Add metadata rows for:

- `make_panel_sequence_blocks`
- `split_panel_sequence_blocks`

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
git commit -m "docs: sync structured sequence block workflow metadata"
```
