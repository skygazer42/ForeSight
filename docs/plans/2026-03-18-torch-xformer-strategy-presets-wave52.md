# Torch xFormer Strategy Presets Wave52 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a paired batch of local and global Torch xFormer preset model keys that package shared trainer strategies into ready-to-train attention recipes.

**Architecture:** Reuse the existing xFormer local and global catalog specs instead of changing any xFormer trainer internals. Clone selected baseline xFormer specs into new strategy-focused preset keys whose defaults enable EMA, SWA, SAM, regularized dropout, long-horizon weighting, and Lookahead. Lock the additions with dedicated registry, optional-dependency, and smoke tests.

**Tech Stack:** Python, pytest, existing model catalog registry.

---

### Task 1: Write failing coverage for xFormer strategy presets

**Files:**
- Create: `tests/test_models_wave52_xformer_strategy_presets_registry.py`
- Create: `tests/test_models_wave52_xformer_strategy_presets_optional_deps.py`
- Create: `tests/test_models_wave52_xformer_strategy_presets_smoke.py`

**Step 1: Write the failing test**

Add a dedicated registry test file listing the new preset keys and asserting their strategy-specific defaults.

Add a dedicated optional-dependency test file that asserts the new keys raise `ImportError` when torch is unavailable for both local and global flows.

Add a smoke test file that instantiates the new local and global xFormer keys on small synthetic series and panel data, checking output shape and finiteness when torch is installed.

The presets for this wave are:
- `torch-xformer-full-ema-direct`
- `torch-xformer-performer-swa-direct`
- `torch-xformer-linformer-sam-direct`
- `torch-xformer-nystrom-regularized-direct`
- `torch-xformer-bigbird-longhorizon-direct`
- `torch-xformer-longformer-lookahead-direct`
- `torch-xformer-full-ema-global`
- `torch-xformer-performer-swa-global`
- `torch-xformer-linformer-sam-global`
- `torch-xformer-nystrom-regularized-global`
- `torch-xformer-bigbird-longhorizon-global`
- `torch-xformer-longformer-lookahead-global`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave52_xformer_strategy_presets_registry.py tests/test_models_wave52_xformer_strategy_presets_optional_deps.py tests/test_models_wave52_xformer_strategy_presets_smoke.py`

Expected: FAIL because the new xFormer preset keys are not registered yet.

### Task 2: Implement the xFormer catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Write minimal implementation**

Add a helper that clones existing local and global xFormer specs into new strategy-focused preset keys with merged `default_params`, reused factories, and copied param help.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave52_xformer_strategy_presets_registry.py tests/test_models_wave52_xformer_strategy_presets_optional_deps.py tests/test_models_wave52_xformer_strategy_presets_smoke.py`

Expected: PASS.

### Task 3: Verify adjacent xFormer slices

**Files:**
- Modify: none

**Step 1: Run focused verification**

Run:
- `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py tests/test_torch_global_validation_messages.py -k "xformer or validation"`
- `python -m py_compile src/foresight/models/catalog/torch_local.py tests/test_models_wave52_xformer_strategy_presets_registry.py tests/test_models_wave52_xformer_strategy_presets_optional_deps.py tests/test_models_wave52_xformer_strategy_presets_smoke.py`
- `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_wave52_xformer_strategy_presets_registry.py tests/test_models_wave52_xformer_strategy_presets_optional_deps.py tests/test_models_wave52_xformer_strategy_presets_smoke.py`

Expected: PASS.
