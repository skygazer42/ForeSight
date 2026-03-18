# Torch Multivariate Strategy Presets Wave50 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a batch of multivariate Torch preset model keys that package existing trainer strategies into ready-to-train graph forecasting recipes.

**Architecture:** Reuse the existing multivariate Torch model specs instead of changing multivariate trainer internals. Clone selected baseline multivariate specs into new preset keys whose defaults enable distinct strategy stacks such as EMA, SWA, SAM, regularized dropout, long-horizon weighting, and Lookahead. Lock the additions with dedicated registry, optional-dependency, and smoke tests.

**Tech Stack:** Python, pytest, existing model catalog registry.

---

### Task 1: Write failing coverage for multivariate strategy presets

**Files:**
- Create: `tests/test_models_wave50_multivariate_strategy_presets_registry.py`
- Create: `tests/test_models_wave50_multivariate_strategy_presets_optional_deps.py`
- Create: `tests/test_models_multivariate_strategy_presets_smoke.py`

**Step 1: Write the failing test**

Add a dedicated registry test file listing the new preset keys and asserting their strategy-specific defaults.

Add a dedicated optional-dependency test file that asserts the new keys raise `ImportError` when torch is unavailable.

Add a smoke test file that instantiates the new multivariate keys on a small wide matrix and checks the output shape and finiteness when torch is installed.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave50_multivariate_strategy_presets_registry.py tests/test_models_wave50_multivariate_strategy_presets_optional_deps.py tests/test_models_multivariate_strategy_presets_smoke.py`

Expected: FAIL because the new multivariate preset keys are not registered yet.

### Task 2: Implement the multivariate catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/multivariate.py`

**Step 1: Write minimal implementation**

Add a helper that clones existing multivariate Torch specs into new strategy-focused preset keys with merged `default_params`, reused factories, and copied param help.

The presets for this wave are:
- `torch-stid-ema-multivariate`
- `torch-stgcn-swa-multivariate`
- `torch-graphwavenet-sam-multivariate`
- `torch-astgcn-regularized-multivariate`
- `torch-agcrn-longhorizon-multivariate`
- `torch-stemgnn-lookahead-multivariate`

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave50_multivariate_strategy_presets_registry.py tests/test_models_wave50_multivariate_strategy_presets_optional_deps.py tests/test_models_multivariate_strategy_presets_smoke.py`

Expected: PASS.

### Task 3: Verify adjacent multivariate slices

**Files:**
- Modify: none

**Step 1: Run focused verification**

Run:
- `PYTHONPATH=src pytest -q tests/test_models_multivariate.py tests/test_models_graph_attention_smoke.py tests/test_models_graph_structure_smoke.py tests/test_models_graph_spectral_smoke.py`
- `python -m py_compile src/foresight/models/catalog/multivariate.py tests/test_models_wave50_multivariate_strategy_presets_registry.py tests/test_models_wave50_multivariate_strategy_presets_optional_deps.py tests/test_models_multivariate_strategy_presets_smoke.py`
- `ruff check src/foresight/models/catalog/multivariate.py tests/test_models_wave50_multivariate_strategy_presets_registry.py tests/test_models_wave50_multivariate_strategy_presets_optional_deps.py tests/test_models_multivariate_strategy_presets_smoke.py`

Expected: PASS.
