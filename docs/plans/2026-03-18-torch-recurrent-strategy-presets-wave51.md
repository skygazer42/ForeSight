# Torch Recurrent Strategy Presets Wave51 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a batch of recurrent Torch preset model keys that package existing trainer strategies into ready-to-train `rnnpaper` and `rnnzoo` recipes.

**Architecture:** Reuse the existing local recurrent catalog specs instead of changing recurrent trainer internals. Clone selected `rnnpaper` and `rnnzoo` baseline specs into new strategy-focused preset keys whose defaults enable distinct strategy stacks such as EMA, SWA, SAM, regularized dropout, long-horizon weighting, and Lookahead. Lock the additions with dedicated registry, optional-dependency, and smoke tests.

**Tech Stack:** Python, pytest, existing model catalog registry.

---

### Task 1: Write failing coverage for recurrent strategy presets

**Files:**
- Create: `tests/test_models_wave51_recurrent_strategy_presets_registry.py`
- Create: `tests/test_models_wave51_recurrent_strategy_presets_optional_deps.py`
- Create: `tests/test_models_wave51_recurrent_strategy_presets_smoke.py`

**Step 1: Write the failing test**

Add a dedicated registry test file listing the new preset keys and asserting their strategy-specific defaults.

Add a dedicated optional-dependency test file that asserts the new keys raise `ImportError` when torch is unavailable.

Add a smoke test file that instantiates the new recurrent keys on a small univariate series and checks the output shape and finiteness when torch is installed.

The presets for this wave are:
- `torch-rnnpaper-lstm-ema-direct`
- `torch-rnnpaper-gru-swa-direct`
- `torch-rnnpaper-qrnn-lookahead-direct`
- `torch-rnnzoo-lstm-sam-direct`
- `torch-rnnzoo-gru-regularized-direct`
- `torch-rnnzoo-qrnn-longhorizon-direct`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave51_recurrent_strategy_presets_registry.py tests/test_models_wave51_recurrent_strategy_presets_optional_deps.py tests/test_models_wave51_recurrent_strategy_presets_smoke.py`

Expected: FAIL because the new recurrent preset keys are not registered yet.

### Task 2: Implement the recurrent catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Write minimal implementation**

Add a helper that clones existing `rnnpaper` and `rnnzoo` specs into new strategy-focused preset keys with merged `default_params`, reused factories, and copied param help.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave51_recurrent_strategy_presets_registry.py tests/test_models_wave51_recurrent_strategy_presets_optional_deps.py tests/test_models_wave51_recurrent_strategy_presets_smoke.py`

Expected: PASS.

### Task 3: Verify adjacent recurrent slices

**Files:**
- Modify: none

**Step 1: Run focused verification**

Run:
- `PYTHONPATH=src pytest -q tests/test_models_rnnpaper_100.py tests/test_models_rnn_zoo_100.py -k "registered or optional or smoke or validation"`
- `python -m py_compile src/foresight/models/catalog/torch_local.py tests/test_models_wave51_recurrent_strategy_presets_registry.py tests/test_models_wave51_recurrent_strategy_presets_optional_deps.py tests/test_models_wave51_recurrent_strategy_presets_smoke.py`
- `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_wave51_recurrent_strategy_presets_registry.py tests/test_models_wave51_recurrent_strategy_presets_optional_deps.py tests/test_models_wave51_recurrent_strategy_presets_smoke.py`

Expected: PASS.
