# Torch Training Strategy Presets Wave49 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a batch of Torch local preset model keys that package existing trainer strategies into ready-to-train forecasting recipes.

**Architecture:** Reuse existing local Torch model specs instead of changing trainer internals. Clone selected baseline specs into new preset keys whose defaults enable distinct strategy stacks such as EMA, SWA, SAM, Lookahead, stronger regularization, and long-horizon weighting. Lock the additions with registry default assertions and optional-dependency coverage tests.

**Tech Stack:** Python, pytest, existing model catalog registry.

---

### Task 1: Write failing coverage for strategy presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing test**

Add a new tuple of expected local preset keys in `tests/test_models_optional_deps_torch.py` and a coverage test that asserts those keys are present in the Torch local registry.

Add a registry test in `tests/test_models_registry.py` that fetches each new preset key and asserts a few strategy-specific default params, such as `ema_decay`, `swa_start_epoch`, `sam_rho`, `lookahead_steps`, `input_dropout`, and `horizon_loss_decay`.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py -k training_strategy tests/test_models_registry.py -k wave30`

Expected: FAIL because the new preset keys are not registered yet.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Write minimal implementation**

Add a helper that clones existing local Torch specs into new strategy-focused preset keys with merged `default_params` and reused factories/param help. Register these presets in `build_torch_local_catalog(...)`.

The presets for this wave are:
- `torch-patchtst-ema-direct`
- `torch-timesnet-swa-direct`
- `torch-timexer-sam-direct`
- `torch-tsmixer-regularized-direct`
- `torch-tft-longhorizon-direct`
- `torch-nbeats-lookahead-direct`

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py -k training_strategy tests/test_models_registry.py -k wave30`

Expected: PASS.

### Task 3: Verify targeted Torch catalog integrity

**Files:**
- Modify: none

**Step 1: Run focused verification**

Run:
- `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py -k \"training_strategy or optional_dep_paths\"`
- `PYTHONPATH=src pytest -q tests/test_models_registry.py -k \"wave30 or wave29 or wave28\"`
- `python -m py_compile src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`
- `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: PASS.
