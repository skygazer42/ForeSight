# Torch Global Training Strategy Presets Wave31 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a batch of Torch global preset model keys that package existing trainer strategies into ready-to-train panel forecasting recipes.

**Architecture:** Reuse the existing Torch global model specs and clone selected baseline global keys into new strategy-focused presets. Each preset keeps the original forecaster factory and param help, but overrides default trainer settings to expose distinct recipes such as EMA, SWA, SAM, stronger regularization, long-horizon weighting, and Lookahead.

**Tech Stack:** Python, pytest, existing model catalog registry.

---

### Task 1: Write failing coverage for global strategy presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing test**

Add a new tuple of expected global preset keys in `tests/test_models_optional_deps_torch.py` and a coverage test that asserts those keys are present in the Torch global registry.

Add a registry test in `tests/test_models_registry.py` that fetches each new preset key and asserts strategy-specific defaults such as `ema_decay`, `swa_start_epoch`, `sam_rho`, `input_dropout`, `horizon_loss_decay`, and `lookahead_steps`.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_training_strategy_torch_global_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_torch_global_catalog_exposes_wave31_training_strategy_preset_defaults`

Expected: FAIL because the new global preset keys are not registered yet.

### Task 2: Implement the global catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`

**Step 1: Write minimal implementation**

Wrap the existing global catalog dict in a local `catalog` variable and add a helper that clones existing global specs into new strategy-focused preset keys with merged `default_params`.

The presets for this wave are:
- `torch-patchtst-ema-global`
- `torch-timesnet-swa-global`
- `torch-timexer-sam-global`
- `torch-tsmixer-regularized-global`
- `torch-tft-longhorizon-global`
- `torch-seq2seq-attn-gru-lookahead-global`

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_training_strategy_torch_global_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_torch_global_catalog_exposes_wave31_training_strategy_preset_defaults`

Expected: PASS.

### Task 3: Verify adjacent registry slices

**Files:**
- Modify: none

**Step 1: Run focused verification**

Run:
- `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py -k "training_strategy or global_preset_torch_models_are_covered_by_optional_dep_paths"`
- `PYTHONPATH=src pytest -q tests/test_models_registry.py -k "wave31 or wave30 or wave29 or wave28"`
- `python -m py_compile src/foresight/models/catalog/torch_local.py src/foresight/models/catalog/torch_global.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`
- `ruff check src/foresight/models/catalog/torch_local.py src/foresight/models/catalog/torch_global.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: PASS.
