# Torch Global RNN Strategy Presets Wave55 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a batch of Torch global RNN preset model keys that package shared trainer strategies into ready-to-train panel recurrent recipes.

**Architecture:** Reuse the existing global RNN catalog specs instead of changing any recurrent trainer internals. Clone selected baseline global RNN and encoder-only specs into new strategy-focused preset keys whose defaults enable EMA, SWA, SAM, stronger regularization, long-horizon weighting, and Lookahead. Lock the additions with dedicated registry, optional-dependency, and smoke tests.

**Tech Stack:** Python, pytest, existing model catalog registry.

---

### Task 1: Write failing coverage for global RNN strategy presets

**Files:**
- Create: `tests/test_models_wave55_global_rnn_strategy_presets_registry.py`
- Create: `tests/test_models_wave55_global_rnn_strategy_presets_optional_deps.py`
- Create: `tests/test_models_wave55_global_rnn_strategy_presets_smoke.py`

**Step 1: Write the failing test**

Add a dedicated registry test file listing the new preset keys and asserting their strategy-specific defaults.

Add a dedicated optional-dependency test file that asserts the new keys raise `ImportError` when torch is unavailable.

Add a smoke test file that instantiates the new global RNN keys on a small synthetic panel and checks output columns and finiteness when torch is installed.

The presets for this wave are:
- `torch-rnn-lstm-ema-global`
- `torch-rnn-gru-swa-global`
- `torch-rnn-lstm-sam-global`
- `torch-rnn-gru-regularized-global`
- `torch-rnn-encoder-longhorizon-global`
- `torch-rnn-encoder-lookahead-global`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave55_global_rnn_strategy_presets_registry.py tests/test_models_wave55_global_rnn_strategy_presets_optional_deps.py tests/test_models_wave55_global_rnn_strategy_presets_smoke.py`

Expected: FAIL because the new global RNN preset keys are not registered yet.

### Task 2: Implement the global RNN catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Write minimal implementation**

Add a helper that clones existing global RNN specs into new strategy-focused preset keys with merged `default_params`, reused factories, and copied param help.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave55_global_rnn_strategy_presets_registry.py tests/test_models_wave55_global_rnn_strategy_presets_optional_deps.py tests/test_models_wave55_global_rnn_strategy_presets_smoke.py`

Expected: PASS.

### Task 3: Verify adjacent global RNN slices

**Files:**
- Modify: none

**Step 1: Run focused verification**

Run:
- `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "rnn and global"`
- `python -m py_compile src/foresight/models/catalog/torch_local.py tests/test_models_wave55_global_rnn_strategy_presets_registry.py tests/test_models_wave55_global_rnn_strategy_presets_optional_deps.py tests/test_models_wave55_global_rnn_strategy_presets_smoke.py`
- `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_wave55_global_rnn_strategy_presets_registry.py tests/test_models_wave55_global_rnn_strategy_presets_optional_deps.py tests/test_models_wave55_global_rnn_strategy_presets_smoke.py`

Expected: PASS.
