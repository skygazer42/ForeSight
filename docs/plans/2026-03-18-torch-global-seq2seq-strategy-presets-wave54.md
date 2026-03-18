# Torch Global Seq2Seq Strategy Presets Wave54 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a batch of Torch global Seq2Seq preset model keys that package shared trainer strategies into ready-to-train panel encoder-decoder recipes.

**Architecture:** Reuse the existing global Seq2Seq catalog specs and clone selected baseline global keys into new strategy-focused presets. Each preset keeps the original forecaster factory and param help, but overrides default trainer settings to expose distinct recipes such as EMA, SWA, SAM, stronger regularization, long-horizon weighting, and Lookahead.

**Tech Stack:** Python, pytest, existing model catalog registry.

---

### Task 1: Write failing coverage for global Seq2Seq strategy presets

**Files:**
- Create: `tests/test_models_wave54_global_seq2seq_strategy_presets_registry.py`
- Create: `tests/test_models_wave54_global_seq2seq_strategy_presets_optional_deps.py`
- Create: `tests/test_models_wave54_global_seq2seq_strategy_presets_smoke.py`

**Step 1: Write the failing test**

Add a dedicated registry test file listing the new preset keys and asserting their strategy-specific defaults.

Add a dedicated optional-dependency test file that asserts the new keys raise `ImportError` when torch is unavailable.

Add a smoke test file that instantiates the new global Seq2Seq keys on a small synthetic panel and checks output columns and finiteness when torch is installed.

The presets for this wave are:
- `torch-seq2seq-lstm-ema-global`
- `torch-seq2seq-gru-swa-global`
- `torch-seq2seq-attn-lstm-sam-global`
- `torch-seq2seq-attn-gru-regularized-global`
- `torch-seq2seq-lstm-longhorizon-global`
- `torch-seq2seq-attn-lstm-lookahead-global`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave54_global_seq2seq_strategy_presets_registry.py tests/test_models_wave54_global_seq2seq_strategy_presets_optional_deps.py tests/test_models_wave54_global_seq2seq_strategy_presets_smoke.py`

Expected: FAIL because the new global Seq2Seq preset keys are not registered yet.

### Task 2: Implement the global Seq2Seq catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`

**Step 1: Write minimal implementation**

Add a helper that clones existing global Seq2Seq specs into new strategy-focused preset keys with merged `default_params`, reused factories, and copied param help.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave54_global_seq2seq_strategy_presets_registry.py tests/test_models_wave54_global_seq2seq_strategy_presets_optional_deps.py tests/test_models_wave54_global_seq2seq_strategy_presets_smoke.py`

Expected: PASS.

### Task 3: Verify adjacent global Seq2Seq slices

**Files:**
- Modify: none

**Step 1: Run focused verification**

Run:
- `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "seq2seq and global"`
- `python -m py_compile src/foresight/models/catalog/torch_global.py tests/test_models_wave54_global_seq2seq_strategy_presets_registry.py tests/test_models_wave54_global_seq2seq_strategy_presets_optional_deps.py tests/test_models_wave54_global_seq2seq_strategy_presets_smoke.py`
- `ruff check src/foresight/models/catalog/torch_global.py tests/test_models_wave54_global_seq2seq_strategy_presets_registry.py tests/test_models_wave54_global_seq2seq_strategy_presets_optional_deps.py tests/test_models_wave54_global_seq2seq_strategy_presets_smoke.py`

Expected: PASS.
