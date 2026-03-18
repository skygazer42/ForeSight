# Torch Seq2Seq Strategy Presets Wave53 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a batch of local Torch Seq2Seq preset model keys that package shared trainer strategies into ready-to-train encoder-decoder recipes.

**Architecture:** Reuse the existing local Seq2Seq catalog specs instead of changing any Seq2Seq trainer internals. Clone selected baseline Seq2Seq specs into new strategy-focused preset keys whose defaults enable EMA, SWA, SAM, regularized dropout, long-horizon weighting, and Lookahead. Lock the additions with dedicated registry, optional-dependency, and smoke tests.

**Tech Stack:** Python, pytest, existing model catalog registry.

---

### Task 1: Write failing coverage for local Seq2Seq strategy presets

**Files:**
- Create: `tests/test_models_wave53_seq2seq_strategy_presets_registry.py`
- Create: `tests/test_models_wave53_seq2seq_strategy_presets_optional_deps.py`
- Create: `tests/test_models_wave53_seq2seq_strategy_presets_smoke.py`

**Step 1: Write the failing test**

Add a dedicated registry test file listing the new preset keys and asserting their strategy-specific defaults.

Add a dedicated optional-dependency test file that asserts the new keys raise `ImportError` when torch is unavailable.

Add a smoke test file that instantiates the new local Seq2Seq keys on a small synthetic series and checks output shape and finiteness when torch is installed.

The presets for this wave are:
- `torch-seq2seq-lstm-ema-direct`
- `torch-seq2seq-gru-swa-direct`
- `torch-seq2seq-attn-lstm-sam-direct`
- `torch-seq2seq-attn-gru-regularized-direct`
- `torch-seq2seq-lstm-longhorizon-direct`
- `torch-seq2seq-attn-gru-lookahead-direct`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave53_seq2seq_strategy_presets_registry.py tests/test_models_wave53_seq2seq_strategy_presets_optional_deps.py tests/test_models_wave53_seq2seq_strategy_presets_smoke.py`

Expected: FAIL because the new Seq2Seq preset keys are not registered yet.

### Task 2: Implement the local Seq2Seq catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Write minimal implementation**

Add a helper that clones existing local Seq2Seq specs into new strategy-focused preset keys with merged `default_params`, reused factories, and copied param help.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave53_seq2seq_strategy_presets_registry.py tests/test_models_wave53_seq2seq_strategy_presets_optional_deps.py tests/test_models_wave53_seq2seq_strategy_presets_smoke.py`

Expected: PASS.

### Task 3: Verify adjacent Seq2Seq slices

**Files:**
- Modify: none

**Step 1: Run focused verification**

Run:
- `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "seq2seq"`
- `python -m py_compile src/foresight/models/catalog/torch_local.py tests/test_models_wave53_seq2seq_strategy_presets_registry.py tests/test_models_wave53_seq2seq_strategy_presets_optional_deps.py tests/test_models_wave53_seq2seq_strategy_presets_smoke.py`
- `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_wave53_seq2seq_strategy_presets_registry.py tests/test_models_wave53_seq2seq_strategy_presets_optional_deps.py tests/test_models_wave53_seq2seq_strategy_presets_smoke.py`

Expected: PASS.
