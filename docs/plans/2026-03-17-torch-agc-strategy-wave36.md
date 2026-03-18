# Torch AGC Strategy Wave 36 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Adaptive Gradient Clipping (AGC) as an orthogonal Torch training strategy across ForeSight local/global/seq2seq trainers so existing Torch forecasters can train with per-parameter adaptive clipping without changing the optimizer API.

**Architecture:** Extend the shared Torch trainer config in `src/foresight/models/torch_nn.py` with AGC controls and validate them alongside the existing trainer strategy fields. Implement AGC as shared gradient helpers that run before the existing norm/value clipping path, then reuse those helpers in shared local/global loops and the custom seq2seq-style loops so behavior stays consistent across local, global, seq2seq, and rnnpaper Torch training paths.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for AGC validation, runtime exposure, and registry metadata

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_sonar_torch_rename_coverage_smoke.py`

**Step 1: Write the failing test**

- Extend shared Torch config coverage with:
  - `agc_clip_factor`
  - `agc_eps`
- Add local/global validation cases for:
  - `agc_clip_factor >= 0`
  - `agc_eps > 0`
- Add runtime smoke tests proving these entrypoints accept AGC controls:
  - `torch-mlp-direct`
  - `torch-timexer-global`
  - `torch-seq2seq-attn-lstm-direct`
  - `torch-rnnpaper-seq2seq-direct`
- Extend registry/default/help assertions with:
  - `agc_clip_factor == 0.0`
  - `agc_eps == 1e-3`
  - help text for both fields

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py -k agc`

Expected: FAIL because AGC fields and training behavior are not implemented yet.

### Task 2: Implement shared AGC config and gradient helpers

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` and `TorchGlobalTrainConfig` with:
  - `agc_clip_factor: float = 0.0`
  - `agc_eps: float = 1e-3`
- Add shared validation for:
  - `agc_clip_factor >= 0`
  - `agc_eps > 0`
- Add shared AGC helpers for:
  - detecting whether AGC is enabled
  - computing parameter/gradient unitwise norms
  - clipping gradients relative to parameter norms before existing clipping
- Update `_apply_torch_gradient_clipping(...)` so AGC runs first, then current norm/value clipping remains available
- Extend runtime coercion/default/help metadata for the two new AGC controls

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py -k agc`

Expected: PASS

### Task 3: Integrate AGC into the remaining custom seq2seq-style training loops

**Files:**
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Reuse `_apply_torch_gradient_clipping(...)` in the custom seq2seq-style loops where direct norm clipping is still inlined.
- Thread `agc_clip_factor` and `agc_eps` through each public config builder / wrapper that forwards shared Torch trainer config.
- Keep AGC orthogonal to existing strategies:
  - SAM changes the optimizer-step procedure
  - AGC adjusts gradients before the final step
  - EMA/SWA/Lookahead remain post-step strategy layers

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_sonar_torch_rename_coverage_smoke.py -k agc`

Expected: PASS

### Task 4: Run targeted verification

**Files:**
- Verify: `tests/test_torch_nn_validation_messages.py`
- Verify: `tests/test_torch_global_validation_messages.py`
- Verify: `tests/test_models_registry.py`
- Verify: `tests/test_sonar_torch_rename_coverage_smoke.py`

**Step 1: Run focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py`

Expected: PASS

**Step 2: Run static verification**

Run: `python -m py_compile src/foresight/models/torch_nn.py src/foresight/models/runtime.py src/foresight/models/torch_global.py src/foresight/models/torch_seq2seq.py src/foresight/models/torch_rnn_paper_zoo.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py`

Expected: PASS

**Step 3: Run lint**

Run: `ruff check src/foresight/models/torch_nn.py src/foresight/models/runtime.py src/foresight/models/torch_global.py src/foresight/models/torch_seq2seq.py src/foresight/models/torch_rnn_paper_zoo.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py`

Expected: PASS
