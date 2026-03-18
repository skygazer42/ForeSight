# Torch SAM Strategy Wave 35 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Sharpness-Aware Minimization (SAM) training strategy across ForeSight Torch trainers so local, global, seq2seq, and rnnpaper Torch models can train with SAM while keeping the base optimizer API (`adam` / `adamw` / `sgd`) unchanged.

**Architecture:** Extend the shared Torch trainer config with orthogonal SAM controls instead of adding a new optimizer enum. Implement SAM as shared training helpers in `src/foresight/models/torch_nn.py` that perturb weights using the current gradient norm, run the second forward/backward pass, then restore parameters and apply the base optimizer step. Reuse those helpers in the shared local/global loops and the custom seq2seq-style loops so behavior stays consistent across user-facing Torch models.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for SAM config exposure, validation, and runtime acceptance

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_sonar_torch_rename_coverage_smoke.py`

**Step 1: Write the failing test**

- Extend `TorchTrainConfig` coverage with:
  - `sam_rho`
  - `sam_adaptive`
- Add local/global validation cases for:
  - `sam_rho >= 0`
- Add runtime smoke tests proving these entrypoints accept SAM controls:
  - `torch-mlp-direct`
  - `torch-timexer-global`
  - `torch-seq2seq-attn-lstm-direct`
  - `torch-rnnpaper-seq2seq-direct`
  - `torch-seq2seq-lstm-global`
- Extend registry/default/help assertions with:
  - `sam_rho == 0.0`
  - `sam_adaptive is False`
  - help text for both fields

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py -k sam`

Expected: FAIL because SAM fields and runtime behavior are not implemented yet.

### Task 2: Implement shared SAM helpers and integrate shared trainers

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` and `TorchGlobalTrainConfig` with:
  - `sam_rho: float = 0.0`
  - `sam_adaptive: bool = False`
- Add shared validation for `sam_rho >= 0`
- Add shared SAM helpers for:
  - detecting SAM enablement
  - computing gradient norm
  - perturbing model weights before the second pass
  - restoring weights and applying the base optimizer step
- Integrate those helpers into:
  - `src/foresight/models/torch_nn.py::_train_loop`
  - `src/foresight/models/torch_global.py::_train_loop_global`
- Extend runtime coercion/default/help metadata for the two new SAM controls

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py -k sam`

Expected: PASS

### Task 3: Integrate SAM into custom seq2seq-style Torch loops

**Files:**
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Reuse the shared SAM helpers inside:
  - `torch_seq2seq.py::_train_seq2seq`
  - `torch_rnn_paper_zoo.py` custom seq2seq training path
  - `torch_global.py` custom global seq2seq training path
- Keep the strategy orthogonal to EMA/SWA/Lookahead:
  - SAM changes the optimizer-step procedure only
  - EMA/SWA/Lookahead continue to run after the base step
- Thread `sam_rho` and `sam_adaptive` through each custom strategy config / `TorchTrainConfig(...)` builder

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_sonar_torch_rename_coverage_smoke.py -k sam`

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
