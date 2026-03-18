# Torch Input Dropout Strategy Wave 39 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add input dropout as another orthogonal Torch training strategy so ForeSight local/global/seq2seq trainers can optionally apply stochastic feature dropout to training inputs without changing inference behavior.

**Architecture:** Extend the shared Torch trainer config with a single `input_dropout` field instead of adding model-specific knobs. Implement input dropout as a shared training-input helper in `src/foresight/models/torch_nn.py`, then reuse that helper in the shared local/global loops and the custom seq2seq-style loops so behavior stays consistent across user-facing Torch models.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for input-dropout config exposure, validation, runtime acceptance, and registry metadata

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_sonar_torch_rename_coverage_smoke.py`

**Step 1: Write the failing test**

- Extend shared Torch config coverage with:
  - `input_dropout`
- Add local/global validation cases for:
  - `input_dropout in [0, 1)`
- Add runtime smoke tests proving these entrypoints accept input-dropout controls:
  - `torch-mlp-direct`
  - `torch-timexer-global`
  - `torch-seq2seq-attn-lstm-direct`
  - `torch-rnnpaper-seq2seq-direct`
  - `torch-seq2seq-lstm-global`
- Extend registry/default/help assertions with:
  - `input_dropout == 0.0`
  - help text for the new field

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py -k input_dropout`

Expected: FAIL because the new input-dropout control is not implemented yet.

### Task 2: Implement shared input-dropout config and helper

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` and `TorchGlobalTrainConfig` with:
  - `input_dropout: float = 0.0`
- Add shared validation:
  - `input_dropout in [0, 1)`
- Add a shared helper that:
  - skips when `input_dropout == 0`
  - applies feature dropout to the training input tensor only
- Reuse the same dropout-applied batch for both SAM passes when SAM is enabled
- Extend runtime coercion/default/help metadata for `input_dropout`

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py -k input_dropout`

Expected: PASS

### Task 3: Thread input dropout through direct seq2seq-style entrypoints

**Files:**
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Thread `input_dropout` through the direct/public builders that construct `TorchTrainConfig(...)` or custom strategy configs.
- Keep input dropout orthogonal to the existing strategy stack:
  - input dropout perturbs training inputs before the forward pass
  - gradient strategies still act after backward
  - EMA / SWA / Lookahead remain post-step layers

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_sonar_torch_rename_coverage_smoke.py -k input_dropout`

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
