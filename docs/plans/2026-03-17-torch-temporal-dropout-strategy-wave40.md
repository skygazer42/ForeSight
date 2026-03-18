# Torch Temporal Dropout Strategy Wave 40 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add temporal dropout as a shared Torch training strategy so ForeSight can randomly drop whole lag timesteps during training without affecting validation or inference behavior.

**Architecture:** Extend the shared Torch trainer config with a single `temporal_dropout` field and implement the masking logic once in `src/foresight/models/torch_nn.py`. Reuse that helper in the shared local/global loops and the custom seq2seq-style loops so all user-facing Torch trainers get the same behavior and SAM still reuses the same augmented batch for both passes.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for temporal-dropout config exposure, validation, registry metadata, and runtime acceptance

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_sonar_torch_rename_coverage_smoke.py`

**Step 1: Write the failing test**

- Extend shared Torch config coverage with:
  - `temporal_dropout`
- Add local/global validation cases for:
  - `temporal_dropout in [0, 1)`
- Add runtime smoke tests proving these entrypoints accept temporal-dropout controls:
  - `torch-mlp-direct`
  - `torch-timexer-global`
  - `torch-seq2seq-attn-lstm-direct`
  - `torch-rnnpaper-seq2seq-direct`
  - `torch-seq2seq-lstm-global`
- Extend registry/default/help assertions with:
  - `temporal_dropout == 0.0`
  - help text for the new field
- Add a helper-level behavioral test showing the dropout mask is shared across features within each timestep

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py -k temporal_dropout`

Expected: FAIL because the new temporal-dropout control is not implemented yet.

### Task 2: Implement shared temporal-dropout config and helper

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` and `TorchGlobalTrainConfig` with:
  - `temporal_dropout: float = 0.0`
- Add shared validation:
  - `temporal_dropout in [0, 1)`
- Add a shared helper that:
  - skips when `temporal_dropout == 0`
  - samples a mask over the time axis
  - applies the same timestep mask across feature channels for each sample
  - preserves activation scale with keep-prob normalization
- Reuse the same temporal-dropout batch for both SAM passes when SAM is enabled
- Extend runtime coercion/default/help metadata for `temporal_dropout`

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py -k temporal_dropout`

Expected: PASS

### Task 3: Thread temporal dropout through direct seq2seq-style entrypoints

**Files:**
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Thread `temporal_dropout` through the direct/public builders that construct `TorchTrainConfig(...)` or custom strategy configs
- Apply temporal dropout only to training inputs before the model forward pass
- Keep the strategy orthogonal to:
  - elementwise `input_dropout`
  - post-backward strategies like gradient noise, GC, and AGC
  - post-step strategies like EMA, SWA, and Lookahead

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_sonar_torch_rename_coverage_smoke.py -k temporal_dropout`

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
