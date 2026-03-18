# Torch Trainer Config Wave 4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add shared Torch trainer checkpoint persistence so local/global/seq2seq Torch models can optionally save best and last checkpoints to disk during training.

**Architecture:** Extend the shared `TorchTrainConfig` surface in `src/foresight/models/torch_nn.py` with minimal checkpoint controls, then implement reusable save helpers around the existing in-memory `best_state` / final-state flow. Reuse that logic from `torch_global.py` and `torch_seq2seq.py`, and thread the new args through every Torch entrypoint plus `runtime.py` metadata so registry-created forecasters accept the same config everywhere.

**Tech Stack:** Python, PyTorch, pathlib, pytest, ruff

---

### Task 1: Add failing tests for checkpoint config and artifact creation

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing test**

- Add local validation coverage for checkpoint save flags without `checkpoint_dir`.
- Add a local positive test that trains a tiny Torch model and asserts `best.pt` / `last.pt` are written.
- Add a matching global positive test for a small global Torch model.
- Extend registry assertions to cover the new default params and help text.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: FAIL because checkpoint config is not implemented yet.

### Task 2: Implement shared checkpoint persistence

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_seq2seq.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` and `TorchGlobalTrainConfig` with:
  - `checkpoint_dir`
  - `save_best_checkpoint`
  - `save_last_checkpoint`
- Add shared validation enforcing that checkpoint saving requires a non-empty directory.
- Add shared helpers that:
  - clone state dicts to CPU
  - create checkpoint directories
  - write `best.pt` and `last.pt`
- Save best checkpoint from the tracked best state and last checkpoint from the final pre-restore state.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

### Task 3: Thread checkpoint args through all Torch entrypoints and runtime metadata

**Files:**
- Modify: `src/foresight/models/runtime.py`
- Modify: Torch entrypoint modules that construct `TorchTrainConfig`, including:
  - `src/foresight/models/torch_nn.py`
  - `src/foresight/models/torch_global.py`
  - `src/foresight/models/torch_seq2seq.py`
  - `src/foresight/models/torch_ct_rnn.py`
  - `src/foresight/models/torch_graph_attention.py`
  - `src/foresight/models/torch_graph_spectral.py`
  - `src/foresight/models/torch_graph_structure.py`
  - `src/foresight/models/torch_probabilistic.py`
  - `src/foresight/models/torch_reservoir.py`
  - `src/foresight/models/torch_rnn_paper_zoo.py`
  - `src/foresight/models/torch_rnn_zoo.py`
  - `src/foresight/models/torch_ssm.py`
  - `src/foresight/models/torch_structured_rnn.py`
  - `src/foresight/models/torch_xformer.py`
  - `src/foresight/models/multivariate.py`

**Step 1: Write minimal implementation**

- Extend `_coerce_torch_extra_train_params`, `_TORCH_COMMON_DEFAULTS`, and `_TORCH_COMMON_PARAM_HELP`.
- Thread the new checkpoint args into every `TorchTrainConfig(...)` / builder call so registry-created forecasters accept them consistently.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py`

Expected: PASS

### Task 4: Run targeted verification and lint

**Files:**
- Verify: `tests/test_models_optional_deps_torch.py`
- Verify: `tests/test_torch_nn_validation_messages.py`
- Verify: `tests/test_torch_global_validation_messages.py`
- Verify: `tests/test_models_registry.py`

**Step 1: Run targeted verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

**Step 2: Run lint on changed files**

Run: `ruff check src/foresight/models/torch_nn.py src/foresight/models/torch_global.py src/foresight/models/torch_seq2seq.py src/foresight/models/runtime.py src/foresight/models/torch_ct_rnn.py src/foresight/models/torch_graph_attention.py src/foresight/models/torch_graph_spectral.py src/foresight/models/torch_graph_structure.py src/foresight/models/torch_probabilistic.py src/foresight/models/torch_reservoir.py src/foresight/models/torch_rnn_paper_zoo.py src/foresight/models/torch_rnn_zoo.py src/foresight/models/torch_ssm.py src/foresight/models/torch_structured_rnn.py src/foresight/models/torch_xformer.py src/foresight/models/multivariate.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS
