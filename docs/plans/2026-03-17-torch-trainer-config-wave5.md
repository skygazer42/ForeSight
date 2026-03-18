# Torch Trainer Config Wave 5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add shared Torch trainer warm-start support so local/global/seq2seq Torch models can load model weights from a saved checkpoint or raw `state_dict` before training.

**Architecture:** Extend the shared `TorchTrainConfig` surface in `src/foresight/models/torch_nn.py` with a checkpoint load path and strictness flag, then add a reusable loader that accepts either this package's saved checkpoint payload (`{"state_dict": ...}`) or a raw PyTorch `state_dict`. Reuse the helper in local/global/seq2seq loops and expose the new controls through `src/foresight/models/runtime.py`.

**Tech Stack:** Python, PyTorch, pathlib, pytest, ruff

---

### Task 1: Add failing tests for checkpoint warm-start

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing test**

- Add shared config field assertions for:
  - `resume_checkpoint_path`
  - `resume_checkpoint_strict`
- Add local/global validation coverage for a missing resume checkpoint path.
- Add a unit test that:
  - saves a tiny model checkpoint
  - loads it through a shared helper
  - asserts the target model weights are restored
- Extend registry assertions to cover defaults/help text for the two new params.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: FAIL because resume-from-checkpoint is not implemented yet.

### Task 2: Implement shared checkpoint loading

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_seq2seq.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` / `TorchGlobalTrainConfig` with:
  - `resume_checkpoint_path`
  - `resume_checkpoint_strict`
- Add shared validation for missing resume checkpoint files.
- Add shared helpers that:
  - read checkpoint files from disk
  - accept either a wrapped checkpoint payload or a raw state dict
  - load the state dict into a model before optimizer creation
- Reuse the helper in local/global/seq2seq loops.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

### Task 3: Thread resume params through Torch runtime metadata and entrypoints

**Files:**
- Modify: `src/foresight/models/runtime.py`
- Modify: Torch entrypoint modules that already expose shared trainer config

**Step 1: Write minimal implementation**

- Extend `_coerce_torch_extra_train_params`, `_TORCH_COMMON_DEFAULTS`, and `_TORCH_COMMON_PARAM_HELP`.
- Thread the new resume args into every `TorchTrainConfig(...)` / builder call so registry-created forecasters accept them consistently.

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

Run: `ruff check src/foresight/models/multivariate.py src/foresight/models/runtime.py src/foresight/models/torch_ct_rnn.py src/foresight/models/torch_global.py src/foresight/models/torch_graph_attention.py src/foresight/models/torch_graph_spectral.py src/foresight/models/torch_graph_structure.py src/foresight/models/torch_nn.py src/foresight/models/torch_probabilistic.py src/foresight/models/torch_reservoir.py src/foresight/models/torch_rnn_paper_zoo.py src/foresight/models/torch_rnn_zoo.py src/foresight/models/torch_seq2seq.py src/foresight/models/torch_ssm.py src/foresight/models/torch_structured_rnn.py src/foresight/models/torch_xformer.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS
