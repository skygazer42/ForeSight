# Torch Trainer Config Wave 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a second wave of high-value Torch trainer controls across ForeSight Torch models: mixed precision, warmup-aware LR floors, and minimum training epochs.

**Architecture:** Extend the shared Torch training config in `src/foresight/models/torch_nn.py` and the global adapter in `src/foresight/models/torch_global.py`, so local/global Torch models inherit the new behavior without introducing a new trainer abstraction. Reuse the existing registry/runtime metadata path in `src/foresight/models/runtime.py`, and patch custom wrappers such as `torch_seq2seq.py` so the shared config surface is consistently accepted everywhere.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for wave-2 trainer config and seq2seq parity

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

- Add local Torch training-config validation tests for:
  - `min_epochs >= 1`
  - `min_epochs <= epochs`
  - `warmup_epochs >= 0`
  - `warmup_epochs <= epochs`
  - `min_lr >= 0`
  - `amp_dtype in {auto, float16, bfloat16}`
  - `amp=True` requiring `device="cuda"`
- Add matching global Torch validation tests.
- Extend registry assertions proving Torch specs expose:
  - `min_epochs`
  - `amp`
  - `amp_dtype`
  - `warmup_epochs`
  - `min_lr`
- Add a seq2seq regression test that exercises `torch-seq2seq-direct` with shared trainer args to catch missing signature forwarding.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: FAIL because the new config surface is not implemented yet and the seq2seq helper signature is incomplete.

### Task 2: Implement shared trainer config, AMP, warmup, and min-epoch behavior

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_seq2seq.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` / `TorchGlobalTrainConfig` with:
  - `min_epochs`
  - `amp`
  - `amp_dtype`
  - `warmup_epochs`
  - `min_lr`
- Validate the new fields and device requirements.
- Update local/global training loops to:
  - enforce `min_epochs` before early stopping
  - apply linear warmup for the first `warmup_epochs`
  - clamp LR floors via `min_lr`
  - support CUDA autocast + gradient scaling when `amp=True`
- Repair `torch_seq2seq.py` shared config builder/signatures so it accepts the full shared trainer config explicitly.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

### Task 3: Expose the new trainer config through runtime metadata and model wrappers

**Files:**
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: Torch wrapper files that already expose shared trainer config in `src/foresight/models/`

**Step 1: Write minimal implementation**

- Extend `_coerce_torch_extra_train_params`, `_TORCH_COMMON_DEFAULTS`, and `_TORCH_COMMON_PARAM_HELP`.
- Thread the new args through local/global Torch wrappers so registry-created forecasters accept them.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py`

Expected: PASS

### Task 4: Verify targeted Torch behavior stays green

**Files:**
- Verify: `tests/test_models_optional_deps_torch.py`
- Verify: `tests/test_torch_nn_validation_messages.py`
- Verify: `tests/test_torch_global_validation_messages.py`
- Verify: `tests/test_models_registry.py`

**Step 1: Run targeted verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

**Step 2: Run lint on changed files**

Run: `ruff check src/foresight/models/torch_nn.py src/foresight/models/torch_global.py src/foresight/models/torch_seq2seq.py src/foresight/models/runtime.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS
