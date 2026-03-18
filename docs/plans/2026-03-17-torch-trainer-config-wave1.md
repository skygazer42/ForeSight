# Torch Trainer Config Wave 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first wave of higher-value Torch trainer configuration options that work across ForeSight local/global Torch models without introducing a new trainer framework.

**Architecture:** Extend the shared Torch training config objects and training loops in `src/foresight/models/torch_nn.py` and `src/foresight/models/torch_global.py`, then expose the same knobs through the existing registry/runtime metadata in `src/foresight/models/runtime.py`. Keep the change on the current lightweight trainer path rather than introducing callbacks, experiment logging, or distributed backends.

**Tech Stack:** Python, PyTorch, pytest

---

### Task 1: Add failing tests for the new trainer config surface

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

- Add local Torch training-config validation tests for:
  - `grad_accum_steps >= 1`
  - `monitor in {auto, train_loss, val_loss}`
  - `monitor_mode in {min, max}`
  - `min_delta >= 0`
  - `num_workers >= 0`
  - `persistent_workers` requiring `num_workers > 0`
  - `scheduler in {none, cosine, step, plateau}`
  - `scheduler_patience >= 1`
- Add the matching global Torch validation tests.
- Add registry assertions proving common Torch model specs expose the new default params and param help strings.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: FAIL because the new config surface is not implemented yet.

### Task 2: Implement the shared trainer config and training-loop behavior

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` and `TorchGlobalTrainConfig` with:
  - `grad_accum_steps`
  - `monitor`
  - `monitor_mode`
  - `min_delta`
  - `num_workers`
  - `pin_memory`
  - `persistent_workers`
  - `scheduler_patience`
- Update both training loops to:
  - validate the new fields
  - build DataLoaders with the new worker options
  - support gradient accumulation
  - support `ReduceLROnPlateau`
  - evaluate the selected monitor consistently across train/validation cases

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py`

Expected: PASS

### Task 3: Expose the new trainer config through model registry defaults

**Files:**
- Modify: `src/foresight/models/runtime.py`

**Step 1: Write minimal implementation**

- Extend `_TORCH_COMMON_DEFAULTS` and `_TORCH_COMMON_PARAM_HELP`.
- Update Torch runtime factories so the new common training params flow into local/global Torch forecasters instead of being swallowed by `**_params`.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py`

Expected: PASS

### Task 4: Verify targeted runtime behavior stays green

**Files:**
- Verify: `tests/test_models_optional_deps_torch.py`
- Verify: `tests/test_torch_nn_validation_messages.py`
- Verify: `tests/test_torch_global_validation_messages.py`
- Verify: `tests/test_models_registry.py`

**Step 1: Run targeted verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

**Step 2: Run lint on changed files**

Run: `ruff check src/foresight/models/torch_nn.py src/foresight/models/torch_global.py src/foresight/models/runtime.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS
