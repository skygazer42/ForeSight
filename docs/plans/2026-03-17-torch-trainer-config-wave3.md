# Torch Trainer Config Wave 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a third wave of shared Torch trainer controls by supporting `onecycle` and `cosine_restarts` learning-rate schedulers across local/global Torch models.

**Architecture:** Extend the shared scheduler/config path in `src/foresight/models/torch_nn.py`, then let `torch_global.py` and `torch_seq2seq.py` inherit the new behavior through the existing shared trainer helpers. Expose the new knobs through `src/foresight/models/runtime.py` and bulk-thread them through Torch wrapper signatures so registry-created forecasters accept the same config surface everywhere.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for the new scheduler surface

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

- Add local Torch validation tests for:
  - `scheduler in {none, cosine, step, plateau, onecycle, cosine_restarts}`
  - `scheduler_restart_period >= 1`
  - `scheduler_restart_mult >= 1`
  - `scheduler_pct_start in (0, 1)`
- Add matching global Torch validation tests.
- Add positive smoke tests proving:
  - a local Torch forecaster can train with `scheduler="onecycle"`
  - a global Torch forecaster can train with `scheduler="cosine_restarts"`
- Extend registry assertions proving Torch specs expose:
  - `scheduler_restart_period`
  - `scheduler_restart_mult`
  - `scheduler_pct_start`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: FAIL because the scheduler config surface is not implemented yet.

### Task 2: Implement shared scheduler behavior

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_seq2seq.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` / `TorchGlobalTrainConfig` with:
  - `scheduler_restart_period`
  - `scheduler_restart_mult`
  - `scheduler_pct_start`
- Update shared validation to enforce the new constraints.
- Extend `_make_torch_scheduler(...)` to support:
  - `cosine_restarts` via `CosineAnnealingWarmRestarts`
  - `onecycle` via `OneCycleLR`
- Update local/global/seq2seq training loops so `onecycle` steps per optimizer step while other schedulers continue to step per epoch.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

### Task 3: Expose the new scheduler config through runtime metadata and wrappers

**Files:**
- Modify: `src/foresight/models/runtime.py`
- Modify: Torch wrapper files in `src/foresight/models/` that already expose shared trainer config

**Step 1: Write minimal implementation**

- Extend `_coerce_torch_extra_train_params`, `_TORCH_COMMON_DEFAULTS`, and `_TORCH_COMMON_PARAM_HELP`.
- Thread the new args through local/global Torch wrappers and seq2seq-specific builders so registry-created forecasters accept them consistently.

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
