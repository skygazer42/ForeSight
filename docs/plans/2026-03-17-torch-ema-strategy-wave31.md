# Torch EMA Strategy Wave 31 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a shared exponential-moving-average training strategy across ForeSight Torch trainers so local/global/seq2seq Torch models can optionally track and restore EMA weights during training.

**Architecture:** Extend the shared Torch training config in `src/foresight/models/torch_nn.py` with EMA controls, then implement reusable EMA helpers that plug into the shared local/global/seq2seq trainer loops and checkpoint payloads. Reuse the same runtime metadata flow in `src/foresight/models/runtime.py`, and patch the remaining custom Torch seq2seq-style loops that already participate in the shared checkpoint/resume path so EMA state is not silently dropped.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for EMA config, checkpoint payloads, and runtime usage

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing test**

- Extend local/global validation coverage for:
  - `ema_decay in [0,1)`
  - `ema_warmup_epochs >= 0`
  - `ema_warmup_epochs <= epochs`
- Extend shared config / registry assertions to cover:
  - `ema_decay`
  - `ema_warmup_epochs`
- Add a local runtime smoke test proving `torch-mlp-direct` accepts EMA controls.
- Add a global runtime smoke test proving `torch-timexer-global` accepts EMA controls.
- Extend checkpoint payload tests so EMA-enabled training writes:
  - `ema_state`
  - `model_state`
- Add a shared loader test proving:
  - `_load_torch_training_state(...)` restores raw `model_state` for resume
  - returned resume metadata includes `ema_state`
  - the generic inference loader still reads deployment `state_dict`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: FAIL because EMA config, checkpoint payload handling, and trainer-loop behavior are not implemented yet.

### Task 2: Implement shared EMA trainer behavior and resume semantics

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` / `TorchGlobalTrainConfig` with:
  - `ema_decay`
  - `ema_warmup_epochs`
- Add shared EMA helpers that:
  - create a frozen EMA shadow model
  - start EMA after the configured warmup
  - update EMA weights after optimizer steps
  - pick the EMA model for validation / best-state snapshots once active
- Extend checkpoint snapshot/load helpers so they can persist and restore:
  - deployment `state_dict`
  - raw `model_state` for training resume
  - `ema_state`
- Update shared local/global/seq2seq loops plus the remaining custom seq2seq-style Torch loop in `torch_rnn_paper_zoo.py` so EMA state participates in:
  - validation
  - best-state tracking
  - checkpoint save/load
  - final `restore_best`

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

### Task 3: Expose EMA controls through runtime metadata and wrapper forwarding

**Files:**
- Modify: `src/foresight/models/runtime.py`
- Modify: Torch entrypoint modules that construct `TorchTrainConfig(...)`, including:
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
- Thread EMA controls through the common Torch wrapper signatures and `TorchTrainConfig(...)` builders so registry-created forecasters accept them consistently.

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
