# Torch SWA Strategy Wave 32 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a lightweight stochastic weight averaging strategy across ForeSight Torch trainers so local/global/seq2seq Torch models can optionally export and resume SWA weights.

**Architecture:** Extend the shared Torch training config in `src/foresight/models/torch_nn.py` with a single `swa_start_epoch` control, treat `-1` as disabled, and reuse the existing checkpoint/resume payload flow to persist raw training weights separately from deployable SWA weights. Keep scope narrow by disallowing simultaneous EMA and SWA so the deploy-state semantics remain unambiguous in shared local/global/seq2seq loops.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for SWA config, checkpoint payloads, and registry metadata

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing test**

- Extend local/global validation coverage for:
  - `swa_start_epoch >= -1`
  - `swa_start_epoch <= epochs`
  - `ema_decay` and `swa_start_epoch` cannot both be enabled
- Extend shared config / registry assertions to cover:
  - `swa_start_epoch`
- Add a local runtime smoke test proving `torch-mlp-direct` accepts SWA controls.
- Add a global runtime smoke test proving `torch-timexer-global` accepts SWA controls.
- Extend checkpoint payload tests so SWA-enabled training writes:
  - `swa_state`
  - `model_state`
- Extend shared loader test proving:
  - `_load_torch_training_state(...)` returns `swa_state`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: FAIL because SWA config, metadata, and trainer-loop behavior are not implemented yet.

### Task 2: Implement shared SWA trainer behavior and resume semantics

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` / `TorchGlobalTrainConfig` / `TorchCheckpointResumeState` with SWA fields.
- Add shared SWA helpers that:
  - create a frozen SWA shadow model
  - start averaging after `swa_start_epoch`
  - update average weights after optimizer steps
  - select SWA weights for validation / best-state snapshots when active
- Extend checkpoint snapshot/load helpers so they persist and restore:
  - `swa_state`
  - raw `model_state`
- Update shared local/global/seq2seq loops plus the remaining custom seq2seq-style Torch loop in `torch_rnn_paper_zoo.py` so SWA participates in:
  - validation
  - best-state tracking
  - checkpoint save/load
  - final `restore_best`

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

### Task 3: Expose SWA controls through runtime metadata and wrapper forwarding

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
- Thread SWA controls through the common Torch wrapper signatures and `TorchTrainConfig(...)` builders so registry-created forecasters accept them consistently.

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
