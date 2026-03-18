# Torch Lookahead Strategy Wave 33 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a lightweight Lookahead training strategy across ForeSight Torch trainers so local/global/seq2seq Torch models can optionally train with slow weights, export deployable Lookahead checkpoints, and resume the slow-weight state.

**Architecture:** Extend the shared Torch training config in `src/foresight/models/torch_nn.py` with `lookahead_steps` and `lookahead_alpha`, treating `lookahead_steps=0` as disabled. Reuse the existing EMA/SWA checkpoint pattern by maintaining a frozen Lookahead slow model, persisting its state plus a step counter, and saving raw fast weights separately in `model_state` whenever deploy weights differ from training weights.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for Lookahead config, checkpoint payloads, and registry metadata

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing test**

- Extend local/global validation coverage for:
  - `lookahead_steps >= 0`
  - `lookahead_alpha in (0, 1]`
- Extend shared config / registry assertions to cover:
  - `lookahead_steps`
  - `lookahead_alpha`
- Add a local runtime smoke test proving `torch-mlp-direct` accepts Lookahead controls.
- Add a global runtime smoke test proving `torch-timexer-global` accepts Lookahead controls.
- Extend checkpoint payload tests so Lookahead-enabled training writes:
  - `lookahead_state`
  - `lookahead_step`
  - `model_state`
- Extend shared loader test proving:
  - `_load_torch_training_state(...)` returns `lookahead_state`
  - `_load_torch_training_state(...)` returns `lookahead_step`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: FAIL because Lookahead config, metadata, and trainer-loop behavior are not implemented yet.

### Task 2: Implement shared Lookahead trainer behavior and resume semantics

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` / `TorchGlobalTrainConfig` / `TorchCheckpointResumeState` with Lookahead fields.
- Add shared Lookahead helpers that:
  - create a frozen slow-weight shadow model
  - sync every `lookahead_steps` optimizer steps
  - blend slow weights with `lookahead_alpha`
  - select slow weights for validation / best-state snapshots when active
- Extend checkpoint snapshot/load helpers so they persist and restore:
  - `lookahead_state`
  - `lookahead_step`
  - raw `model_state`
- Update shared local/global/seq2seq loops plus the remaining custom seq2seq-style Torch loop in `torch_rnn_paper_zoo.py` so Lookahead participates in:
  - optimizer-step updates
  - validation / best-state tracking
  - checkpoint save/load
  - final `restore_best`

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

### Task 3: Expose Lookahead controls through runtime metadata and wrapper forwarding

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
- Thread Lookahead controls through the common Torch wrapper signatures and `TorchTrainConfig(...)` builders so registry-created forecasters accept them consistently.

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
