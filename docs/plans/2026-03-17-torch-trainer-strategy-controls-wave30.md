# Torch Trainer Strategy Controls Wave 30 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new shared Torch trainer strategy wave by supporting richer `ReduceLROnPlateau` controls and configurable gradient-clipping modes across ForeSight Torch models.

**Architecture:** Extend the shared Torch training config in `src/foresight/models/torch_nn.py`, mirror the fields through `src/foresight/models/torch_global.py` and `src/foresight/models/torch_seq2seq.py`, and expose the same knobs through `src/foresight/models/runtime.py`. Keep the change inside the existing lightweight trainer path so local/global/seq2seq Torch entrypoints inherit the new behavior without introducing callbacks or a new trainer abstraction.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for the new trainer strategy surface

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing test**

- Extend local/global validation coverage for:
  - `grad_clip_mode in {norm, value}`
  - `grad_clip_value >= 0`
  - `scheduler_plateau_factor in (0,1)`
  - `scheduler_plateau_threshold >= 0`
- Add a local runtime smoke test proving `torch-mlp-direct` accepts:
  - `grad_clip_mode="value"`
  - `grad_clip_value > 0`
  - `scheduler="plateau"` with non-default plateau controls
- Add a matching global runtime smoke test for `torch-timexer-global`.
- Extend registry assertions proving Torch specs expose the new defaults and help text.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: FAIL because the new shared trainer strategy controls are not implemented yet.

### Task 2: Implement shared trainer strategy controls

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_seq2seq.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` / `TorchGlobalTrainConfig` with:
  - `grad_clip_mode`
  - `grad_clip_value`
  - `scheduler_plateau_factor`
  - `scheduler_plateau_threshold`
- Validate the new fields with stable shared error messages.
- Update shared local/global/seq2seq loops so gradient clipping supports:
  - norm clipping via `clip_grad_norm_`
  - value clipping via `clip_grad_value_`
- Extend `ReduceLROnPlateau` creation so it respects the new plateau controls.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

### Task 3: Expose the new controls through runtime metadata and wrapper forwarding

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
- Thread the new shared strategy args into every builder / wrapper that forwards shared Torch trainer config so registry-created forecasters accept them consistently.

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
