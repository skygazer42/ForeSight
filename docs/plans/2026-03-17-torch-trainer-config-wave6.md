# Torch Trainer Config Wave 6 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade Torch trainer checkpoints from model-only warm start to full training resume by persisting and restoring trainer state such as optimizer, scheduler, scaler, epoch, and early-stop metadata.

**Architecture:** Extend the shared checkpoint payload in `src/foresight/models/torch_nn.py` to capture trainer metadata alongside model weights, then add a shared loader that can restore trainer state when present while remaining backward compatible with older model-only checkpoints and raw `state_dict` files. Reuse that machinery in the shared local/global/seq2seq loops and patch the remaining custom Torch trainers that already expose checkpoint save/load behavior.

**Tech Stack:** Python, PyTorch, dataclasses, copy, pathlib, pytest, ruff

---

### Task 1: Add failing tests for full trainer-state resume

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`

**Step 1: Write the failing test**

- Add a unit test that saves a checkpoint payload containing:
  - `state_dict`
  - `optimizer_state`
  - `scheduler_state`
  - `epoch`
  - `best_monitor`
  - `bad_epochs`
  - `best_epoch`
  - `best_state`
- Load that payload into a fresh model/optimizer/scheduler via a shared helper and assert:
  - model weights are restored
  - optimizer state is restored
  - scheduler state is restored
  - resume metadata is returned intact
- Add a checkpoint file payload test proving `last.pt` now contains trainer state keys.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: FAIL because full resume metadata is not implemented yet.

### Task 2: Implement shared checkpoint payload and resume helpers

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`

**Step 1: Write minimal implementation**

- Add a shared trainer-resume metadata structure/helper.
- Extend checkpoint save helpers so `last.pt` stores:
  - optimizer/scheduler/scaler state
  - `epoch`
  - `best_monitor`
  - `bad_epochs`
  - `best_epoch`
  - `best_state`
- Save best checkpoint with equivalent best-epoch metadata when available.
- Extend the shared load helper so it can:
  - restore model weights
  - optionally restore optimizer/scheduler/scaler state
  - return resume metadata
  - remain backward compatible with model-only checkpoints
- Use returned metadata to continue loops from the saved epoch and preserve early-stop tracking.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

### Task 3: Verify remaining checkpoint-aware trainers

**Files:**
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`

**Step 1: Write minimal implementation**

- Patch the non-shared custom trainer loops that already participate in checkpoint save/load so they save and restore the same trainer-state payload shape.

**Step 2: Run targeted tests**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py`

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
