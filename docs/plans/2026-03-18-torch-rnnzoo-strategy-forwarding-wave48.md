# Torch RNNZoo Strategy Forwarding Wave 48 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore the Torch RNN Zoo local wrappers to a working default runtime path and align them with the shared Torch training-strategy surface.

**Architecture:** Extend the shared RNN Zoo training-config builder and the public `torch_rnnzoo_direct_forecast(...)` entrypoint so runtime-forwarded trainer strategy params no longer fail with unexpected-keyword errors. Add a validation-forwarding test over the existing RNN Zoo smoke keys to prove invalid shared strategy values now reach shared trainer validation.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing RNN Zoo validation-forwarding tests

**Files:**
- Modify: `tests/test_models_rnn_zoo_100.py`

**Step 1: Write the failing test**

- Add validation coverage for the existing smoke keys:
  - `torch-rnnzoo-elman-direct`
  - `torch-rnnzoo-peephole-lstm-ln-direct`
  - `torch-rnnzoo-indrnn-attn-direct`
  - `torch-rnnzoo-qrnn-proj-direct`
  - `torch-rnnzoo-phased-lstm-bidir-direct`
- Validate both:
  - `horizon_loss_decay=0.0` -> `"horizon_loss_decay must be > 0"`
  - `sam_rho=-0.1` -> `"sam_rho must be >= 0"`

**Step 2: Run test to verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_models_rnn_zoo_100.py -k "smoke or validation"`

Expected: FAIL because runtime currently forwards shared strategy params that the RNN Zoo wrapper does not yet accept.

### Task 2: Thread shared strategy params through the RNN Zoo wrapper

**Files:**
- Modify: `src/foresight/models/torch_rnn_zoo.py`

**Step 1: Write minimal implementation**

- Add these shared training strategy args to `_build_rnnzoo_train_config(...)` and `torch_rnnzoo_direct_forecast(...)`:
  - `sam_rho`
  - `sam_adaptive`
  - `input_dropout`
  - `temporal_dropout`
  - `grad_noise_std`
  - `gc_mode`
  - `agc_clip_factor`
  - `agc_eps`
- Pass them into `TorchTrainConfig(...)`

**Step 2: Run test to verify GREEN**

Run: `PYTHONPATH=src pytest -q tests/test_models_rnn_zoo_100.py -k "smoke or validation"`

Expected: PASS

### Task 3: Run focused verification

**Files:**
- Verify: `src/foresight/models/torch_rnn_zoo.py`
- Verify: `tests/test_models_rnn_zoo_100.py`

**Step 1: Run targeted verification**

Run: `PYTHONPATH=src pytest -q tests -k "rnnzoo"`

Expected: PASS

**Step 2: Run syntax verification**

Run: `python -m py_compile src/foresight/models/torch_rnn_zoo.py tests/test_models_rnn_zoo_100.py`

Expected: PASS

**Step 3: Run lint**

Run: `ruff check src/foresight/models/torch_rnn_zoo.py tests/test_models_rnn_zoo_100.py`

Expected: PASS
