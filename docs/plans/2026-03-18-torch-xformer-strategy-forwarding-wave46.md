# Torch XFormer Strategy Forwarding Wave 46 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore the local xformer direct family to a working default runtime path and align it with the shared Torch training-strategy surface already forwarded by runtime.

**Architecture:** Keep the existing runtime forwarding in place, but extend `torch_xformer_direct_forecast(...)` so it accepts the missing shared training-strategy arguments and threads them into `TorchTrainConfig`. Add a focused validation-forwarding test for one xformer local key to prove invalid shared strategy values now reach shared trainer validation instead of failing with unexpected-keyword errors.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing xformer validation-forwarding test

**Files:**
- Modify: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Step 1: Write the failing test**

- Add validation coverage for `torch-xformer-performer-ln-gelu-direct`:
  - `horizon_loss_decay=0.0` -> `"horizon_loss_decay must be > 0"`
  - `sam_rho=-0.1` -> `"sam_rho must be >= 0"`

**Step 2: Run test to verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "xformer_local"`

Expected: FAIL because the xformer direct wrapper currently does not accept the full shared strategy surface and default runtime forwarding already crashes on `sam_rho`.

### Task 2: Thread shared strategy params through xformer direct wrapper

**Files:**
- Modify: `src/foresight/models/torch_xformer.py`

**Step 1: Write minimal implementation**

- Add the remaining shared training strategy args to `torch_xformer_direct_forecast(...)`:
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

Run: `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "xformer_local"`

Expected: PASS

### Task 3: Run focused verification

**Files:**
- Verify: `src/foresight/models/torch_xformer.py`
- Verify: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Step 1: Run syntax verification**

Run: `python -m py_compile src/foresight/models/torch_xformer.py tests/test_models_torch_xformer_seq2seq_smoke.py`

Expected: PASS

**Step 2: Run lint**

Run: `ruff check src/foresight/models/torch_xformer.py tests/test_models_torch_xformer_seq2seq_smoke.py`

Expected: PASS
