# Torch SSM And CT-RNN Strategy Forwarding Wave 47 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore the remaining state-space and continuous-time local Torch wrappers to a working default runtime path and align them with the shared Torch training-strategy surface.

**Architecture:** Extend the affected direct wrappers in `torch_ssm.py` and `torch_ct_rnn.py` so they accept the same shared training-strategy controls already forwarded by runtime, and thread those values into `TorchTrainConfig`. Add one parameterized validation-forwarding test over the currently failing local keys so invalid shared strategy values prove the end-to-end runtime chain is no longer breaking on unexpected-keyword errors.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing validation-forwarding coverage for the affected local wrappers

**Files:**
- Modify: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Step 1: Write the failing test**

- Add a shared validation-forwarding test covering:
  - `torch-lmu-direct`
  - `torch-s4d-direct`
  - `torch-ltc-direct`
  - `torch-cfc-direct`
  - `torch-xlstm-direct`
  - `torch-mamba2-direct`
  - `torch-s4-direct`
  - `torch-s5-direct`
  - `torch-griffin-direct`
  - `torch-hawk-direct`
- Validate both:
  - `horizon_loss_decay=0.0` -> `"horizon_loss_decay must be > 0"`
  - `sam_rho=-0.1` -> `"sam_rho must be >= 0"`

**Step 2: Run test to verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "state_space_torch_local_smoke or continuous_time_torch_local_smoke or revival_torch_local_smoke or ssm_torch_local_smoke or recurrent_revival_torch_local_smoke"`

Expected: FAIL because runtime currently forwards shared strategy params that these wrappers do not yet accept.

### Task 2: Thread shared strategy params through SSM and CT-RNN direct wrappers

**Files:**
- Modify: `src/foresight/models/torch_ssm.py`
- Modify: `src/foresight/models/torch_ct_rnn.py`

**Step 1: Write minimal implementation**

- Add these shared strategy args to all affected direct wrappers:
  - `sam_rho`
  - `sam_adaptive`
  - `input_dropout`
  - `temporal_dropout`
  - `grad_noise_std`
  - `gc_mode`
  - `agc_clip_factor`
  - `agc_eps`
- Pass them into each affected `TorchTrainConfig(...)`

**Step 2: Run test to verify GREEN**

Run: `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "state_space_torch_local_smoke or continuous_time_torch_local_smoke or revival_torch_local_smoke or ssm_torch_local_smoke or recurrent_revival_torch_local_smoke"`

Expected: PASS

### Task 3: Run focused verification

**Files:**
- Verify: `src/foresight/models/torch_ssm.py`
- Verify: `src/foresight/models/torch_ct_rnn.py`
- Verify: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Step 1: Run full targeted smoke**

Run: `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py`

Expected: PASS

**Step 2: Run syntax verification**

Run: `python -m py_compile src/foresight/models/torch_ssm.py src/foresight/models/torch_ct_rnn.py tests/test_models_torch_xformer_seq2seq_smoke.py`

Expected: PASS

**Step 3: Run lint**

Run: `ruff check src/foresight/models/torch_ssm.py src/foresight/models/torch_ct_rnn.py tests/test_models_torch_xformer_seq2seq_smoke.py`

Expected: PASS
