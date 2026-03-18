# Torch Probabilistic Strategy Forwarding Wave 45 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore the local probabilistic Torch families to a working default runtime path and align them with the shared Torch training-strategy surface.

**Architecture:** Update the local probabilistic catalog factory so merged runtime defaults no longer fail at instantiation, then extend `torch_probabilistic_direct_forecast(...)` with the remaining shared trainer strategy controls and thread them into `TorchTrainConfig`. Add validation-forwarding tests to prove invalid shared strategy inputs now reach shared training validation.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add probabilistic validation-forwarding tests

**Files:**
- Modify: `tests/test_models_probabilistic_smoke.py`

**Step 1: Write the failing test**

- Add validation coverage for:
  - `horizon_loss_decay=0.0` -> `"horizon_loss_decay must be > 0"`
  - `sam_rho=-0.1` -> `"sam_rho must be >= 0"`
- Cover both:
  - `torch-timegrad-direct`
  - `torch-tactis-direct`

**Step 2: Run test to verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_models_probabilistic_smoke.py`

Expected: FAIL because the probabilistic factory currently rejects merged runtime defaults and does not yet forward the full shared strategy surface.

### Task 2: Thread shared strategy params through probabilistic catalog and wrapper

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`
- Modify: `src/foresight/models/torch_probabilistic.py`

**Step 1: Write minimal implementation**

- Add `**params` handling to the probabilistic local catalog factory and forward `**extra_params` into `torch_probabilistic_direct_forecast(...)`
- Add the remaining shared training strategy args to `torch_probabilistic_direct_forecast(...)`:
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

Run: `PYTHONPATH=src pytest -q tests/test_models_probabilistic_smoke.py`

Expected: PASS

### Task 3: Run focused verification

**Files:**
- Verify: `src/foresight/models/catalog/torch_local.py`
- Verify: `src/foresight/models/torch_probabilistic.py`
- Verify: `tests/test_models_probabilistic_smoke.py`

**Step 1: Run syntax verification**

Run: `python -m py_compile src/foresight/models/catalog/torch_local.py src/foresight/models/torch_probabilistic.py tests/test_models_probabilistic_smoke.py`

Expected: PASS

**Step 2: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py src/foresight/models/torch_probabilistic.py tests/test_models_probabilistic_smoke.py`

Expected: PASS
