# Torch Graph Attention Strategy Forwarding Wave 44 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore graph-attention multivariate models to a working default state and make them honor the shared Torch training-strategy surface exposed by runtime defaults.

**Architecture:** Extend `torch_graph_attention_forecast(...)` with the missing shared trainer strategy arguments so the existing catalog `**_params` forwarding no longer crashes on default runtime settings. Add validation-forwarding tests to prove graph-attention wrappers now route invalid shared strategy values into shared `TorchTrainConfig` validation instead of failing with unexpected-keyword errors.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add graph-attention validation-forwarding tests

**Files:**
- Modify: `tests/test_models_graph_attention_smoke.py`

**Step 1: Write the failing test**

- Add validation coverage for:
  - `horizon_loss_decay=0.0` -> `"horizon_loss_decay must be > 0"`
  - `sam_rho=-0.1` -> `"sam_rho must be >= 0"`
- Cover both:
  - `torch-astgcn-multivariate`
  - `torch-gman-multivariate`

**Step 2: Run test to verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_models_graph_attention_smoke.py`

Expected: FAIL because graph-attention wrappers do not yet accept the full shared strategy surface and currently crash on default forwarded params.

### Task 2: Thread shared strategy params through graph-attention wrapper

**Files:**
- Modify: `src/foresight/models/torch_graph_attention.py`

**Step 1: Write minimal implementation**

- Add the remaining shared training strategy args to `torch_graph_attention_forecast(...)`:
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

Run: `PYTHONPATH=src pytest -q tests/test_models_graph_attention_smoke.py`

Expected: PASS

### Task 3: Run focused verification

**Files:**
- Verify: `src/foresight/models/torch_graph_attention.py`
- Verify: `tests/test_models_graph_attention_smoke.py`

**Step 1: Run syntax verification**

Run: `python -m py_compile src/foresight/models/torch_graph_attention.py tests/test_models_graph_attention_smoke.py`

Expected: PASS

**Step 2: Run lint**

Run: `ruff check src/foresight/models/torch_graph_attention.py tests/test_models_graph_attention_smoke.py`

Expected: PASS
