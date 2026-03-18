# Torch Graph Spectral Strategy Forwarding Wave 43 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the graph-spectral multivariate Torch families honor the full shared training-strategy surface already exposed by runtime defaults and catalog help text.

**Architecture:** Extend `torch_graph_spectral_forecast(...)` to accept the remaining shared trainer strategy controls, thread them into `TorchTrainConfig`, and update the multivariate catalog factory to forward late-bound runtime params through to the wrapper. Add focused smoke/validation tests that prove invalid shared strategy values are no longer swallowed by the catalog layer.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing graph-spectral validation tests

**Files:**
- Modify: `tests/test_models_graph_spectral_smoke.py`

**Step 1: Write the failing test**

- Add validation-forwarding coverage for:
  - `horizon_loss_decay=0.0` -> `"horizon_loss_decay must be > 0"`
  - `sam_rho=-0.1` -> `"sam_rho must be >= 0"`
- Cover both:
  - `torch-stemgnn-multivariate`
  - `torch-fouriergnn-multivariate`

**Step 2: Run test to verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_models_graph_spectral_smoke.py -k validation`

Expected: FAIL because graph-spectral catalog/runtime currently swallows these shared strategy params.

### Task 2: Thread shared strategy params through graph-spectral wrapper and catalog

**Files:**
- Modify: `src/foresight/models/torch_graph_spectral.py`
- Modify: `src/foresight/models/catalog/multivariate.py`

**Step 1: Write minimal implementation**

- Add the remaining shared training strategy args to `torch_graph_spectral_forecast(...)`:
  - `sam_rho`
  - `sam_adaptive`
  - `input_dropout`
  - `temporal_dropout`
  - `grad_noise_std`
  - `gc_mode`
  - `agc_clip_factor`
  - `agc_eps`
- Pass them into `TorchTrainConfig(...)`
- Forward `**_params` from the graph-spectral multivariate catalog factory into `torch_graph_spectral_forecast(...)`

**Step 2: Run test to verify GREEN**

Run: `PYTHONPATH=src pytest -q tests/test_models_graph_spectral_smoke.py -k validation`

Expected: PASS

### Task 3: Run focused verification

**Files:**
- Verify: `src/foresight/models/torch_graph_spectral.py`
- Verify: `src/foresight/models/catalog/multivariate.py`
- Verify: `tests/test_models_graph_spectral_smoke.py`

**Step 1: Run targeted tests**

Run: `PYTHONPATH=src pytest -q tests/test_models_graph_spectral_smoke.py`

Expected: PASS

**Step 2: Run syntax verification**

Run: `python -m py_compile src/foresight/models/torch_graph_spectral.py src/foresight/models/catalog/multivariate.py tests/test_models_graph_spectral_smoke.py`

Expected: PASS

**Step 3: Run lint**

Run: `ruff check src/foresight/models/torch_graph_spectral.py src/foresight/models/catalog/multivariate.py tests/test_models_graph_spectral_smoke.py`

Expected: PASS
