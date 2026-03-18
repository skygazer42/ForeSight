# Torch Multivariate Horizon Loss Decay Wave 42 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `horizon_loss_decay` support to the multivariate Torch forecasting stack so runtime-exposed strategy controls work consistently for STID, STGCN, GraphWaveNet, and graph-structure proxy wrappers.

**Architecture:** Thread one shared scalar training control through the multivariate Torch public signatures into `TorchTrainConfig`, reusing the existing weighted-loss implementation already living in `torch_nn.py`. Add tests that prove runtime acceptance for direct multivariate Torch models and validation propagation for graph-structure wrappers that currently swallow unknown extra params.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing runtime tests for multivariate and graph-structure horizon-loss-decay support

**Files:**
- Modify: `tests/test_models_multivariate.py`
- Modify: `tests/test_models_graph_structure_smoke.py`

**Step 1: Write the failing test**

- Add a multivariate Torch smoke test proving these runtime entrypoints accept `horizon_loss_decay=0.5`:
  - `torch-stid-multivariate`
  - `torch-stgcn-multivariate`
  - `torch-graphwavenet-multivariate`
- Add a graph-structure validation test proving `torch-agcrn-multivariate` forwards invalid `horizon_loss_decay=0.0` into shared Torch validation with:
  - `"horizon_loss_decay must be > 0"`

**Step 2: Run the focused tests to verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_models_multivariate.py tests/test_models_graph_structure_smoke.py -k horizon_loss_decay`

Expected: FAIL because multivariate Torch wrappers do not yet accept or forward `horizon_loss_decay`.

### Task 2: Thread horizon-loss-decay through multivariate Torch wrappers and graph-structure proxy

**Files:**
- Modify: `src/foresight/models/multivariate.py`
- Modify: `src/foresight/models/torch_graph_structure.py`

**Step 1: Write minimal implementation**

- Add `horizon_loss_decay: float = 1.0` to:
  - `torch_stid_forecast(...)`
  - `torch_stgcn_forecast(...)`
  - `torch_graphwavenet_forecast(...)`
  - `torch_graph_structure_forecast(...)`
- Pass `horizon_loss_decay=float(horizon_loss_decay)` into each affected `TorchTrainConfig(...)`
- Forward `horizon_loss_decay` from `torch_graph_structure_forecast(...)` into `torch_graphwavenet_forecast(...)`

**Step 2: Run the focused tests to verify GREEN**

Run: `PYTHONPATH=src pytest -q tests/test_models_multivariate.py tests/test_models_graph_structure_smoke.py -k horizon_loss_decay`

Expected: PASS

### Task 3: Run targeted verification

**Files:**
- Verify: `src/foresight/models/multivariate.py`
- Verify: `src/foresight/models/torch_graph_structure.py`
- Verify: `tests/test_models_multivariate.py`
- Verify: `tests/test_models_graph_structure_smoke.py`

**Step 1: Run the targeted tests**

Run: `PYTHONPATH=src pytest -q tests/test_models_multivariate.py tests/test_models_graph_structure_smoke.py`

Expected: PASS

**Step 2: Run syntax verification**

Run: `python -m py_compile src/foresight/models/multivariate.py src/foresight/models/torch_graph_structure.py tests/test_models_multivariate.py tests/test_models_graph_structure_smoke.py`

Expected: PASS

**Step 3: Run lint**

Run: `ruff check src/foresight/models/multivariate.py src/foresight/models/torch_graph_structure.py tests/test_models_multivariate.py tests/test_models_graph_structure_smoke.py`

Expected: PASS
