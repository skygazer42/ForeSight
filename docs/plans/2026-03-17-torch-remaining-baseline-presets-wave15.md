# Torch Remaining Baseline Presets Wave15 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add meaningful preset variants for the remaining local Torch baseline families that do not all expose natural `deep` and `wide` capacity axes.

**Architecture:** Extend the local Torch catalog with a small set of semantically useful preset IDs. Use `deep/wide` where the family has real depth/width knobs and `long/wide` where the family is effectively shallow but benefits from longer context windows or wider hidden projections.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave15 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-retnet-{deep,wide}-direct`
- `torch-lightts-{long,wide}-direct`
- `torch-sparsetsf-{long,wide}-direct`
- `torch-tide-{long,wide}-direct`
- `torch-nlinear-long-direct`
- `torch-dlinear-long-direct`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: failures for missing wave15 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Add variants with these overrides:
- RetNet: `num_layers=4`, `d_model=128`, `nhead=8`, `ffn_dim=256`
- LightTS: `lags=192`, `chunk_len=24`, `d_model=128`
- SparseTSF: `lags=336`, `d_model=128`
- TiDE: `lags=192`, `d_model=128`, `hidden_size=256`
- NLinear: `lags=192`
- DLinear: `lags=192`

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 3: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
