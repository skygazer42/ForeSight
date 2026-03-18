# Torch Transformer SSM Presets Wave14 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add another batch of `deep` and `wide` local Torch presets for transformer-heavy and state-space model families.

**Architecture:** Extend the local Torch catalog with preset IDs that reuse existing factories and only override depth or width defaults. Guard the additions with registry coverage tests, optional-dependency coverage tests, and explicit default-parameter assertions for the new preset surface.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave14 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-informer-{deep,wide}-direct`
- `torch-autoformer-{deep,wide}-direct`
- `torch-nonstationary-transformer-{deep,wide}-direct`
- `torch-fedformer-{deep,wide}-direct`
- `torch-itransformer-{deep,wide}-direct`
- `torch-timesnet-{deep,wide}-direct`
- `torch-tft-{deep,wide}-direct`
- `torch-timexer-{deep,wide}-direct`
- `torch-s4d-{deep,wide}-direct`
- `torch-s4-{deep,wide}-direct`
- `torch-s5-{deep,wide}-direct`
- `torch-mamba2-{deep,wide}-direct`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: failures for missing wave14 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Add `deep` and `wide` variants with these overrides:
- Informer / Autoformer / Nonstationary Transformer / iTransformer: `num_layers=4`, `d_model=128`, `nhead=8`, `dim_feedforward=512`
- FEDformer: `num_layers=4`, `d_model=128`, `ffn_dim=512`
- TimesNet: `num_layers=4`, `d_model=128`
- TFT: `lstm_layers=2`, `d_model=128`, `nhead=8`
- TimeXer: `num_layers=4`, `d_model=128`, `nhead=8`
- S4D / S4 / S5 / Mamba2: `num_layers=4`, `d_model=128`

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 3: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
