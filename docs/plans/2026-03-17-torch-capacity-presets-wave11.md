# Torch Capacity Presets Wave11 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add high-capacity `deep` and `wide` Torch local presets for another batch of trainable time-series architectures.

**Architecture:** Extend the local Torch catalog with new preset keys that reuse existing factories and only change default hyperparameters. Guard the surface with registry and optional-dependency tests plus explicit assertions for the new preset defaults.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave11 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-kan-{deep,wide}-direct`
- `torch-scinet-{deep,wide}-direct`
- `torch-etsformer-{deep,wide}-direct`
- `torch-esrnn-{deep,wide}-direct`
- `torch-patchtst-{deep,wide}-direct`
- `torch-crossformer-{deep,wide}-direct`
- `torch-pyraformer-{deep,wide}-direct`
- `torch-tsmixer-{deep,wide}-direct`
- `torch-nhits-{deep,wide}-direct`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: failures for missing wave11 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Reuse the existing model factories and add the new `deep` / `wide` variants by overriding only the depth or width knobs:
- KAN: `num_layers=4`, `d_model=128`
- SCINet: `num_stages=4`, `d_model=128`, `ffn_dim=256`
- ETSformer: `num_layers=4`, `d_model=128`, `nhead=8`, `dim_feedforward=512`
- ESRNN: `num_layers=4`, `hidden_size=128`
- PatchTST: `num_layers=4`, `d_model=128`, `nhead=8`, `dim_feedforward=512`
- Crossformer: `num_layers=4`, `d_model=128`, `nhead=8`, `dim_feedforward=512`
- Pyraformer: `num_layers=4`, `d_model=128`, `nhead=8`, `dim_feedforward=512`
- TSMixer: `num_blocks=6`, `d_model=128`, `token_mixing_hidden=256`, `channel_mixing_hidden=256`
- N-HiTS: `num_blocks=8`, `layer_width=256`

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 3: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
