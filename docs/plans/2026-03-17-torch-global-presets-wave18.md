# Torch Global Presets Wave18 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `deep` and `wide` global Torch presets for the next sequence and state-space panel forecasting families that still expose only a base global config.

**Architecture:** Extend the global Torch catalog with more preset IDs that reuse the existing global forecaster factories and only override depth/width defaults. Protect the new API surface with tuple-based registration tests, optional-dependency coverage tests, and explicit default-parameter assertions.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave18 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-fnet-{deep,wide}-global`
- `torch-gmlp-{deep,wide}-global`
- `torch-ssm-{deep,wide}-global`
- `torch-mamba-{deep,wide}-global`
- `torch-rwkv-{deep,wide}-global`
- `torch-hyena-{deep,wide}-global`
- `torch-dilated-rnn-{deep,wide}-global`
- `torch-transformer-encdec-{deep,wide}-global`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave18_global_preset_defaults`

Expected: failures for missing wave18 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`

**Step 1: Add minimal catalog entries**

Reuse the existing global factories and add preset IDs with these overrides:
- FNet: `num_layers=6`, `d_model=128`, `dim_feedforward=512`
- gMLP: `num_layers=6`, `d_model=128`, `ffn_dim=256`
- SSM: `num_layers=6`, `d_model=128`
- Mamba: `num_layers=6`, `d_model=128`
- RWKV: `num_layers=6`, `d_model=128`, `ffn_dim=256`
- Hyena: `num_layers=6`, `d_model=128`, `ffn_dim=256`
- Dilated-RNN: `num_layers=5`, `d_model=128`
- Transformer encoder-decoder: `num_layers=4`, `d_model=128`, `nhead=8`, `dim_feedforward=512`

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave18_global_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_global.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
