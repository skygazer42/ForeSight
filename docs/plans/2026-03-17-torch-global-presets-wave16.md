# Torch Global Presets Wave16 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `deep` and `wide` global Torch presets for the highest-value panel forecasting architectures that currently expose only a base global config.

**Architecture:** Extend the global Torch catalog with additional preset IDs that reuse the existing global forecaster factories and only override model-capacity defaults. Protect the surface with registry presence tests, optional-dependency coverage tests, and explicit assertions for the new global default values.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave16 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-tft-{deep,wide}-global`
- `torch-timexer-{deep,wide}-global`
- `torch-retnet-{deep,wide}-global`
- `torch-informer-{deep,wide}-global`
- `torch-autoformer-{deep,wide}-global`
- `torch-fedformer-{deep,wide}-global`
- `torch-nonstationary-transformer-{deep,wide}-global`
- `torch-patchtst-{deep,wide}-global`
- `torch-crossformer-{deep,wide}-global`
- `torch-pyraformer-{deep,wide}-global`
- `torch-itransformer-{deep,wide}-global`
- `torch-timesnet-{deep,wide}-global`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave16_global_preset_defaults`

Expected: failures for missing wave16 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`

**Step 1: Add minimal catalog entries**

Reuse the existing global factories and add preset IDs with these overrides:
- TFT: `lstm_layers=2`, `d_model=128`, `nhead=8`
- TimeXer / Informer / Autoformer / Nonstationary Transformer / iTransformer / PatchTST / Crossformer / Pyraformer: `num_layers=4`, `d_model=128`, `nhead=8`, `dim_feedforward=512`
- RetNet: `num_layers=4`, `d_model=128`, `nhead=8`, `ffn_dim=512`
- FEDformer: `num_layers=4`, `d_model=128`, `ffn_dim=512`
- TimesNet: `num_layers=4`, `d_model=128`

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave16_global_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_global.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
