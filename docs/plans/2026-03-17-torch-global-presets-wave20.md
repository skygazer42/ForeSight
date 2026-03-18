# Torch Global Presets Wave20 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semantic and capacity-oriented global Torch presets for the remaining low-risk baseline families with clear precedent from existing local defaults.

**Architecture:** Extend the global Torch catalog with a small mixed batch of presets that either mirror existing direct-model `long` and `wide` defaults or reuse the common recurrent depth/width pattern already established in other global Torch models. Protect the additions with registration checks, optional-dependency coverage checks, and explicit default-parameter assertions.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave20 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-deepar-{deep,wide}-global`
- `torch-tide-long-global`
- `torch-tide-wide-global`
- `torch-nlinear-long-global`
- `torch-dlinear-long-global`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave20_global_preset_defaults`

Expected: failures for missing wave20 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`

**Step 1: Add minimal catalog entries**

Reuse the existing global factories and add preset IDs with these overrides:
- DeepAR: `num_layers=2`, `dropout=0.1`; and `hidden_size=128`
- TiDE: `context_length=192`; and `d_model=128`, `hidden_size=256`
- NLinear: `context_length=192`
- DLinear: `context_length=192`

Preserve the existing base training defaults and helper strings for each family.

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave20_global_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_global.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
