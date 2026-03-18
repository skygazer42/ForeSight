# Torch RetNet Recursive Presets Wave28 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semantic `deep` and `wide` recursive Torch presets for the remaining RetNet local family.

**Architecture:** Extend the local Torch catalog with two additional `torch-retnet-recursive` preset IDs that reuse the existing recursive RetNet factory and only override block depth or model width defaults. Protect the new surface with registration checks, optional-dependency coverage checks, and explicit default-parameter assertions.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave28 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-retnet-deep-recursive`
- `torch-retnet-wide-recursive`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_retnet_recursive_preset_torch_local_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_retnet_recursive_preset_torch_local_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave28_retnet_recursive_preset_defaults`

Expected: failures for missing wave28 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Reuse the existing recursive RetNet factory and add preset IDs with these overrides:
- Deep: `num_layers=4`
- Wide: `d_model=128`, `nhead=8`, `ffn_dim=256`

Preserve the existing lag window, dropout, and trainer defaults.

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_retnet_recursive_preset_torch_local_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_retnet_recursive_preset_torch_local_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave28_retnet_recursive_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
