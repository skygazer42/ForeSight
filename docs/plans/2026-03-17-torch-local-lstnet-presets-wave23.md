# Torch Local LSTNet Presets Wave23 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semantic `long` and `wide` direct Torch presets for the remaining local LSTNet family.

**Architecture:** Extend the local Torch catalog with two additional `torch-lstnet-direct` preset IDs that reuse the existing direct LSTNet factory and only override lag-window or hidden-width defaults. Protect the new surface with registration checks, optional-dependency coverage checks, and explicit default-parameter assertions.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave23 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-lstnet-long-direct`
- `torch-lstnet-wide-direct`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_local_lstnet_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_local_lstnet_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave23_lstnet_preset_defaults`

Expected: failures for missing wave23 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Reuse the existing direct LSTNet factory and add preset IDs with these overrides:
- Long: `lags=192`
- Wide: `cnn_channels=32`, `rnn_hidden=64`

Preserve the existing kernel, skip, highway, dropout, and trainer defaults.

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_local_lstnet_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_local_lstnet_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave23_lstnet_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
