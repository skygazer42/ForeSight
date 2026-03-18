# Torch Probabilistic Presets Wave26 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semantic `long` and `wide` direct Torch presets for the remaining TimeGrad and TACTiS probabilistic wrapper families.

**Architecture:** Extend `_make_wave1_probabilistic_specs()` with four additional preset IDs that reuse the existing probabilistic wrapper factory and only override lag-window or hidden-width defaults. Protect the new surface with registration checks, optional-dependency coverage checks, and explicit default-parameter assertions.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave26 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-timegrad-long-direct`
- `torch-timegrad-wide-direct`
- `torch-tactis-long-direct`
- `torch-tactis-wide-direct`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_probabilistic_preset_torch_local_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_probabilistic_preset_torch_local_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave26_probabilistic_preset_defaults`

Expected: failures for missing wave26 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Reuse the existing probabilistic wrapper factory and add preset IDs with these overrides:
- Long: `lags=48`
- Wide: `hidden_size=64`

Preserve the existing attention-head divisibility, dropout, loss, and trainer defaults.

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_probabilistic_preset_torch_local_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_probabilistic_preset_torch_local_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave26_probabilistic_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
