# Torch Structured RNN Presets Wave24 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semantic `long` and `wide` direct Torch presets for the structured recurrent wrapper families.

**Architecture:** Extend `_make_wave1_structured_rnn_specs()` with six additional preset IDs that reuse the existing structured recurrent wrapper factory and only override lag-window or hidden-width defaults. Protect the new surface with registration checks, optional-dependency coverage checks, and explicit default-parameter assertions.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave24 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-multidim-rnn-long-direct`
- `torch-multidim-rnn-wide-direct`
- `torch-grid-lstm-long-direct`
- `torch-grid-lstm-wide-direct`
- `torch-structural-rnn-long-direct`
- `torch-structural-rnn-wide-direct`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_structured_rnn_preset_torch_local_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_structured_rnn_preset_torch_local_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave24_structured_rnn_preset_defaults`

Expected: failures for missing wave24 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Reuse the existing structured recurrent wrapper factory and add preset IDs with these overrides:
- Long: `lags=48`
- Wide: `hidden_size=64`

Preserve the existing variant selection and trainer defaults.

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_structured_rnn_preset_torch_local_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_structured_rnn_preset_torch_local_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave24_structured_rnn_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
