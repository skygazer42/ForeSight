# Torch Recursive RNN Presets Wave25 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semantic `deep` and `wide` recursive Torch presets for the remaining DeepAR and QRNN local families.

**Architecture:** Extend the local Torch catalog with four additional recursive preset IDs that reuse the existing DeepAR and QRNN recursive factories and only override recurrent depth, dropout, or hidden width defaults. Protect the new surface with registration checks, optional-dependency coverage checks, and explicit default-parameter assertions.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave25 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-deepar-deep-recursive`
- `torch-deepar-wide-recursive`
- `torch-qrnn-deep-recursive`
- `torch-qrnn-wide-recursive`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_recursive_rnn_preset_torch_local_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_recursive_rnn_preset_torch_local_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave25_recursive_rnn_preset_defaults`

Expected: failures for missing wave25 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Reuse the existing recursive RNN factories and add preset IDs with these overrides:
- Deep: `num_layers=2`, `dropout=0.1`
- Wide: `hidden_size=64`

Preserve the existing lag window, quantile/default loss behavior, and trainer defaults.

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_recursive_rnn_preset_torch_local_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_recursive_rnn_preset_torch_local_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave25_recursive_rnn_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
