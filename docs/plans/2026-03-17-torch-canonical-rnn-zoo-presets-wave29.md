# Torch Canonical RNN Zoo Presets Wave29 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semantic `long` and `wide` direct Torch presets for the canonical LSTM/GRU/QRNN entries in the `rnnpaper` and `rnnzoo` families.

**Architecture:** Extend `_make_torch_rnnpaper_specs()` and `_make_torch_rnnzoo_specs()` with additional preset IDs that reuse the existing zoo factories and only override lag-window or hidden-width defaults. Restrict the expansion to canonical `lstm`, `gru`, and `qrnn` direct keys so the parameter semantics remain honest across the selected architectures.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave29 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-rnnpaper-lstm-long-direct`
- `torch-rnnpaper-lstm-wide-direct`
- `torch-rnnpaper-gru-long-direct`
- `torch-rnnpaper-gru-wide-direct`
- `torch-rnnpaper-qrnn-long-direct`
- `torch-rnnpaper-qrnn-wide-direct`
- `torch-rnnzoo-lstm-long-direct`
- `torch-rnnzoo-lstm-wide-direct`
- `torch-rnnzoo-gru-long-direct`
- `torch-rnnzoo-gru-wide-direct`
- `torch-rnnzoo-qrnn-long-direct`
- `torch-rnnzoo-qrnn-wide-direct`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_canonical_rnn_zoo_preset_torch_local_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_canonical_rnn_zoo_preset_torch_local_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave29_canonical_rnn_zoo_preset_defaults`

Expected: failures for missing wave29 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Reuse the existing zoo factories and add preset IDs with these overrides:
- Long: `lags=48`
- Wide: `hidden_size=64`

Keep the existing dropout, auxiliary kernel/attention defaults, and trainer defaults unchanged.

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_canonical_rnn_zoo_preset_torch_local_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_canonical_rnn_zoo_preset_torch_local_models_are_registered tests/test_models_registry.py::test_torch_local_catalog_exposes_wave29_canonical_rnn_zoo_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
