# Torch Global Presets Wave21 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `deep` and `wide` global Torch presets for the shared global RNN backbone family.

**Architecture:** Extend the helper-driven global RNN registrations with extra preset IDs that reuse the shared `torch_rnn_global_forecaster` factory and only override recurrent capacity defaults. Protect the new surface with registration checks, optional-dependency coverage checks, and explicit default-parameter assertions.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave21 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-rnn-lstm-{deep,wide}-global`
- `torch-rnn-gru-{deep,wide}-global`
- `torch-rnn-encoder-{deep,wide}-global`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave21_global_preset_defaults`

Expected: failures for missing wave21 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Reuse the existing `_add_global_rnn` helper and add preset IDs with these overrides:
- LSTM / GRU deep: `num_layers=2`, `dropout=0.1`
- LSTM / GRU wide: `hidden_size=128`
- Encoder deep: `hidden_size=32`, `num_layers=2`, `dropout=0.1`
- Encoder wide: `hidden_size=64`

Preserve the existing `cell`, normalization, and training defaults.

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave21_global_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
