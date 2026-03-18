# Torch Global Presets Wave19 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `deep` and `wide` global Torch presets for the shared Seq2Seq encoder-decoder panel forecasting family.

**Architecture:** Extend the global Torch catalog with additional `seq2seq-*global` preset IDs that reuse the existing shared seq2seq global forecaster factory and only override recurrent capacity defaults. Protect the new surface with registration checks, optional-dependency coverage checks, and explicit default-parameter assertions.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave19 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-seq2seq-lstm-{deep,wide}-global`
- `torch-seq2seq-gru-{deep,wide}-global`
- `torch-seq2seq-attn-lstm-{deep,wide}-global`
- `torch-seq2seq-attn-gru-{deep,wide}-global`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave19_global_preset_defaults`

Expected: failures for missing wave19 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`

**Step 1: Add minimal catalog entries**

Reuse the existing global seq2seq factory and add preset IDs with these overrides:
- Deep presets: `num_layers=2`, `dropout=0.1`
- Wide presets: `hidden_size=128`

Preserve the existing base `cell`, `attention`, `teacher_forcing`, and other training defaults for each of the four seq2seq variants.

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave19_global_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_global.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
