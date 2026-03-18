# Torch Global Presets Wave17 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `deep` and `wide` global Torch presets for the next batch of high-value non-transformer panel forecasting architectures that currently expose only a base global config.

**Architecture:** Extend the global Torch catalog with additional preset IDs that reuse the existing global forecaster factories and only override capacity-oriented defaults. Protect the new surface with registry presence tests, optional-dependency coverage tests, and explicit assertions for the added default values.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave17 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-tsmixer-{deep,wide}-global`
- `torch-nbeats-{deep,wide}-global`
- `torch-nhits-{deep,wide}-global`
- `torch-tcn-{deep,wide}-global`
- `torch-wavenet-{deep,wide}-global`
- `torch-resnet1d-{deep,wide}-global`
- `torch-inception-{deep,wide}-global`
- `torch-kan-{deep,wide}-global`
- `torch-scinet-{deep,wide}-global`
- `torch-etsformer-{deep,wide}-global`
- `torch-esrnn-{deep,wide}-global`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave17_global_preset_defaults`

Expected: failures for missing wave17 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`

**Step 1: Add minimal catalog entries**

Reuse the existing global factories and add preset IDs with these overrides:
- TSMixer: `num_blocks=6`, `d_model=128`, `token_mixing_hidden=256`, `channel_mixing_hidden=256`
- N-BEATS: `num_blocks=5`, `layer_width=512`
- N-HiTS: `num_blocks=8`, `layer_width=512`
- TCN: `channels=(64, 64, 64, 64)`, `channels=(128, 128, 128)`
- WaveNet: `num_layers=8`, `channels=64`
- ResNet1D: `num_blocks=6`, `channels=64`
- Inception: `num_blocks=5`, `channels=64`, `bottleneck_channels=32`
- KAN: `num_layers=4`, `d_model=128`
- SCINet: `num_stages=4`, `d_model=128`, `ffn_dim=256`
- ETSformer: `num_layers=4`, `d_model=128`, `nhead=8`, `dim_feedforward=512`
- ESRNN: `num_layers=4`, `hidden_size=128`

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_global_preset_torch_models_are_covered_by_optional_dep_paths tests/test_models_registry.py::test_global_preset_torch_models_are_registered tests/test_models_registry.py::test_torch_global_catalog_exposes_wave17_global_preset_defaults`

Expected: pass.

**Step 3: Run full verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 4: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_global.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
