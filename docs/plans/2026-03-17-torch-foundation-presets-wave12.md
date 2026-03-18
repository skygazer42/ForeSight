# Torch Foundation Presets Wave12 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add another batch of high-capacity `deep` and `wide` presets for foundational Torch local forecasting architectures.

**Architecture:** Reuse the existing local model factories and expose more trainable preset IDs by changing only default depth/width parameters. Lock the additions with registry presence tests, optional-dependency coverage tests, and explicit assertions for key default values.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave12 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-mlp-{deep,wide}-direct`
- `torch-lstm-{deep,wide}-direct`
- `torch-gru-{deep,wide}-direct`
- `torch-tcn-{deep,wide}-direct`
- `torch-nbeats-{deep,wide}-direct`
- `torch-transformer-{deep,wide}-direct`
- `torch-cnn-{deep,wide}-direct`
- `torch-resnet1d-{deep,wide}-direct`
- `torch-wavenet-{deep,wide}-direct`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: failures for missing wave12 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Add `deep` and `wide` variants with these overrides:
- MLP: `hidden_sizes=(64, 64, 64)`, `hidden_sizes=(128, 128)`
- LSTM: `num_layers=2`, `dropout=0.1`, `hidden_size=128`
- GRU: `num_layers=2`, `dropout=0.1`, `hidden_size=128`
- TCN: `channels=(16, 16, 16, 16)`, `channels=(32, 32, 32)`
- N-BEATS: `num_blocks=5`, `layer_width=128`
- Transformer: `num_layers=4`, `d_model=128`, `nhead=8`, `dim_feedforward=512`
- CNN: `channels=(32, 32, 32, 32)`, `channels=(64, 64, 64)`
- ResNet1D: `num_blocks=6`, `channels=64`
- WaveNet: `num_layers=8`, `channels=64`

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 3: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
