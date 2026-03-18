# Torch Recurrent Stateful Presets Wave13 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `deep` and `wide` local Torch presets for recurrent and stateful time-series architectures that still only expose a base config.

**Architecture:** Reuse the existing local recurrent/stateful factories and publish more trainable preset IDs by changing only depth or width defaults. Verify the additions through registry coverage tests, optional-dependency coverage tests, and explicit default-parameter assertions.

**Tech Stack:** Python, PyTorch model registry, pytest, ruff

---

### Task 1: Add failing regression coverage for wave13 presets

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

Add new key tuples and default-parameter assertions for:
- `torch-bilstm-{deep,wide}-direct`
- `torch-bigru-{deep,wide}-direct`
- `torch-attn-gru-{deep,wide}-direct`
- `torch-lmu-{deep,wide}-direct`
- `torch-ltc-{deep,wide}-direct`
- `torch-cfc-{deep,wide}-direct`
- `torch-xlstm-{deep,wide}-direct`
- `torch-griffin-{deep,wide}-direct`
- `torch-hawk-{deep,wide}-direct`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: failures for missing wave13 preset keys/defaults.

### Task 2: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Add minimal catalog entries**

Add `deep` and `wide` variants with these overrides:
- BiLSTM: `num_layers=2`, `dropout=0.1`, `hidden_size=128`
- BiGRU: `num_layers=2`, `dropout=0.1`, `hidden_size=128`
- Attn-GRU: `num_layers=2`, `hidden_size=128`
- LMU: `num_layers=2`, `d_model=128`, `memory_dim=64`
- LTC: `num_layers=2`, `hidden_size=128`
- CfC: `num_layers=2`, `hidden_size=128`, `backbone_hidden=256`
- xLSTM: `num_layers=2`, `hidden_size=128`
- Griffin: `num_layers=2`, `hidden_size=128`
- Hawk: `num_layers=2`, `hidden_size=128`

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass.

**Step 3: Run lint**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: pass with no diagnostics.
