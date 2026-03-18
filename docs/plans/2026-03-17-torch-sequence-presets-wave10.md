# Torch Sequence Presets Wave 10 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add more trainable Torch local preset variants for sequence-model families that currently only expose base configurations.

**Architecture:** Extend the local Torch catalog in `src/foresight/models/catalog/torch_local.py` with `deep` and `wide` variants for FNet, gMLP, linear-attention, Inception, Mamba, RWKV, Hyena, and Dilated-RNN. Keep the work strictly catalog-level, then verify the new keys and representative defaults through optional-deps and registry tests.

**Tech Stack:** Python, pytest, ruff

---

### Task 1: Add failing tests for sequence preset variants

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

- Add a new Torch local key coverage tuple for the new sequence preset family keys.
- Assert the new keys are covered by optional-dependency and registry paths.
- Add registry assertions for representative preset defaults, proving deeper variants increase layer/block counts and wider variants increase hidden dimensions or channels.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: FAIL because the new preset keys are not registered yet.

### Task 2: Register new preset variants

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Write minimal implementation**

- Add `deep` and `wide` catalog entries for:
  - `torch-fnet-*`
  - `torch-gmlp-*`
  - `torch-linear-attn-*`
  - `torch-inception-*`
  - `torch-mamba-*`
  - `torch-rwkv-*`
  - `torch-hyena-*`
  - `torch-dilated-rnn-*`
- Reuse the existing factories and param-help dictionaries.
- Follow the same description and preset naming patterns as other existing deep/wide local Torch families.

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: PASS

### Task 3: Run targeted verification

**Files:**
- Verify: `tests/test_models_optional_deps_torch.py`
- Verify: `tests/test_models_registry.py`

**Step 1: Run focused verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: PASS

**Step 2: Run lint on changed files**

Run: `ruff check src/foresight/models/catalog/torch_local.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: PASS
