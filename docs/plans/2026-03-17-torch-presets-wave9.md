# Torch Local Presets Wave 9 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add more high-value trainable Torch local preset variants so the package exposes deeper and wider configurations for several existing neural forecasting families without changing their core implementations.

**Architecture:** Extend the local Torch catalog in `src/foresight/models/catalog/torch_local.py` with `deep` and `wide` presets for selected families that already have meaningful depth/width parameters: TimeMixer, FreTS, FiLM, MICN, Koopa, and SAMformer. Lock the new keys and preset defaults with optional-deps and registry tests.

**Tech Stack:** Python, pytest, ruff

---

### Task 1: Add failing tests for new preset variants

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Write the failing tests**

- Extend the Torch local key coverage tuples so they expect the new `deep` and `wide` variants for:
  - `torch-timemixer-*`
  - `torch-frets-*`
  - `torch-film-*`
  - `torch-micn-*`
  - `torch-koopa-*`
  - `torch-samformer-*`
- Add registry assertions proving the new preset specs exist.
- Add registry assertions verifying a representative set of preset defaults, especially:
  - deeper variants increase `num_blocks` or `num_layers`
  - wider variants increase `d_model`, `latent_dim`, `nhead`, or related hidden widths

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py`

Expected: FAIL because the new preset keys are not registered yet.

### Task 2: Implement preset registrations

**Files:**
- Modify: `src/foresight/models/catalog/torch_local.py`

**Step 1: Write minimal implementation**

- Add `deep` and `wide` catalog entries for the targeted families.
- Reuse the existing factories and parameter help dictionaries.
- Keep the presets DRY and consistent with existing naming/description patterns already used by Perceiver, SegRNN, ModernTCN, Basisformer, WITRAN, CrossGNN, Pathformer, TimesMamba, TinyTimeMixer, and FITS.

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
