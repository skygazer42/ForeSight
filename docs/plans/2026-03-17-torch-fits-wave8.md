# Torch FITS Direct Wave 8 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new trainable Torch local model family, `FITS`-style direct forecasters, so the package gains a distinct frequency-interpolation baseline beyond the existing token/patch mixers.

**Architecture:** Implement a compact low-frequency interpolation model in `src/foresight/models/torch_nn.py` that transforms lag windows into truncated FFT spectra, learns a small MLP to upsample low-frequency bins into an extended horizon spectrum, then reconstructs the forecast tail with `irfft`. Expose it via a dedicated runtime factory and local Torch catalog entries with base/deep/wide presets, and lock behavior down with optional-deps, registry, and validation tests.

**Tech Stack:** Python, NumPy, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for FITS registry/runtime exposure

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_torch_nn_validation_messages.py`

**Step 1: Write the failing tests**

- Add the new model keys `torch-fits-direct`, `torch-fits-deep-direct`, and `torch-fits-wide-direct` to the optional Torch registration coverage list.
- Add a smoke case for `torch-fits-direct` to the Torch optional-deps smoke matrix.
- Add a registry assertion proving the model spec exists and exposes `low_freq_bins`, `hidden_size`, and `num_layers` help strings.
- Add a shared validation-message assertion for `_LOW_FREQ_BINS_MIN_MSG`.
- Add a structural validation case proving invalid `low_freq_bins` in `torch_fits_direct_forecast(...)` raises `_LOW_FREQ_BINS_MIN_MSG`.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py tests/test_torch_nn_validation_messages.py`

Expected: FAIL because the FITS model key/factory/function do not exist yet.

### Task 2: Implement FITS direct forecaster and runtime wiring

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/catalog/torch_local.py`
- Modify: `src/foresight/models/__init__.py`

**Step 1: Write minimal implementation**

- Add `_LOW_FREQ_BINS_MIN_MSG` to the shared Torch validation constants.
- Add `torch_fits_direct_forecast(...)` to `torch_nn.py`.
- Build the model around low-frequency FFT truncation, an MLP-based frequency interpolation head, and `irfft` reconstruction of an extended context-plus-horizon sequence.
- Reuse shared validation messages for `lags`, `hidden_size`, `num_layers`, and `dropout`.
- Add `_factory_torch_fits_direct(...)` to `runtime.py`.
- Register base/deep/wide `torch-fits-*` variants in the local Torch catalog.
- Re-export the new function in `src/foresight/models/__init__.py`.

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py tests/test_torch_nn_validation_messages.py`

Expected: PASS

### Task 3: Run targeted Torch verification

**Files:**
- Verify: `tests/test_models_optional_deps_torch.py`
- Verify: `tests/test_models_registry.py`
- Verify: `tests/test_torch_nn_validation_messages.py`

**Step 1: Run focused verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py tests/test_torch_nn_validation_messages.py`

Expected: PASS

**Step 2: Run lint on changed files**

Run: `ruff check src/foresight/models/torch_nn.py src/foresight/models/runtime.py src/foresight/models/catalog/torch_local.py src/foresight/models/__init__.py tests/test_models_optional_deps_torch.py tests/test_models_registry.py tests/test_torch_nn_validation_messages.py`

Expected: PASS
