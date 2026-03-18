# Torch TinyTimeMixer Direct Wave 7 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new trainable Torch local model family, `TinyTimeMixer`-style direct forecasters, so the package gains another lightweight patch-mixer baseline without another cross-cutting trainer-config expansion.

**Architecture:** Implement a compact patch-based mixer in `src/foresight/models/torch_nn.py` that reuses the shared Torch training loop and validation messages. Expose it through a dedicated runtime factory and catalog entries in the local Torch catalog, including deeper/wider preset variants, then lock behavior with registry, optional-dependency, and validation tests.

**Tech Stack:** Python, NumPy, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for TinyTimeMixer registry/runtime exposure

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_torch_nn_validation_messages.py`

**Step 1: Write the failing tests**

- Add the new model keys to the optional Torch registration coverage list.
- Add a smoke test case for `torch-tinytimemixer-direct` in the Torch optional-deps smoke matrix.
- Add a registry assertion proving the model spec exists and exposes the intended TinyTimeMixer parameter help strings.
- Add a structural validation case proving invalid `patch_len` reuses `_PATCH_LEN_MIN_MSG`.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py tests/test_models_registry.py tests/test_torch_nn_validation_messages.py`

Expected: FAIL because the model key/factory/function do not exist yet.

### Task 2: Implement TinyTimeMixer direct forecaster and runtime wiring

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/catalog/torch_local.py`
- Modify: `src/foresight/models/__init__.py`

**Step 1: Write minimal implementation**

- Add `torch_tinytimemixer_direct_forecast(...)` to `torch_nn.py`.
- Use patchification over the lag window, a small stack of patch-token mixer blocks, and the shared `_train_loop(...)`.
- Reuse shared validation messages for `lags`, `patch_len`, `d_model`, `num_blocks`, and `dropout`.
- Add `_factory_torch_tinytimemixer_direct(...)` to `runtime.py`.
- Register the base model plus deep/wide preset variants in the local Torch catalog.
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
