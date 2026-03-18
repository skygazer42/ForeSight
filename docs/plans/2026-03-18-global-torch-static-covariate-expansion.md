# Global Torch Static Covariate Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand `static_cols` runtime support from the initial TFT/Informer/Autoformer path to additional global torch model families that already consume multi-channel panel inputs during training and inference.

**Architecture:** Reuse the existing `_build_panel_dataset(...)` static covariate broadcast logic and thread `static_cols` through selected global torch predictor/factory pairs. Keep catalog capability exposure synchronized with implementation by deriving `static_cols` metadata from factory signatures instead of hard-coding a small allowlist.

**Tech Stack:** Python, pandas, NumPy, PyTorch, pytest, Ruff

---

### Task 1: Lock the expected public surface in tests

**Files:**
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`

**Step 1: Write the failing test**

Add registry assertions for representative global torch families that should expose `supports_static_cols=True` after this batch, covering at least one transformer-like model, one mixer/convolutional model, and one recurrent family.

Extend the optional-deps torch smoke test so representative newly-supported global models can be instantiated with `static_cols=("store_size",)` and produce finite predictions.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py::test_model_spec_capabilities_reflect_model_family_support tests/test_models_optional_deps_torch.py::test_torch_global_models_support_static_covariates_when_installed`

Expected: FAIL because the new model specs still report `supports_static_cols=False` and/or the chosen factories do not accept `static_cols`.

### Task 2: Thread static covariates through selected global torch model families

**Files:**
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

Add `static_cols: Any = ()` to the selected public global torch factories and `static_cols: Any` to their private `_predict_torch_*` helpers.

Normalize `static_cols` alongside `x_cols` and pass the normalized tuple into `_build_panel_dataset(...)`.

Only apply this to model families whose forward path already consumes the multi-channel panel tensor rather than ignoring extra covariates.

**Step 2: Run focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_torch_global_models_support_static_covariates_when_installed`

Expected: PASS

### Task 3: Keep catalog/capabilities aligned automatically

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`
- Test: `tests/test_models_registry.py`

**Step 1: Write minimal implementation**

Replace the fixed `static_covariate_factories` allowlist with signature-based detection so every global torch catalog entry whose factory exposes `static_cols` receives the default param and help text consistently.

**Step 2: Run focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py::test_model_spec_capabilities_reflect_model_family_support`

Expected: PASS

### Task 4: Verify the batch end-to-end

**Files:**
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/catalog/torch_global.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`

**Step 1: Run targeted regressions**

Run: `PYTHONPATH=src pytest -q tests/test_torch_global_static_covariates.py tests/test_models_registry.py::test_model_spec_capabilities_reflect_model_family_support tests/test_models_optional_deps_torch.py::test_torch_global_models_support_static_covariates_when_installed tests/test_static_covariate_services.py`

Expected: PASS

**Step 2: Run static verification**

Run: `python -m py_compile src/foresight/models/torch_global.py src/foresight/models/catalog/torch_global.py tests/test_models_registry.py tests/test_models_optional_deps_torch.py`

Run: `ruff check src/foresight/models/torch_global.py src/foresight/models/catalog/torch_global.py tests/test_models_registry.py tests/test_models_optional_deps_torch.py`

Expected: PASS

**Step 3: Commit**

```bash
git add docs/plans/2026-03-18-global-torch-static-covariate-expansion.md \
  src/foresight/models/torch_global.py \
  src/foresight/models/catalog/torch_global.py \
  tests/test_models_registry.py \
  tests/test_models_optional_deps_torch.py
git commit -m "feat: expand torch global static covariate coverage"
```
