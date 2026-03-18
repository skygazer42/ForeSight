# Torch Global Static Covariates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-phase `static_cols` interface for the shared Torch global TFT/Informer/Autoformer family so panel-level static covariates can be passed once per series from `long_df` and reused during training/inference.

**Architecture:** Keep the existing Torch global training code shape intact by extending the shared `_build_panel_dataset(...)` path used by the TFT/Informer/Autoformer global forecasters. `static_cols` will be numeric-only in this first phase, validated as series-constant features, then broadcast across each training/prediction window as additional per-step channels. Expose the capability centrally for specs backed by these shared factories instead of editing every catalog entry by hand.

**Tech Stack:** Python, pytest, pandas, numpy, PyTorch-optional global model registry

---

### Task 1: Add failing dataset-level tests

**Files:**
- Create: `tests/test_torch_global_static_covariates.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write a failing builder test that accepts `static_cols` and broadcasts them**

```python
x_train, *_ = torch_global._build_panel_dataset(
    long_df,
    cutoff=cutoff,
    horizon=3,
    context_length=8,
    x_cols=("promo",),
    static_cols=("store_size",),
    normalize=False,
    max_train_size=None,
    sample_step=2,
    add_time_features=False,
)
```

Assert that the static feature channel is present and constant across every timestep inside one sample window.

**Step 2: Write a failing builder test that allows missing future `static_cols` values**

```python
future_rows["store_size"] = np.nan
```

Assert that the dataset still builds because the static value is recovered from the series-level observed rows.

**Step 3: Write a failing builder test that rejects non-static per-series values**

```python
long_df.loc[mask, "store_size"] = np.linspace(1.0, 2.0, mask.sum())
```

Assert `ValueError` with a message that identifies the offending `static_cols` column.

**Step 4: Run red**

Run: `PYTHONPATH=src pytest -q tests/test_torch_global_static_covariates.py`
Expected: FAIL because `_build_panel_dataset(...)` does not yet accept or validate `static_cols`.

### Task 2: Add failing registry/smoke coverage

**Files:**
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `README.md`

**Step 1: Add a failing capability assertion**

```python
tft_global = get_model_spec("torch-tft-global")
assert tft_global.capabilities["supports_static_cols"] is True
```

Also extend the documented capability flag list expectation to include `supports_static_cols`.

**Step 2: Add a failing representative smoke test**

```python
g = make_global_forecaster(
    "torch-tft-global",
    context_length=32,
    epochs=2,
    batch_size=32,
    x_cols=("promo",),
    static_cols=("store_size",),
)
```

Build a panel where future rows omit `store_size`, then assert prediction succeeds and emits finite `yhat`.

**Step 3: Run red**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py::test_model_spec_capabilities_reflect_model_family_support tests/test_models_registry.py::test_readme_documents_all_model_capability_flags tests/test_models_optional_deps_torch.py::test_torch_global_models_support_static_covariates_when_installed`
Expected: FAIL because capabilities and runtime do not yet expose `static_cols`.

### Task 3: Implement Torch global `static_cols`

**Files:**
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/catalog/torch_global.py`
- Modify: `src/foresight/models/specs.py`
- Modify: `README.md`

**Step 1: Extend shared normalization and dataset building**

Add:

```python
def _normalize_static_cols(static_cols: Any) -> tuple[str, ...]:
    return _normalize_x_cols(static_cols)
```

Extend `_build_panel_dataset(...)` to:
- accept `static_cols`
- verify columns exist
- derive one numeric static vector per series from non-null unique values
- reject missing or non-constant series-level values
- broadcast the static vector across `seq_len`

**Step 2: Thread `static_cols` through the shared Torch global TFT/Informer/Autoformer factories**

Update the relevant shared/global factory functions so user-supplied `static_cols` reaches `_build_panel_dataset(...)`.

**Step 3: Expose capability centrally in the Torch global catalog for the supported family**

After the catalog is built, post-process specs whose factories are backed by the shared global TFT/Informer/Autoformer runtime:

```python
param_help.setdefault("static_cols", "Optional static covariate columns from long_df (series-constant)")
default_params.setdefault("static_cols", ())
```

This avoids editing every individual spec by hand and automatically covers the supported family plus preset clones.

**Step 4: Add capability inference**

In `ModelSpec.capabilities`, infer:

```python
supports_static_cols = "static_cols" in self.param_help
```

and include it in the capabilities payload.

**Step 5: Update README capability docs**

Document `supports_static_cols` in the model capability table.

### Task 4: Verify focused and adjacent behavior

**Files:**
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/catalog/torch_global.py`
- Modify: `src/foresight/models/specs.py`
- Modify: `tests/test_torch_global_static_covariates.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`

**Step 1: Run focused test coverage**

Run: `PYTHONPATH=src pytest -q tests/test_torch_global_static_covariates.py tests/test_models_registry.py::test_model_spec_capabilities_reflect_model_family_support tests/test_models_registry.py::test_readme_documents_all_model_capability_flags tests/test_models_optional_deps_torch.py::test_torch_global_models_support_static_covariates_when_installed`
Expected: PASS.

**Step 2: Run adjacent Torch global smoke**

Run: `PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py::test_torch_global_models_smoke_when_installed tests/test_global_forecaster_api.py::test_global_forecaster_object_supports_fit_then_predict`
Expected: PASS.

**Step 3: Run syntax verification**

Run: `python -m py_compile src/foresight/models/torch_global.py src/foresight/models/catalog/torch_global.py src/foresight/models/specs.py tests/test_torch_global_static_covariates.py tests/test_models_registry.py tests/test_models_optional_deps_torch.py`
Expected: PASS with no output.

**Step 4: Run lint verification**

Run: `ruff check src/foresight/models/torch_global.py src/foresight/models/catalog/torch_global.py src/foresight/models/specs.py tests/test_torch_global_static_covariates.py tests/test_models_registry.py tests/test_models_optional_deps_torch.py`
Expected: PASS.
