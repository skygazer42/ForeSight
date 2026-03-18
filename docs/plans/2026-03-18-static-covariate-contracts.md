# Static Covariate Contracts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Promote `static_cols` from a model-specific Torch-global parameter into a shared data contract that is recognized by covariate parsing, `to_long`, and the long-format forecast/eval service entrypoints.

**Architecture:** Keep the current numeric `static_cols` runtime behavior in Torch global models, but formalize the parameter earlier in the stack. Extend `CovariateSpec` and param normalization to parse `static_cols`, let `to_long` carry static columns and attrs without treating them as time-varying future covariates, and add service-layer validation so unsupported models fail early with clear messages.

**Tech Stack:** Python, pytest, pandas, numpy, service-layer forecasting/evaluation APIs

---

### Task 1: Add failing contract and formatting tests

**Files:**
- Modify: `tests/test_contracts_covariates.py`
- Modify: `tests/test_contracts_params.py`
- Modify: `tests/test_data_to_long.py`

**Step 1: Write failing contract tests for `static_cols`**

```python
spec = resolve_model_param_covariates(
    {"x_cols": "promo", "historic_x_cols": ("stock",), "static_cols": ("store_size",)}
)
assert spec.static_cols == ("store_size",)
```

Also add a `normalize_static_cols(...)` test for raw values and model-param dicts.

**Step 2: Write a failing `to_long(...)` static-cols test**

```python
out = to_long(
    df,
    time_col="week",
    y_col="sales",
    id_cols=("store",),
    static_cols=("store_size",),
)
```

Assert the column is preserved and `out.attrs["static_cols"] == ("store_size",)`.

**Step 3: Write a failing `to_long(..., prepare=True)` static-cols test**

Use a gapped time index and assert the prepared output still contains a per-series constant `store_size` column after regularization.

**Step 4: Run red**

Run: `PYTHONPATH=src pytest -q tests/test_contracts_covariates.py tests/test_contracts_params.py tests/test_data_to_long.py`
Expected: FAIL because shared contracts and `to_long` do not yet parse or preserve `static_cols`.

### Task 2: Add failing forecast/eval service tests

**Files:**
- Create: `tests/test_static_covariate_services.py`

**Step 1: Write a failing forecast service test with a dummy global model**

Register a temporary global test model whose `factory` expects `static_cols` support and uses `store_size` from `long_df` to emit deterministic predictions.

```python
pred = forecast_model_long_df(
    model=key,
    long_df=long_df,
    horizon=2,
    model_params={"static_cols": ("store_size",)},
)
```

Assert forecast succeeds when the model advertises `supports_static_cols`.

**Step 2: Write a failing eval service test**

Use the same dummy global model with `eval_model_long_df(...)` and assert metrics are returned when `static_cols` is supplied.

**Step 3: Write a failing unsupported-model validation test**

Use a model spec that does **not** advertise `supports_static_cols`, call `forecast_model_long_df(...)` with `model_params={"static_cols": ("store_size",)}`, and assert a clear `ValueError` mentioning `static_cols`.

**Step 4: Run red**

Run: `PYTHONPATH=src pytest -q tests/test_static_covariate_services.py`
Expected: FAIL because service-layer normalization/validation does not yet understand `static_cols`.

### Task 3: Implement shared `static_cols` support

**Files:**
- Modify: `src/foresight/contracts/covariates.py`
- Modify: `src/foresight/contracts/params.py`
- Modify: `src/foresight/contracts/__init__.py`
- Modify: `src/foresight/data/format.py`
- Modify: `src/foresight/services/forecasting.py`
- Modify: `src/foresight/services/evaluation.py`

**Step 1: Extend `CovariateSpec` and model-param parsing**

Add:

```python
static_cols: tuple[str, ...] = ()
```

and thread `static_cols` through `resolve_covariate_roles(...)` / `resolve_model_param_covariates(...)`.

Keep `all_x_cols` scoped to historic/future dynamic covariates so existing code paths do not misclassify static columns as time-varying exogenous features.

**Step 2: Add param normalization for static columns**

In `contracts/params.py`, implement:

```python
def normalize_static_cols(raw: Any) -> tuple[str, ...]:
    ...
```

and export it through `contracts/__init__.py`.

**Step 3: Teach `to_long(...)` to preserve static columns**

Add a `static_cols=` keyword, copy those columns into the long frame, set `out.attrs["static_cols"]`, and preserve static per-series values across `prepare=True` regularization by mapping a validated per-`unique_id` constant lookup back onto the prepared frame.

**Step 4: Add service-layer static-cols validation**

In forecast/eval long-format services:
- normalize `static_cols` from `model_params`
- fail early when `static_cols` are passed to models without `supports_static_cols`
- keep passing `model_params` through unchanged so already-supported Torch global models continue to work

### Task 4: Verify focused and adjacent behavior

**Files:**
- Modify: `src/foresight/contracts/covariates.py`
- Modify: `src/foresight/contracts/params.py`
- Modify: `src/foresight/data/format.py`
- Modify: `src/foresight/services/forecasting.py`
- Modify: `src/foresight/services/evaluation.py`
- Modify: `tests/test_contracts_covariates.py`
- Modify: `tests/test_contracts_params.py`
- Modify: `tests/test_data_to_long.py`
- Create: `tests/test_static_covariate_services.py`

**Step 1: Run focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_contracts_covariates.py tests/test_contracts_params.py tests/test_data_to_long.py tests/test_static_covariate_services.py`
Expected: PASS.

**Step 2: Run adjacent API regression**

Run: `PYTHONPATH=src pytest -q tests/test_forecast_api.py::test_forecast_model_long_df_supports_global_models_with_future_covariates tests/test_eval_local_xreg.py::test_eval_model_long_df_supports_generic_local_xreg_models_with_future_x_cols tests/test_models_optional_deps_torch.py::test_torch_global_models_support_static_covariates_when_installed`
Expected: PASS.

**Step 3: Run syntax verification**

Run: `python -m py_compile src/foresight/contracts/covariates.py src/foresight/contracts/params.py src/foresight/contracts/__init__.py src/foresight/data/format.py src/foresight/services/forecasting.py src/foresight/services/evaluation.py tests/test_contracts_covariates.py tests/test_contracts_params.py tests/test_data_to_long.py tests/test_static_covariate_services.py`
Expected: PASS with no output.

**Step 4: Run lint verification**

Run: `ruff check src/foresight/contracts/covariates.py src/foresight/contracts/params.py src/foresight/contracts/__init__.py src/foresight/data/format.py src/foresight/services/forecasting.py src/foresight/services/evaluation.py tests/test_contracts_covariates.py tests/test_contracts_params.py tests/test_data_to_long.py tests/test_static_covariate_services.py`
Expected: PASS.
