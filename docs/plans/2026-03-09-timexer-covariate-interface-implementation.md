# TimeXer Covariate-Aware Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the local/global covariate-aware forecasting interface needed for a future `TimeXer` family, without forcing the `TimeXer` model itself into the same change set.

**Architecture:** Keep the current registry-driven forecasting design, but replace hard-coded local `x_cols` model checks in `forecast_model_long_df()` and `eval_model_long_df()` with a generic capability-driven local exogenous path. Local models that support `x_cols` should be callable as `f(train, horizon, train_exog=..., future_exog=...)`, while `requires_future_covariates` becomes a first-class capability guard for both local and global interfaces.

**Tech Stack:** Python 3.10+, NumPy, Pandas, pytest

---

### Task 1: Lock Generic Local XReg Behavior In Tests

**Files:**
- Modify: `tests/test_forecast_api.py`
- Modify: `tests/test_eval_local_xreg.py`

**Step 1: Add a dummy local xreg model test for forecast**

Monkeypatch the registry with a temporary local model that:
- advertises `x_cols`
- sets `requires_future_covariates=True`
- consumes `train_exog` / `future_exog`

Require `forecast_model_long_df()` to work for that model without any model-key special casing.

**Step 2: Add a dummy local xreg model test for eval**

Require `eval_model_long_df()` to work for the same temporary local model.

**Step 3: Add capability-guard tests**

Require both local and global models with `requires_future_covariates=True` to fail clearly when `x_cols` are omitted.

**Step 4: Run RED**

```bash
PYTHONPATH=src pytest -q tests/test_forecast_api.py -k "generic_local_xreg or requires_future_covariates"
PYTHONPATH=src pytest -q tests/test_eval_local_xreg.py -k "generic_local_xreg or requires_future_covariates"
```

---

### Task 2: Implement Generic Local Covariate-Aware Callable Path

**Files:**
- Modify: `src/foresight/forecast.py`
- Modify: `src/foresight/eval_forecast.py`

**Step 1: Add a shared calling convention**

Use local forecasters as:

```python
f(train_y, horizon, train_exog=train_exog, future_exog=future_exog)
```

This must become the generic local `x_cols` path instead of hard-coded model-name branches.

**Step 2: Keep interval behavior conservative**

Keep local `interval_levels with x_cols` support limited to models that truly support it today.

**Step 3: Add capability-driven guards**

If `requires_future_covariates=True` and no `x_cols` are supplied, fail with a clear error before forecasting.

**Step 4: Run GREEN**

```bash
PYTHONPATH=src pytest -q tests/test_forecast_api.py -k "generic_local_xreg or requires_future_covariates"
PYTHONPATH=src pytest -q tests/test_eval_local_xreg.py -k "generic_local_xreg or requires_future_covariates"
```

---

### Task 3: Prepare Registry Capability Semantics For TimeXer

**Files:**
- Modify: `src/foresight/models/registry.py`
- Modify: `tests/test_models_registry.py`

**Step 1: Add a regression test for `requires_future_covariates` overrides**

Ensure `ModelSpec.capabilities` preserves explicit overrides.

**Step 2: Keep default behavior unchanged**

Models without the override should still default to `False`.

**Step 3: Verify**

```bash
PYTHONPATH=src pytest -q tests/test_models_registry.py -k requires_future_covariates
```

---

### Task 4: Check Existing XReg Surfaces Still Work

**Files:**
- Verify only

**Step 1: Run existing local xreg tests**

```bash
PYTHONPATH=src pytest -q tests/test_forecast_api.py -k "sarimax_with_future_covariates or auto_arima_with_future_covariates"
PYTHONPATH=src pytest -q tests/test_eval_local_xreg.py
```

**Step 2: Run existing global x_cols tests**

```bash
PYTHONPATH=src pytest -q tests/test_forecast_api.py -k "global_models_with_future_covariates"
```

**Step 3: Stop after interface checkpoint**

Do not implement `torch-timexer-direct` or `torch-timexer-global` in this batch.
