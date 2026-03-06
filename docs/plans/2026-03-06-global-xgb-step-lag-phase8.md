# Global XGBoost Step-Lag Phase 8 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add five more global/panel XGBoost forecasting models on top of the existing step-lag pipeline: `xgb-msle-step-lag-global`, `xgb-mae-step-lag-global`, `xgb-huber-step-lag-global`, `xgb-poisson-step-lag-global`, and `xgb-tweedie-step-lag-global`.

**Architecture:** Reuse the shared global XGBoost step-lag helper in `src/foresight/models/global_regression.py` so the new models differ only by objective and a small amount of objective-specific validation. Register each key in `src/foresight/models/registry.py`, expand smoke/interface/optional-dependency tests, and add concise documentation entries.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest, xgboost.

---

### Task 1: Add failing tests for the new global model keys

**Files:**
- Modify: `tests/test_models_global_regression_smoke.py`
- Modify: `tests/test_models_global_interface.py`
- Modify: `tests/test_models_optional_deps_xgb.py`

**Step 1: Write the failing test**

Add smoke coverage for:
- `xgb-msle-step-lag-global`
- `xgb-mae-step-lag-global`
- `xgb-huber-step-lag-global`
- `xgb-poisson-step-lag-global`
- `xgb-tweedie-step-lag-global`

Add interface assertions so these keys are marked `interface="global"` and `requires=("xgb",)`.

Add missing-dependency coverage so `make_global_forecaster(...)` raises `ImportError` when `xgboost` is absent.

**Step 2: Run test to verify it fails**

Run:
- `pytest -q tests/test_models_global_regression_smoke.py -k 'xgb_msle_step_lag_global_smoke or xgb_mae_step_lag_global_smoke or xgb_huber_step_lag_global_smoke or xgb_poisson_step_lag_global_smoke or xgb_tweedie_step_lag_global_smoke'`
- `pytest -q tests/test_models_global_interface.py tests/test_models_optional_deps_xgb.py -k 'xgb or global_models_are_marked_interface_global or make_forecaster_rejects_global_models'`

Expected: FAIL because the new keys are not registered yet.

### Task 2: Implement the new global XGBoost forecasters

**Files:**
- Modify: `src/foresight/models/global_regression.py`

**Step 1: Write minimal implementation**

Extend the shared helper so point-forecast models can override the XGBoost objective and optional objective-specific params.

Add:
- `xgb_msle_step_lag_global_forecaster()`
- `xgb_mae_step_lag_global_forecaster()`
- `xgb_huber_step_lag_global_forecaster()`
- `xgb_poisson_step_lag_global_forecaster()`
- `xgb_tweedie_step_lag_global_forecaster()`

Keep the existing square-error, DART, linear, random-forest, and quantile behavior intact.

### Task 3: Register the new model specs

**Files:**
- Modify: `src/foresight/models/registry.py`

**Step 1: Write minimal implementation**

Import the new forecaster factories and add five `ModelSpec` entries:
- `xgb-msle-step-lag-global`
- `xgb-mae-step-lag-global`
- `xgb-huber-step-lag-global`
- `xgb-poisson-step-lag-global`
- `xgb-tweedie-step-lag-global`

Each spec should:
- set `requires=("xgb",)`
- set `interface="global"`
- include lag-derived/global panel params
- expose objective-specific params such as `huber_slope` and `tweedie_variance_power` where needed

### Task 4: Verify and update docs

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`

**Step 1: Run verification**

Run:
- `ruff check src/foresight/models/global_regression.py src/foresight/models/registry.py tests/test_models_global_regression_smoke.py tests/test_models_global_interface.py tests/test_models_optional_deps_xgb.py`
- `python -m compileall -q src/foresight/models src/foresight/features`
- `pytest -q tests/test_models_global_regression_smoke.py tests/test_models_global_interface.py tests/test_models_optional_deps_xgb.py tests/test_models_registry_more_models.py`

**Step 2: Update docs**

Add a short README mention and an `Unreleased` changelog note for the five new global XGBoost models.
