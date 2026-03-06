# Global XGBoost Step-Lag Phase 9 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the remaining low-risk global/panel XGBoost objective variants on the shared step-lag pipeline: `xgb-gamma-step-lag-global` and `xgb-logistic-step-lag-global`.

**Architecture:** Reuse the shared global XGBoost step-lag helper in `src/foresight/models/global_regression.py`, adding only thin wrappers for the two missing objectives. Extend smoke tests with objective-appropriate long-format panel fixtures so `gamma` sees strictly-positive targets and `logistic` sees `[0,1]` targets, then register the keys in `src/foresight/models/registry.py` and update brief docs.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest, xgboost.

---

### Task 1: Add failing tests for the new global model keys

**Files:**
- Modify: `tests/test_models_global_regression_smoke.py`
- Modify: `tests/test_models_global_interface.py`
- Modify: `tests/test_models_optional_deps_xgb.py`

**Step 1: Write the failing test**

Add smoke coverage for:
- `xgb-gamma-step-lag-global`
- `xgb-logistic-step-lag-global`

Use objective-safe panel fixtures:
- strictly positive `y` for gamma
- `y in [0,1]` for logistic

Add interface assertions so these keys are marked `interface="global"` and `requires=("xgb",)`.

Add missing-dependency coverage so `make_global_forecaster(...)` raises `ImportError` when `xgboost` is absent.

**Step 2: Run test to verify it fails**

Run:
- `pytest -q tests/test_models_global_regression_smoke.py -k 'xgb_gamma_step_lag_global_smoke or xgb_logistic_step_lag_global_smoke'`
- `pytest -q tests/test_models_global_interface.py tests/test_models_optional_deps_xgb.py -k 'xgb or global_models_are_marked_interface_global or make_forecaster_rejects_global_models'`

Expected: FAIL because the new keys are not registered yet.

### Task 2: Implement the new global XGBoost forecasters

**Files:**
- Modify: `src/foresight/models/global_regression.py`

**Step 1: Write minimal implementation**

Add:
- `xgb_gamma_step_lag_global_forecaster()`
- `xgb_logistic_step_lag_global_forecaster()`

Both should be thin wrappers over the existing shared XGBoost global helper, using:
- `reg:gamma`
- `reg:logistic`

Keep all existing global XGBoost variants unchanged.

### Task 3: Register the new model specs

**Files:**
- Modify: `src/foresight/models/registry.py`

**Step 1: Write minimal implementation**

Import the new forecaster factories and add two `ModelSpec` entries:
- `xgb-gamma-step-lag-global`
- `xgb-logistic-step-lag-global`

Each spec should:
- set `requires=("xgb",)`
- set `interface="global"`
- include lag-derived/global panel params
- document the target-domain expectation in the description

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

Add a short README mention and an `Unreleased` changelog note for the two new global XGBoost models.
