# Global XGBoost Step-Lag Phase 7 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three more global/panel XGBoost forecasting models on top of the existing step-lag pipeline: `xgb-dart-step-lag-global`, `xgb-linear-step-lag-global`, and `xgbrf-step-lag-global`.

**Architecture:** Reuse the existing global point-forecast panel pipeline in `src/foresight/models/global_regression.py`. Factor the current XGBoost global implementation just enough to support alternate booster/estimator choices, then register the new keys in `src/foresight/models/registry.py` and extend smoke/interface/xgboost-optional-dependency coverage plus brief docs.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest, xgboost.

---

### Task 1: Add failing tests for the new global model keys

**Files:**
- Modify: `tests/test_models_global_regression_smoke.py`
- Modify: `tests/test_models_global_interface.py`
- Modify: `tests/test_models_optional_deps_xgb.py`

**Step 1: Write the failing test**

Add smoke coverage for:
- `xgb-dart-step-lag-global`
- `xgb-linear-step-lag-global`
- `xgbrf-step-lag-global`

Add interface assertions so these keys are marked `interface="global"` and `requires=("xgb",)`.

Add missing-dependency coverage so `make_global_forecaster(...)` raises `ImportError` when `xgboost` is absent.

**Step 2: Run test to verify it fails**

Run:
- `pytest -q tests/test_models_global_regression_smoke.py`
- `pytest -q tests/test_models_global_interface.py tests/test_models_optional_deps_xgb.py`

Expected: FAIL because the new keys are not registered yet.

### Task 2: Implement the new global XGBoost forecasters

**Files:**
- Modify: `src/foresight/models/global_regression.py`

**Step 1: Write minimal implementation**

Add:
- `xgb_dart_step_lag_global_forecaster()`
- `xgb_linear_step_lag_global_forecaster()`
- `xgbrf_step_lag_global_forecaster()`

Keep the existing `xgb-step-lag-global` behavior intact while reusing as much of its training/prediction path as possible.

### Task 3: Register the new model specs

**Files:**
- Modify: `src/foresight/models/registry.py`

**Step 1: Write minimal implementation**

Import the new forecaster factories and add three `ModelSpec` entries:
- `xgb-dart-step-lag-global`
- `xgb-linear-step-lag-global`
- `xgbrf-step-lag-global`

Each spec should:
- set `requires=("xgb",)`
- set `interface="global"`
- include lag-derived/global panel params
- expose booster-specific params consistent with the local XGBoost variants

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

Add a short README mention and an `Unreleased` changelog note for the three new global XGBoost models.
