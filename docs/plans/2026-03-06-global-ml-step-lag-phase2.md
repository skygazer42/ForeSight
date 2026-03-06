# Global ML Step-Lag Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three more scikit-learn global/panel forecasting models on top of the existing step-lag pipeline: `decision-tree-step-lag-global`, `bagging-step-lag-global`, and `gbrt-step-lag-global`.

**Architecture:** Reuse the shared point-forecast panel helper in `src/foresight/models/global_regression.py` so each new algorithm only needs estimator-specific parameter parsing and a small sklearn model factory. Register each model in `src/foresight/models/registry.py` with defaults and param help aligned with the matching local scikit-learn models, then cover them through smoke, interface, and missing-dependency tests.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest, scikit-learn.

---

### Task 1: Add failing tests for the new global model keys

**Files:**
- Modify: `tests/test_models_global_regression_smoke.py`
- Modify: `tests/test_models_global_interface.py`
- Modify: `tests/test_models_optional_deps_ml.py`

**Step 1: Write the failing test**

Add smoke coverage for:
- `decision-tree-step-lag-global`
- `bagging-step-lag-global`
- `gbrt-step-lag-global`

Add interface assertions so these keys are marked `interface="global"` and `requires=("ml",)`.

Add missing-dependency coverage so `make_global_forecaster(...)` raises `ImportError` when `sklearn` is absent.

**Step 2: Run test to verify it fails**

Run:
- `pytest -q tests/test_models_global_regression_smoke.py`
- `pytest -q tests/test_models_global_interface.py tests/test_models_optional_deps_ml.py`

Expected: FAIL because the new keys are not registered yet.

### Task 2: Implement the new global sklearn forecasters

**Files:**
- Modify: `src/foresight/models/global_regression.py`

**Step 1: Write minimal implementation**

Add:
- `decision_tree_step_lag_global_forecaster()`
- `bagging_step_lag_global_forecaster()`
- `gbrt_step_lag_global_forecaster()`

Each should:
- validate/normalize parameters
- build the sklearn estimator
- call the existing shared global point-model helper
- mirror the local model defaults where reasonable

### Task 3: Register the new model specs

**Files:**
- Modify: `src/foresight/models/registry.py`

**Step 1: Write minimal implementation**

Import the new forecaster factories and add three `ModelSpec` entries:
- `decision-tree-step-lag-global`
- `bagging-step-lag-global`
- `gbrt-step-lag-global`

Each spec should:
- set `requires=("ml",)`
- set `interface="global"`
- include lag-derived/global panel params
- expose estimator-specific params consistent with the local equivalents

### Task 4: Verify and update docs

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`

**Step 1: Run tests**

Run:
- `pytest -q tests/test_models_global_regression_smoke.py tests/test_models_global_interface.py tests/test_models_optional_deps_ml.py tests/test_models_registry_more_models.py`

**Step 2: Update docs**

Add a short README mention and an `Unreleased` changelog note for the three new global scikit-learn models.
