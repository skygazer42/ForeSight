# Global ML Step-Lag Phase 5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three more scikit-learn global/panel forecasting models on top of the existing step-lag pipeline: `huber-step-lag-global`, `quantile-step-lag-global`, and `sgd-step-lag-global`.

**Architecture:** Reuse the shared point-forecast panel helper in `src/foresight/models/global_regression.py`. Each model only needs estimator-specific parameter parsing plus a small sklearn estimator factory. Register the new keys in `src/foresight/models/registry.py`, then extend smoke/interface/missing-dependency coverage and brief docs.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest, scikit-learn.

---

### Task 1: Add failing tests for the new global model keys

**Files:**
- Modify: `tests/test_models_global_regression_smoke.py`
- Modify: `tests/test_models_global_interface.py`
- Modify: `tests/test_models_optional_deps_ml.py`

**Step 1: Write the failing test**

Add smoke coverage for:
- `huber-step-lag-global`
- `quantile-step-lag-global`
- `sgd-step-lag-global`

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
- `huber_step_lag_global_forecaster()`
- `quantile_step_lag_global_forecaster()`
- `sgd_step_lag_global_forecaster()`

Each should:
- validate and normalize parameters
- construct the sklearn estimator
- call the shared global point-model helper
- mirror the matching local model defaults where practical

### Task 3: Register the new model specs

**Files:**
- Modify: `src/foresight/models/registry.py`

**Step 1: Write minimal implementation**

Import the new forecaster factories and add three `ModelSpec` entries:
- `huber-step-lag-global`
- `quantile-step-lag-global`
- `sgd-step-lag-global`

Each spec should:
- set `requires=("ml",)`
- set `interface="global"`
- include lag-derived/global panel params
- expose estimator-specific params consistent with the local equivalents

### Task 4: Verify and update docs

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`

**Step 1: Run verification**

Run:
- `ruff check src/foresight/models/global_regression.py src/foresight/models/registry.py tests/test_models_global_regression_smoke.py tests/test_models_global_interface.py tests/test_models_optional_deps_ml.py`
- `python -m compileall -q src/foresight/models src/foresight/features`
- `pytest -q tests/test_models_global_regression_smoke.py tests/test_models_global_interface.py tests/test_models_optional_deps_ml.py tests/test_models_registry_more_models.py`

**Step 2: Update docs**

Add a short README mention and an `Unreleased` changelog note for the three new global scikit-learn models.
