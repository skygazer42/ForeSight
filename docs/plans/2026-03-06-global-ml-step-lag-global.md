# Global ML Step-Lag Models Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three new global/panel scikit-learn forecasters that reuse the existing step-lag training pipeline: `ridge-step-lag-global`, `rf-step-lag-global`, and `extra-trees-step-lag-global`.

**Architecture:** Reuse the shared helpers already in `src/foresight/models/global_regression.py` for panel validation, lag-window expansion, derived lag features, optional covariates, time features, and prediction-table assembly. Keep the new forecasters as point-only regressors with the same `interface="global"` contract as `hgb-step-lag-global`, and expose them through the central registry with scikit-learn-gated metadata.

**Tech Stack:** Python 3.10+, numpy, pandas, scikit-learn, pytest.

---

### Task 1: Add failing tests for new global sklearn model keys

**Files:**
- Modify: `tests/test_models_global_interface.py`
- Modify: `tests/test_models_global_regression_smoke.py`

**Step 1: Write the failing test**

Add interface/registry assertions for:
- `ridge-step-lag-global`
- `rf-step-lag-global`
- `extra-trees-step-lag-global`

Add smoke tests that run `cross_validation_predictions_long_df(...)` for the three new models on the small panel fixture and assert the standard prediction columns exist.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_models_global_interface.py tests/test_models_global_regression_smoke.py
```

Expected:
- FAIL because the new registry keys and forecaster factories do not exist yet.

### Task 2: Implement the three sklearn global forecasters

**Files:**
- Modify: `src/foresight/models/global_regression.py`

**Step 1: Write minimal implementation**

Add:
- `ridge_step_lag_global_forecaster`
- `rf_step_lag_global_forecaster`
- `extra_trees_step_lag_global_forecaster`

Implementation rules:
- Reuse `_validate_long_df`, `_panel_step_lag_train_xy`, `_panel_step_lag_predict_X`.
- Keep the same shared params as `hgb-step-lag-global`: `lags`, `roll_windows`, `roll_stats`, `diff_lags`, `x_cols`, `add_time_features`, `id_feature`, `step_scale`, `max_train_size`, `sample_step`.
- Add only model-specific hyperparameters needed by the sklearn estimator.
- Return point forecasts only (`unique_id`, `ds`, `yhat`).

### Task 3: Register the new models

**Files:**
- Modify: `src/foresight/models/registry.py`

**Step 1: Write minimal implementation**

Import the new factories and add three `ModelSpec` entries with:
- `requires=("ml",)`
- `interface="global"`
- default params and `param_help` aligned with the estimator-specific options
- descriptions matching existing global model wording

### Task 4: Verify green

**Files:**
- (No changes)

**Step 1: Run targeted tests**

```bash
pytest -q tests/test_models_global_interface.py tests/test_models_global_regression_smoke.py
```

**Step 2: Run supporting regression coverage**

```bash
pytest -q tests/test_models_registry_more_models.py tests/test_features_tabular.py tests/test_features_time.py
```

Expected:
- PASS
