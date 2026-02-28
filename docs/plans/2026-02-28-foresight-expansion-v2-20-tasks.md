# ForeSight Toolkit Expansion v2 (20 Tasks) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Further expand the `foresight` package from “baseline forecasting + simple eval” into a more complete (still lightweight-by-default) time-series benchmarking toolkit: rolling-origin CV outputs, conformal intervals, richer metrics, more classical models (incl. intermittent demand), optional ML models, better CLI ergonomics, and a README that clearly positions the toolkit relative to mainstream TS projects.

**Architecture (inspired by mainstream projects):**
- **Long/panel-friendly tabular format**: `unique_id/ds/y` (Nixtla / StatsForecast convention; Prophet uses `ds/y`).
- **Backtesting-first**: rolling-origin cross-validation producing a *predictions table* (like StatsForecast/NeuralForecast `cross_validation` output).
- **Probabilistic evaluation as a first-class citizen**: interval coverage/width + scaled interval scores (GluonTS-style evaluator metrics).
- **Optional heavy deps**: keep core to `numpy/pandas`; gate advanced models behind extras (e.g. `.[ml]`, `.[stats]`).

**Tech Stack:** Python 3.10+, `numpy`, `pandas`, `pytest`, `ruff`. Optional: `scikit-learn` (`.[ml]`), `statsmodels` (`.[stats]`).

**Note:** Per user request, **do not commit** while executing this plan.

---

### Task 1: Add a generic rolling-origin split generator

**Files:**
- Create: `src/foresight/splits.py`
- Test: `tests/test_splits.py`

**Work:**
- Implement an iterator that yields `(train_start, train_end, test_start, test_end)` for rolling-origin evaluation.
- Support `horizon`, `step_size`, `min_train_size`, optional `max_train_size` (rolling window).

**Verify:** `pytest -q tests/test_splits.py`

---

### Task 2: Extend `walk_forward` to support rolling windows (`max_train_size`)

**Files:**
- Modify: `src/foresight/backtesting.py:1`
- Test: `tests/test_backtesting_max_train_size.py`

**Work:**
- Add `max_train_size: int | None = None`.
- Ensure existing behavior is unchanged when `None`.

**Verify:** `pytest -q tests/test_backtesting_max_train_size.py`

---

### Task 3: Add cross-validation predictions table (panel-aware)

**Files:**
- Create: `src/foresight/cv.py`
- Test: `tests/test_cv_predictions.py`

**Work:**
- Implement `cross_validation_predictions(...) -> pd.DataFrame`
- Output columns: `unique_id`, `ds`, `cutoff`, `step`, `y`, `yhat`, `model`
- Work for both single-series and panel datasets via `to_long(...)`.

**Verify:** `pytest -q tests/test_cv_predictions.py`

---

### Task 4: Add metrics evaluation over predictions tables

**Files:**
- Create: `src/foresight/eval_predictions.py`
- Test: `tests/test_eval_predictions.py`

**Work:**
- Compute MAE/RMSE/MAPE/sMAPE + optional metrics from a predictions DataFrame.
- Support aggregation overall and by `step`.

**Verify:** `pytest -q tests/test_eval_predictions.py`

---

### Task 5: Add conformal interval calibration from CV residuals

**Files:**
- Create: `src/foresight/conformal.py`
- Test: `tests/test_conformal.py`

**Work:**
- Fit symmetric conformal intervals from `abs(y - yhat)` residuals.
- Support `levels=(0.8, 0.9, 0.95)` and `per_step=True/False`.
- Provide helper to attach interval columns to a predictions table.

**Verify:** `pytest -q tests/test_conformal.py`

---

### Task 6: Add probabilistic/interval metrics (coverage, width, pinball, MSIS)

**Files:**
- Modify: `src/foresight/metrics.py:1`
- Test: `tests/test_metrics_probabilistic.py`

**Work:**
- Add `pinball_loss`, `interval_coverage`, `mean_interval_width`, and a scaled interval score (MSIS-like).

**Verify:** `pytest -q tests/test_metrics_probabilistic.py`

---

### Task 7: Intermittent demand models (Croston + TSB)

**Files:**
- Create: `src/foresight/models/intermittent.py`
- Modify: `src/foresight/models/registry.py:1`
- Test: `tests/test_models_intermittent.py`

**Verify:** `pytest -q tests/test_models_intermittent.py`

---

### Task 8: ADIDA-style aggregation baseline (intermittent demand)

**Files:**
- Modify: `src/foresight/models/intermittent.py:1`
- Modify: `src/foresight/models/registry.py:1`
- Test: `tests/test_models_adida.py`

---

### Task 9: Fourier regression forecaster (trend + Fourier seasonality)

**Files:**
- Create: `src/foresight/models/fourier.py`
- Modify: `src/foresight/models/registry.py:1`
- Test: `tests/test_models_fourier.py`

---

### Task 10: Polynomial trend forecaster

**Files:**
- Create: `src/foresight/models/trend.py`
- Modify: `src/foresight/models/registry.py:1`
- Test: `tests/test_models_trend.py`

---

### Task 11: Auto-tuned smoothing models (grid search)

**Files:**
- Modify: `src/foresight/models/smoothing.py:1`
- Modify: `src/foresight/models/registry.py:1`
- Test: `tests/test_models_smoothing_auto.py`

---

### Task 12: Add a transform pipeline wrapper model (`pipeline`)

**Files:**
- Create: `src/foresight/transforms.py`
- Modify: `src/foresight/models/registry.py:1`
- Test: `tests/test_pipeline_model.py`

**Work:**
- Implement transforms: `log1p`, `standardize`, `diff1`.
- Implement registry model `pipeline` with params: `base`, `transforms`.

---

### Task 13: Add direct multi-horizon linear regression on lag features

**Files:**
- Modify: `src/foresight/models/regression.py:1`
- Modify: `src/foresight/models/registry.py:1`
- Test: `tests/test_models_regression_direct.py`

---

### Task 14: Optional sklearn tree-based lag model (RandomForest)

**Files:**
- Modify: `src/foresight/models/regression.py:1`
- Modify: `src/foresight/models/registry.py:1`
- Test: `tests/test_models_regression_optional_tree.py`

---

### Task 15: CLI command: export CV predictions (`foresight cv run`)

**Files:**
- Modify: `src/foresight/cli.py:1`
- Test: `tests/test_cli_cv_run.py`

---

### Task 16: CLI command: evaluate a CSV without dataset registry

**Files:**
- Modify: `src/foresight/cli.py:1`
- Create: `src/foresight/io.py`
- Test: `tests/test_cli_eval_csv.py`

---

### Task 17: Extend `eval run` with optional conformal intervals + interval metrics

**Files:**
- Modify: `src/foresight/cli.py:1`
- Modify: `src/foresight/eval_forecast.py:1`
- Test: `tests/test_cli_eval_run_intervals.py`

---

### Task 18: Improve dataset validation for time monotonicity and duplicates

**Files:**
- Modify: `src/foresight/cli.py:1`
- Modify: `src/foresight/data/format.py:1`
- Test: `tests/test_datasets_validate_time_checks.py`

---

### Task 19: Add examples for CV + conformal + intermittent demand

**Files:**
- Create: `examples/cv_and_conformal.py`
- Create: `examples/intermittent_demand.py`

---

### Task 20: README deep update (CV, conformal, pipeline, CSV eval, model zoo)

**Files:**
- Modify: `README.md:1`

**Verify:**
```bash
python tools/check_no_ipynb.py
ruff check src tests tools
ruff format --check src tests tools
python -m compileall -q src tools
pytest -q
```

