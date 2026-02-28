# ForeSight Algorithm Expansion (20 Tasks) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand ForeSight from “naive baselines + CLI” into a small-but-solid forecasting toolkit with a model zoo (statistical + ML-style), richer evaluation/backtesting (per-horizon/per-series), and clearer docs/README — while keeping the default install lightweight.

**Architecture:** Use mainstream TS project patterns:
- **Unified model interface + model registry** (Darts / sktime-style).
- **Long/panel-friendly tabular data format** with `id/time/value` columns (common across Prophet, tsfresh, PyTorch Forecasting, Nixtla).
- **Backtesting-first evaluation**: walk-forward, per-horizon metrics, and consistent reporting.

Implementation approach:
- Keep core dependency-free beyond `numpy`/`pandas`.
- Add optional extras for heavier algorithms (`statsmodels`, `scikit-learn`, `torch`) but do not require them for CI.
- Provide built-in “classic” algorithms implemented in pure numpy where feasible (baselines, smoothing, theta, AR via OLS).

**Tech Stack:** Python 3.10+, `numpy`, `pandas`, `pytest`, `ruff`. Optional extras: `statsmodels`, `scikit-learn`, `prophet`, `torch`.

**Note:** Per user request, **do not commit** while executing this plan.

---

### Task 1: Add a model registry (`ModelSpec`) and `models list/info` plumbing

**Files:**
- Create: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py:1`
- Test: `tests/test_models_registry.py`

**Step 1: Write the failing test**
Add a test that `list_models()` contains at least `naive-last` and `seasonal-naive`, and each model has a description.

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_models_registry.py -q`
Expected: FAIL (module missing).

**Step 3: Write minimal implementation**
Add `ModelSpec` (key, description, params schema, requires extras, callable factory) + `list_models()` + `get_model_spec()`.

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_models_registry.py -q`
Expected: PASS

**Step 5: Checkpoint (no commit)**
Run: `git diff --stat`

---

### Task 2: Add more baseline models (mean/median/drift/moving-average)

**Files:**
- Create: `src/foresight/models/baselines.py`
- Modify: `src/foresight/models/__init__.py:1`
- Test: `tests/test_models_baselines.py`

**Steps:** TDD: add tests for each model output shape and a few known cases, implement functions, re-run tests.

---

### Task 3: Add seasonal mean baseline (`seasonal-mean`)

**Files:**
- Modify: `src/foresight/models/baselines.py:1`
- Test: `tests/test_models_seasonal_mean.py`

---

### Task 4: Add exponential smoothing models (SES / Holt / Holt-Winters additive)

**Files:**
- Create: `src/foresight/models/smoothing.py`
- Modify: `src/foresight/models/__init__.py:1`
- Test: `tests/test_models_smoothing.py`

---

### Task 5: Add Theta method (pure numpy)

**Files:**
- Create: `src/foresight/models/theta.py`
- Modify: `src/foresight/models/__init__.py:1`
- Test: `tests/test_models_theta.py`

---

### Task 6: Add autoregression via OLS (`ar-ols`) with `p` parameter

**Files:**
- Create: `src/foresight/models/ar.py`
- Modify: `src/foresight/models/__init__.py:1`
- Test: `tests/test_models_ar.py`

---

### Task 7: Add lag feature utilities (supervised tabular conversion)

**Files:**
- Create: `src/foresight/features/lag.py`
- Create: `src/foresight/features/__init__.py`
- Test: `tests/test_features_lag.py`

---

### Task 8: Add a numpy linear-regression forecaster on lag features (`lr-lag`)

**Files:**
- Create: `src/foresight/models/regression.py`
- Modify: `src/foresight/models/__init__.py:1`
- Test: `tests/test_models_regression.py`

---

### Task 9: Add optional `scikit-learn` regressors (only if installed)

**Files:**
- Modify: `src/foresight/models/regression.py:1`
- Modify: `pyproject.toml:1` (extras)
- Test: `tests/test_models_regression_optional.py`

---

### Task 10: Add optional `statsmodels` wrappers (ARIMA / ETS), gated by extras

**Files:**
- Create: `src/foresight/models/statsmodels_wrap.py`
- Modify: `pyproject.toml:1` (extras)
- Test: `tests/test_models_optional_deps.py`

---

### Task 11: Upgrade dataset specs to include `time_col`, `group_cols`, and `default_y`

**Files:**
- Modify: `src/foresight/datasets/registry.py:1`
- Modify: `src/foresight/datasets/loaders.py:1`
- Test: `tests/test_dataset_specs_extended.py`

---

### Task 12: Add `foresight.data.to_long()` (id/time/y canonical format)

**Files:**
- Create: `src/foresight/data/format.py`
- Create: `src/foresight/data/__init__.py`
- Test: `tests/test_data_to_long.py`

---

### Task 13: Add panel/group evaluation (`eval_forecaster`) over `group_cols`

**Files:**
- Create: `src/foresight/eval_forecast.py`
- Modify: `src/foresight/eval.py:1` (reuse internals)
- Test: `tests/test_eval_panel.py`

---

### Task 14: Add new metrics (MSE/WAPE/MASE/RMSSE) + per-series aggregation

**Files:**
- Modify: `src/foresight/metrics.py:1`
- Test: `tests/test_metrics_extended.py`

---

### Task 15: Add residual bootstrap prediction intervals for point-forecast models

**Files:**
- Create: `src/foresight/intervals.py`
- Test: `tests/test_intervals_bootstrap.py`

---

### Task 16: Add CLI `models list` / `models info <key>`

**Files:**
- Modify: `src/foresight/cli.py:1`
- Test: `tests/test_cli_models.py`

---

### Task 17: Add CLI `eval run --model <key>` (generic eval) and extend leaderboard

**Files:**
- Modify: `src/foresight/cli.py:1`
- Test: `tests/test_cli_eval_run.py`
- Test: `tests/test_cli_leaderboard_models.py`

---

### Task 18: Add examples (no notebooks): `examples/` runnable scripts

**Files:**
- Create: `examples/quickstart_eval.py`
- Create: `examples/leaderboard.py`

---

### Task 19: Deep README upgrade (model zoo, data format, extras, comparisons)

**Files:**
- Modify: `README.md:1`

---

### Task 20: Final verification (format/lint/compile/test + CLI smoke)

**Commands:**
```bash
python tools/check_no_ipynb.py
ruff check src tests tools
ruff format --check src tests tools
python -m compileall -q src tools
pytest -q

# Optional smoke
foresight models list
foresight eval run --model naive-last --dataset catfish --y-col Total --horizon 3 --step 3 --min-train-size 12
```

