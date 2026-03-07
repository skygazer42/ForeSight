# ForeSight Package Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evolve `foresight` from a model-rich benchmarking toolkit into a fuller time-series package with stable training/inference APIs, stronger data handling, and higher-value forecasting workflows.

**Architecture:** Prioritize infrastructure before adding more algorithms. First establish an object-style forecaster API and a dedicated forecast workflow on top of the existing registry and backtesting layers. Then strengthen data contracts and probabilistic evaluation, and only after that add larger modeling surface areas such as multivariate and hierarchical forecasting.

**Tech Stack:** Python 3.10, numpy, pandas, pytest, ruff, optional scikit-learn/statsmodels/xgboost/lightgbm/catboost/torch.

---

## Priority Order

- **P0:** Product foundation needed for real package usage
- **P1:** Forecasting-method coverage that materially broadens the package
- **P2:** Platform and research ergonomics

### Task 1: Add Persistent Forecaster Objects (P0)

**Files:**
- Create: `src/foresight/base.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py`
- Modify: `src/foresight/__init__.py`
- Test: `tests/test_forecaster_api.py`
- Test: `tests/test_global_forecaster_api.py`
- Docs: `README.md`

**Step 1: Write the failing tests**

Add tests that define the desired public contract:
- local forecaster object supports `fit(y)` then `predict(horizon)`
- global forecaster object supports `fit(long_df)` then `predict(cutoff, horizon)`
- object wrappers preserve existing registry defaults and model params

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_forecaster_api.py tests/test_global_forecaster_api.py`

Expected: FAIL because object API does not exist.

**Step 3: Write minimal implementation**

Implement lightweight base protocols/classes:
- `BaseForecaster`
- `BaseGlobalForecaster`
- thin registry-backed wrappers for local and global models

Do not replace current callable API yet. Keep backward compatibility.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_forecaster_api.py tests/test_global_forecaster_api.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/base.py src/foresight/models/registry.py src/foresight/models/__init__.py src/foresight/__init__.py tests/test_forecaster_api.py tests/test_global_forecaster_api.py README.md
git commit -m "feat(api): add persistent forecaster objects"
```

### Task 2: Add Forecast-First Python API and CLI (P0)

**Files:**
- Create: `src/foresight/forecast.py`
- Modify: `src/foresight/cli.py`
- Modify: `src/foresight/models/registry.py`
- Test: `tests/test_forecast_api.py`
- Test: `tests/test_cli_forecast.py`
- Docs: `README.md`
- Docs: `examples/quickstart_eval.py`

**Step 1: Write the failing tests**

Add tests for:
- `forecast_model(...)` on a single series returns forecast rows for future horizon
- `forecast_model_long_df(...)` on panel data returns `unique_id/ds/yhat`
- CLI command such as `foresight forecast csv ...` returns forecast payload instead of backtest metrics

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_forecast_api.py tests/test_cli_forecast.py`

Expected: FAIL because no forecast module/CLI command exists.

**Step 3: Write minimal implementation**

Build dedicated forecast helpers that:
- fit on all available history
- use the new object API when available
- infer forecast timestamps from existing frequency where possible
- keep output schema consistent with current `cv` prediction tables

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_forecast_api.py tests/test_cli_forecast.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/forecast.py src/foresight/cli.py src/foresight/models/registry.py tests/test_forecast_api.py tests/test_cli_forecast.py README.md examples/quickstart_eval.py
git commit -m "feat(forecast): add forecast-first API and CLI"
```

### Task 3: Strengthen Data Preparation Utilities (P0)

**Files:**
- Modify: `src/foresight/data/format.py`
- Modify: `src/foresight/io.py`
- Create: `src/foresight/data/prep.py`
- Test: `tests/test_data_prep.py`
- Test: `tests/test_data_to_long.py`
- Docs: `README.md`

**Step 1: Write the failing tests**

Add tests for:
- inferring regular frequency from `ds`
- resampling and gap-filling missing timestamps per series
- optional missing-value policies for `y` and `x_cols`
- rejecting mixed/irregular frequencies when user asks for strict mode

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_data_prep.py tests/test_data_to_long.py`

Expected: FAIL because prep helpers do not exist.

**Step 3: Write minimal implementation**

Add a prep layer with focused helpers:
- `infer_series_frequency(...)`
- `prepare_long_df(...)`
- gap handling modes such as `error`, `drop`, `ffill`, `zero`, `interpolate`

Keep timezone and holiday logic out of scope for this phase.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_data_prep.py tests/test_data_to_long.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/format.py src/foresight/io.py src/foresight/data/prep.py tests/test_data_prep.py tests/test_data_to_long.py README.md
git commit -m "feat(data): add long-format preparation helpers"
```

### Task 4: Add Serialization and Reuse of Trained Models (P0)

**Files:**
- Create: `src/foresight/serialization.py`
- Modify: `src/foresight/base.py`
- Modify: `src/foresight/cli.py`
- Test: `tests/test_serialization.py`
- Test: `tests/test_cli_forecast.py`
- Docs: `README.md`

**Step 1: Write the failing tests**

Add tests for:
- save/load round trip for fitted local forecaster object
- save/load round trip for fitted global forecaster object
- CLI can save fitted artifact and reuse it for prediction

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_serialization.py tests/test_cli_forecast.py`

Expected: FAIL because serialization layer does not exist.

**Step 3: Write minimal implementation**

Start with pickle-based artifact persistence plus metadata:
- model key
- model params
- package version
- train schema summary

Do not add ONNX/export yet.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_serialization.py tests/test_cli_forecast.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/serialization.py src/foresight/base.py src/foresight/cli.py tests/test_serialization.py tests/test_cli_forecast.py README.md
git commit -m "feat(serialization): persist fitted forecasters"
```

### Task 5: Expand Probabilistic Forecast Evaluation (P1)

**Files:**
- Modify: `src/foresight/metrics.py`
- Modify: `src/foresight/conformal.py`
- Modify: `src/foresight/eval_forecast.py`
- Test: `tests/test_metrics_probabilistic.py`
- Test: `tests/test_conformal.py`
- Docs: `README.md`

**Step 1: Write the failing tests**

Add tests for:
- weighted interval score / Winkler-style score naming consistency
- CRPS for quantile grids or empirical samples
- calibration summary output from conformal predictions

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_metrics_probabilistic.py tests/test_conformal.py`

Expected: FAIL for missing metrics or payload fields.

**Step 3: Write minimal implementation**

Add proper-scoring-rule support without changing current metric names:
- CRPS approximation from quantile grid
- calibration gap summaries
- interval sharpness summaries

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_metrics_probabilistic.py tests/test_conformal.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/metrics.py src/foresight/conformal.py src/foresight/eval_forecast.py tests/test_metrics_probabilistic.py tests/test_conformal.py README.md
git commit -m "feat(prob): add richer probabilistic evaluation"
```

### Task 6: Add Hyperparameter Search Workflow (P1)

**Files:**
- Create: `src/foresight/tuning.py`
- Modify: `src/foresight/cli.py`
- Test: `tests/test_tuning.py`
- Test: `tests/test_cli_tuning.py`
- Docs: `README.md`

**Step 1: Write the failing tests**

Add tests for:
- grid-search over a small parameter space using walk-forward metrics
- selecting best config by metric
- CLI output for tuning summary

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_tuning.py tests/test_cli_tuning.py`

Expected: FAIL because tuning module/CLI command does not exist.

**Step 3: Write minimal implementation**

Implement a first-party, dependency-light search layer:
- deterministic grid search only
- metric selection such as `mae`, `rmse`, `smape`
- support both dataset-key and `long_df` style evaluation

Leave Optuna/Bayesian search for later.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_tuning.py tests/test_cli_tuning.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/tuning.py src/foresight/cli.py tests/test_tuning.py tests/test_cli_tuning.py README.md
git commit -m "feat(tuning): add first-party grid search"
```

### Task 7: Add Multivariate Forecasting Support (P1)

**Files:**
- Create: `src/foresight/models/multivariate.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/eval_forecast.py`
- Create: `tests/test_models_multivariate.py`
- Create: `tests/test_eval_multivariate.py`
- Docs: `README.md`

**Step 1: Write the failing tests**

Add tests for:
- package-level `VAR` or similar multivariate baseline on a wide matrix
- evaluation payload for multiple target columns
- clear error path when user mixes multivariate models with single-`y` API

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_models_multivariate.py tests/test_eval_multivariate.py`

Expected: FAIL because multivariate model path does not exist.

**Step 3: Write minimal implementation**

Keep scope narrow:
- start with statsmodels-backed `VAR`
- add a separate multivariate API path instead of overloading current `unique_id/ds/y`
- define explicit input contract for multiple targets

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_models_multivariate.py tests/test_eval_multivariate.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/models/multivariate.py src/foresight/models/registry.py src/foresight/eval_forecast.py tests/test_models_multivariate.py tests/test_eval_multivariate.py README.md
git commit -m "feat(multivariate): add package-level VAR support"
```

### Task 8: Add Hierarchical Forecast Reconciliation (P1)

**Files:**
- Create: `src/foresight/hierarchical.py`
- Modify: `src/foresight/data/format.py`
- Modify: `src/foresight/eval_forecast.py`
- Create: `tests/test_hierarchical.py`
- Docs: `README.md`

**Step 1: Write the failing tests**

Add tests for:
- bottom-up reconciliation from child series to parent series
- top-down split using historical proportions
- consistency checks on reconciled forecasts

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_hierarchical.py`

Expected: FAIL because hierarchical module does not exist.

**Step 3: Write minimal implementation**

Start with:
- hierarchy spec input
- bottom-up and top-down only
- reconciliation applied to forecast tables, not to training itself

Leave MinT for later.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_hierarchical.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/hierarchical.py src/foresight/data/format.py src/foresight/eval_forecast.py tests/test_hierarchical.py README.md
git commit -m "feat(hierarchy): add basic forecast reconciliation"
```

### Task 9: Improve Package Ergonomics and Metadata (P2)

**Files:**
- Modify: `src/foresight/__init__.py`
- Modify: `README.md`
- Modify: `docs/INSTALL.md`
- Modify: `docs/DEVELOPMENT.md`
- Test: `tests/test_cli_models.py`
- Test: `tests/test_cli_eval.py`

**Step 1: Write the failing tests**

Add tests that assert:
- version/help output mentions new forecast/tuning commands
- docs examples remain in sync with public CLI/API names

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_cli_models.py tests/test_cli_eval.py`

Expected: FAIL if docs/help text is stale.

**Step 3: Write minimal implementation**

Update public-facing docs and examples:
- distinguish benchmarking vs forecasting workflows
- document object API
- document data preparation and artifact persistence

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_cli_models.py tests/test_cli_eval.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/__init__.py README.md docs/INSTALL.md docs/DEVELOPMENT.md tests/test_cli_models.py tests/test_cli_eval.py
git commit -m "docs: align package docs with new forecasting workflow"
```

## Recommended Execution Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6
7. Task 7
8. Task 8
9. Task 9

## What Not To Do Yet

- Do not add another large batch of model wrappers before Task 1 and Task 2 land.
- Do not add ONNX, distributed training, or model-serving infra in the next phase.
- Do not widen the public input contract until the object API and forecast API are stable.
- Do not mix multivariate and hierarchical support into the same first implementation.
