# Data Processing Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade ForeSight's data-processing surface so forecasting models can consume richer covariate semantics, explicit future data, reusable preprocessing pipelines, stronger lag grammar, and hierarchy-aware exogenous handling.

**Architecture:** Keep the existing long-format `unique_id/ds/y` core, but stop treating all exogenous data as a single undifferentiated `x_cols` bucket. Borrow the best ideas from adjacent projects without copying their entire APIs: split covariates by role, introduce explicit future-data entry points, add composable transformers for preprocessing, and extend the data layer so the registry and forecast/eval code can make capability-driven decisions instead of relying on ad hoc conventions.

**Tech Stack:** Python 3.10+, NumPy, Pandas, pytest, existing ForeSight registry/data/forecast modules

---

## Scope Guardrails

- Do not mix this work with new model-family expansion unless a batch explicitly needs a new model to prove the data contract.
- Preserve backward compatibility for existing `x_cols` call sites where feasible; prefer additive APIs and compatibility shims over hard breaks.
- Keep `unique_id/ds/y` as the canonical long-format core.
- Ship the new surface in small, test-first batches so forecast/eval/CLI remain coherent.

---

### Task 1: Split Covariates By Role

**Files:**
- Modify: `src/foresight/data/format.py`
- Modify: `src/foresight/data/prep.py`
- Modify: `src/foresight/data/__init__.py`
- Modify: `src/foresight/forecast.py`
- Modify: `src/foresight/eval_forecast.py`
- Test: `tests/test_data_to_long.py`
- Test: `tests/test_data_prep.py`
- Test: `tests/test_forecast_api.py`
- Test: `tests/test_eval_local_xreg.py`

**Target surface:**
- `historic_x_cols`
- `future_x_cols`
- compatibility alias: `x_cols`

**Step 1: Write the failing format tests**

Add tests requiring `to_long(...)` to support:
- `historic_x_cols=("promo_hist",)`
- `future_x_cols=("promo_futr",)`
- compatibility with existing `x_cols=...`

**Step 2: Run RED**

```bash
PYTHONPATH=src pytest -q tests/test_data_to_long.py -k "historic_x_cols or future_x_cols or x_cols_compat"
```

**Step 3: Implement minimal format changes**

- Keep `x_cols` as a compatibility alias
- write both role-specific covariates into the long DataFrame
- preserve column order deterministically

**Step 4: Extend prep rules**

Make `prepare_long_df(...)` validate/fill missing values separately for:
- target `y`
- historic covariates
- future covariates

**Step 5: Run GREEN**

```bash
PYTHONPATH=src pytest -q tests/test_data_to_long.py tests/test_data_prep.py
```

---

### Task 2: Add Explicit Future Data Inputs

**Files:**
- Modify: `src/foresight/forecast.py`
- Modify: `src/foresight/eval_forecast.py`
- Modify: `src/foresight/cli.py`
- Test: `tests/test_forecast_api.py`
- Test: `tests/test_cli_forecast.py`

**Target surface:**
- `forecast_model_long_df(..., future_df=...)`
- optional CLI support for a separate future covariates file

**Step 1: Write the failing tests**

Require forecasting to support:
- history in one DataFrame
- future covariates in a separate `future_df`

Keep the current "future rows with missing y" path working too.

**Step 2: Run RED**

```bash
PYTHONPATH=src pytest -q tests/test_forecast_api.py -k "future_df"
PYTHONPATH=src pytest -q tests/test_cli_forecast.py -k "future_df"
```

**Step 3: Implement the minimal API**

- merge or normalize `future_df` internally into the same canonical path
- validate per-series horizon coverage
- validate required future covariate columns

**Step 4: Run GREEN**

```bash
PYTHONPATH=src pytest -q tests/test_forecast_api.py -k "future_df or covariates"
PYTHONPATH=src pytest -q tests/test_cli_forecast.py -k "future_df or x_cols"
```

---

### Task 3: Add Reusable Preprocessing Transformers

**Files:**
- Create: `src/foresight/transforms/preprocessing.py`
- Modify: `src/foresight/transforms.py`
- Modify: `src/foresight/__init__.py`
- Test: `tests/test_transforms.py`
- Test: `tests/test_forecast_api.py`

**Target surface:**
- `MissingValueImputer`
- `StandardScaler`
- `Differencer`
- `BoxCoxTransformer`

**Step 1: Write failing unit tests**

Require simple fit/transform/inverse_transform behavior for each transformer.

**Step 2: Run RED**

```bash
PYTHONPATH=src pytest -q tests/test_transforms.py -k "imputer or scaler or differencer or boxcox"
```

**Step 3: Implement minimal transformer protocol**

Each transformer should expose:
- `fit(...)`
- `transform(...)`
- `inverse_transform(...)` where appropriate

Do not build a complex sklearn clone. Keep it small and ForeSight-specific.

**Step 4: Use one transformer in forecast/eval smoke**

Add one narrow end-to-end test proving transformed data can still forecast.

**Step 5: Run GREEN**

```bash
PYTHONPATH=src pytest -q tests/test_transforms.py tests/test_forecast_api.py -k "transform"
```

---

### Task 4: Extend Lag Grammar

**Files:**
- Modify: `src/foresight/features/lag.py`
- Modify: `src/foresight/features/tabular.py`
- Modify: `src/foresight/models/regression.py`
- Modify: `src/foresight/models/global_regression.py`
- Test: `tests/test_features_lag.py`
- Test: `tests/test_models_lag_derived_features.py`

**Target surface:**
- `target_lags`
- `historic_x_lags`
- `future_x_lags`

**Step 1: Write the failing tests**

Require lag feature builders to distinguish:
- target-series lag windows
- historic exogenous lag windows
- future exogenous aligned features

**Step 2: Run RED**

```bash
PYTHONPATH=src pytest -q tests/test_features_lag.py tests/test_models_lag_derived_features.py -k "historic_x_lags or future_x_lags"
```

**Step 3: Implement the minimal grammar**

Start with:
- `int`
- tuple/list of ints

Do not add dict-based grammar unless the first version proves insufficient.

**Step 4: Run GREEN**

```bash
PYTHONPATH=src pytest -q tests/test_features_lag.py tests/test_models_lag_derived_features.py
```

---

### Task 5: Add Hierarchical Exogenous Aggregation Rules

**Files:**
- Modify: `src/foresight/hierarchical.py`
- Modify: `src/foresight/data/format.py`
- Test: `tests/test_hierarchical.py`

**Target surface:**
- `exog_agg={"promo": "sum", "temp": "mean"}`

**Step 1: Write the failing tests**

Require hierarchy helpers to aggregate selected exogenous columns with per-column rules.

**Step 2: Run RED**

```bash
PYTHONPATH=src pytest -q tests/test_hierarchical.py -k "exog_agg"
```

**Step 3: Implement minimal aggregation support**

Support:
- `sum`
- `mean`
- `min`
- `max`

Reject unknown aggregations clearly.

**Step 4: Run GREEN**

```bash
PYTHONPATH=src pytest -q tests/test_hierarchical.py -k "exog_agg"
```

---

### Task 6: Update Public Surfaces

**Files:**
- Modify: `README.md`
- Modify: `docs/api.md`
- Modify: `docs/models.md`
- Modify: `docs/DEVELOPMENT.md`

**Step 1: Document the new data taxonomy**

Explain:
- target `y`
- historic covariates
- future covariates
- static features

**Step 2: Document migration**

Show that legacy `x_cols` still works and how to move to role-specific arguments.

**Step 3: Regenerate model docs if capability metadata changes**

```bash
PYTHONPATH=src python tools/generate_model_capability_docs.py
PYTHONPATH=src python tools/check_capability_docs.py
```

---

## Recommended Execution Order

1. Covariate role split
2. Explicit `future_df` / future-data entry points
3. Preprocessing transformers
4. Lag grammar extension
5. Hierarchical exogenous aggregation
6. Public docs sync
