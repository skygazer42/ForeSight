# Balanced TS Workflows and Local Models Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a compact wave of long-format TS data workflows and four lightweight local sklearn lag models without changing the forecast/evaluation service paths.

**Architecture:** Keep new preprocessing logic in `src/foresight/data/` and reuse existing feature builders in `src/foresight/features/`. Keep new local models on the existing `catalog/ml.py -> runtime.py -> regression.py` path, with tests focused on deterministic helper behavior, CLI wiring, and optional dependency paths.

**Tech Stack:** Python 3.10, pandas, numpy, pytest, scikit-learn optional dependency, argparse CLI

---

### Task 1: Add Long-Format Workflow Helpers

**Files:**
- Create: `src/foresight/data/workflows.py`
- Modify: `src/foresight/data/__init__.py`
- Test: `tests/test_data_workflows.py`

**Step 1: Write the failing tests**

```python
def test_align_long_df_regularizes_panel_frequency() -> None:
    out = align_long_df(long_df, freq="D", agg="last")
    assert out["ds"].tolist() == list(pd.date_range("2020-01-01", periods=3, freq="D"))


def test_clip_long_df_outliers_clips_per_series() -> None:
    out = clip_long_df_outliers(long_df, method="iqr", columns=("y",), iqr_k=1.5)
    assert float(out.loc[out["ds"] == "2020-01-04", "y"].iloc[0]) < 100.0


def test_enrich_long_df_calendar_appends_prefixed_features() -> None:
    out = enrich_long_df_calendar(long_df, prefix="cal_")
    assert "cal_time_idx" in out.columns
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workflows.py -q`
Expected: FAIL with import errors or missing helper functions

**Step 3: Write the minimal implementation**

```python
def align_long_df(...): ...
def clip_long_df_outliers(...): ...
def enrich_long_df_calendar(...): ...
```

Implement only the validated wave-1 behavior:

- explicit long-format validation
- deterministic sorting
- per-series frequency alignment / resampling
- per-series numeric clipping
- prefixed calendar column expansion using `build_time_features()`

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data_workflows.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/data/__init__.py tests/test_data_workflows.py
git commit -m "feat: add ts data workflow helpers"
```

### Task 2: Add Supervised Training-Frame Builder

**Files:**
- Modify: `src/foresight/data/workflows.py`
- Modify: `src/foresight/__init__.py`
- Test: `tests/test_data_workflows.py`
- Test: `tests/test_root_import.py`

**Step 1: Write the failing tests**

```python
def test_make_supervised_frame_long_single_step_schema() -> None:
    out = make_supervised_frame(long_df, lags=3, horizon=1)
    assert {"unique_id", "ds", "target_t", "y_target"} <= set(out.columns)


def test_make_supervised_frame_long_multi_step_schema() -> None:
    out = make_supervised_frame(long_df, lags=4, horizon=2)
    assert {"y_t+1", "y_t+2"} <= set(out.columns)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: FAIL because `make_supervised_frame` is not exported yet

**Step 3: Write the minimal implementation**

```python
def make_supervised_frame(...): ...
```

Implement:

- long and wide input normalization
- lag/derived/seasonal/time feature reuse from existing feature modules
- single-step and direct multi-step target columns
- clear `feat_*` naming and metadata columns

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data_workflows.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/data/workflows.py src/foresight/__init__.py tests/test_data_workflows.py tests/test_root_import.py
git commit -m "feat: add supervised training frame builder"
```

### Task 3: Wire New CLI Data Commands

**Files:**
- Modify: `src/foresight/cli_data.py`
- Test: `tests/test_cli_data.py`

**Step 1: Write the failing tests**

```python
def test_cli_data_align_long_json(tmp_path: Path) -> None:
    proc = _run_cli("data", "align-long", "--path", str(csv_path), "--freq", "D", "--format", "json")
    assert proc.returncode == 0


def test_cli_data_make_supervised_json(tmp_path: Path) -> None:
    proc = _run_cli("data", "make-supervised", "--path", str(csv_path), "--lags", "3", "--horizon", "2", "--format", "json")
    assert proc.returncode == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli_data.py -q`
Expected: FAIL because the subcommands are not registered yet

**Step 3: Write the minimal implementation**

```python
def _cmd_data_align_long(args: argparse.Namespace) -> int: ...
def _cmd_data_clip_outliers(args: argparse.Namespace) -> int: ...
def _cmd_data_calendar_features(args: argparse.Namespace) -> int: ...
def _cmd_data_make_supervised(args: argparse.Namespace) -> int: ...
```

Register only the wave-1 options needed for validated behavior.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli_data.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli_data.py tests/test_cli_data.py
git commit -m "feat: add cli data workflow commands"
```

### Task 4: Add Local Bayesian / Sparse / Margin Lag Models

**Files:**
- Modify: `src/foresight/models/regression.py`
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/catalog/ml.py`
- Test: `tests/test_models_optional_deps_ml.py`
- Test: `tests/test_forecaster_api.py`

**Step 1: Write the failing tests**

```python
def test_new_local_ml_models_are_registered_as_optional() -> None:
    for key in ("bayesian-ridge-lag", "ard-lag", "omp-lag", "passive-aggressive-lag"):
        assert "ml" in get_model_spec(key).requires


def test_make_forecaster_passive_aggressive_accepts_legacy_uppercase_c_keyword(...) -> None:
    ...
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models_optional_deps_ml.py tests/test_forecaster_api.py -q`
Expected: FAIL because the model keys and factories are missing

**Step 3: Write the minimal implementation**

```python
def bayesian_ridge_lag_direct_forecast(...): ...
def ard_lag_direct_forecast(...): ...
def omp_lag_direct_forecast(...): ...
def passive_aggressive_lag_direct_forecast(...): ...
```

Also add matching runtime factories and `catalog/ml.py` registrations with parameter help and defaults aligned to the local lag family.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models_optional_deps_ml.py tests/test_forecaster_api.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/models/regression.py src/foresight/models/runtime.py src/foresight/models/catalog/ml.py tests/test_models_optional_deps_ml.py tests/test_forecaster_api.py
git commit -m "feat: add local bayesian and sparse lag models"
```

### Task 5: Add End-to-End Workflow Smoke Coverage

**Files:**
- Modify: `tests/test_data_workflows.py`

**Step 1: Write the failing test**

```python
def test_balanced_workflow_smoke_from_prepare_to_supervised() -> None:
    prepared = prepare_long_df(long_df, freq="D", y_missing="interpolate")
    aligned = align_long_df(prepared, freq="D", agg="last")
    clipped = clip_long_df_outliers(aligned, method="iqr")
    enriched = enrich_long_df_calendar(clipped, prefix="cal_")
    frame = make_supervised_frame(enriched, lags=3, horizon=2)
    assert not frame.empty
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_workflows.py::test_balanced_workflow_smoke_from_prepare_to_supervised -q`
Expected: FAIL until all helpers interoperate cleanly

**Step 3: Write minimal implementation adjustments**

```python
# Fix only the integration gaps discovered by the smoke test.
```

**Step 4: Run the verification suite**

Run: `pytest tests/test_data_workflows.py tests/test_cli_data.py tests/test_data_prep.py tests/test_features_tabular.py tests/test_models_optional_deps_ml.py tests/test_forecaster_api.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_data_workflows.py
git commit -m "test: add ts workflow smoke coverage"
```

### Task 6: Final Quality Gate

**Files:**
- Modify: `README.md` (only if public-facing data command examples need an update)

**Step 1: Run formatting and lint checks**

Run: `ruff check src tests`
Expected: PASS

**Step 2: Run the targeted verification suite**

Run: `pytest tests/test_data_workflows.py tests/test_cli_data.py tests/test_data_prep.py tests/test_features_tabular.py tests/test_models_optional_deps_ml.py tests/test_forecaster_api.py tests/test_root_import.py -q`
Expected: PASS

**Step 3: Update docs if command examples changed**

```markdown
# Add only minimal README examples for new workflow helpers if needed.
```

**Step 4: Re-run verification**

Run: `ruff check src tests && pytest tests/test_data_workflows.py tests/test_cli_data.py tests/test_data_prep.py tests/test_features_tabular.py tests/test_models_optional_deps_ml.py tests/test_forecaster_api.py tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add README.md src tests
git commit -m "feat: ship balanced ts workflows wave"
```
