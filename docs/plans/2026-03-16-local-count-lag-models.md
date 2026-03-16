# Local Count Lag Models Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three local sklearn count-family lag forecasters that fill the current local/global parity gap: `poisson-lag`, `gamma-lag`, and `tweedie-lag`.

**Architecture:** Reuse the existing local lag model path in `src/foresight/models/catalog/ml.py`, `src/foresight/models/runtime.py`, and `src/foresight/models/regression.py`. Keep these models direct multi-horizon only, with the same lag-derived, seasonal, and Fourier feature plumbing already used by `huber-lag` and `quantile-lag`.

**Tech Stack:** Python 3.10, numpy, pandas, pytest, scikit-learn optional dependency

---

### Task 1: Add Registry and Optional-Dependency Coverage

**Files:**
- Modify: `tests/test_models_optional_deps_ml.py`

**Step 1: Write the failing test**

```python
def test_ml_models_are_registered_as_optional() -> None:
    for key in ("poisson-lag", "gamma-lag", "tweedie-lag"):
        spec = get_model_spec(key)
        assert "ml" in spec.requires
```

Also add the new keys to the missing-dependency loop so `make_forecaster()` raises `ImportError` when sklearn is unavailable.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_optional_deps_ml.py -q`
Expected: FAIL because the new model keys are not registered yet

**Step 3: Write the minimal implementation**

Only update the test file at this step. Do not write production code yet.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_models_optional_deps_ml.py -q`
Expected: FAIL with missing model-spec or lookup errors for the new keys

**Step 5: Commit**

```bash
git add tests/test_models_optional_deps_ml.py
git commit -m "test: cover local count lag model registration"
```

### Task 2: Add Local Smoke and Validation Tests

**Files:**
- Modify: `tests/test_models_lag_derived_features.py`

**Step 1: Write the failing tests**

```python
def test_local_count_lag_models_smoke_when_sklearn_installed() -> None:
    configs = [
        ("poisson-lag", {"lags": 8, "alpha": 0.1, "max_iter": 200}),
        ("gamma-lag", {"lags": 8, "alpha": 0.1, "max_iter": 200}),
        ("tweedie-lag", {"lags": 8, "power": 1.5, "alpha": 0.1, "max_iter": 200}),
    ]


def test_gamma_lag_rejects_non_positive_targets() -> None:
    ...
```

Use positive synthetic data for the happy path and explicit zero/negative examples for domain validation.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_lag_derived_features.py -q`
Expected: FAIL because the new model keys are missing

**Step 3: Write the minimal implementation**

Only update tests in this step.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_models_lag_derived_features.py -q`
Expected: FAIL on missing registry entries or missing runtime wiring

**Step 5: Commit**

```bash
git add tests/test_models_lag_derived_features.py
git commit -m "test: add local count lag smoke coverage"
```

### Task 3: Add API Coercion Coverage

**Files:**
- Modify: `tests/test_forecaster_api.py`

**Step 1: Write the failing tests**

```python
def test_make_forecaster_tweedie_lag_coerces_string_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ...
```

Patch the runtime regression function and assert that `lags`, `power`, `alpha`, and `max_iter` arrive as numeric values after passing string inputs through `make_forecaster()`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_forecaster_api.py -q`
Expected: FAIL because the factory function does not exist yet

**Step 3: Write the minimal implementation**

Only update the test file at this step.

**Step 4: Run test to verify it fails for the right reason**

Run: `pytest tests/test_forecaster_api.py -q`
Expected: FAIL because the runtime model factory is missing

**Step 5: Commit**

```bash
git add tests/test_forecaster_api.py
git commit -m "test: cover tweedie lag API coercion"
```

### Task 4: Implement Local Count Lag Models

**Files:**
- Modify: `src/foresight/models/regression.py`
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/catalog/ml.py`

**Step 1: Write the minimal implementation**

Add three regression functions:

```python
def poisson_lag_direct_forecast(...): ...
def gamma_lag_direct_forecast(...): ...
def tweedie_lag_direct_forecast(...): ...
```

Implementation requirements:

- import sklearn lazily
- use `MultiOutputRegressor`
- reuse `_compute_feature_start_t()`, `_make_lagged_xy_multi()`, `_augment_lag_matrix()`, and `_augment_lag_feat_row()`
- validate target support before fitting
- reuse the existing Tweedie target-domain helper if possible

Then add matching runtime factories and catalog registrations.

**Step 2: Run focused tests to verify they pass**

Run: `pytest tests/test_models_optional_deps_ml.py tests/test_models_lag_derived_features.py tests/test_forecaster_api.py -q`
Expected: PASS

**Step 3: Refactor only if needed**

If repeated validation logic appears, extract a small shared helper without changing behavior.

**Step 4: Run focused tests again**

Run: `pytest tests/test_models_optional_deps_ml.py tests/test_models_lag_derived_features.py tests/test_forecaster_api.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/models/regression.py src/foresight/models/runtime.py src/foresight/models/catalog/ml.py tests/test_models_optional_deps_ml.py tests/test_models_lag_derived_features.py tests/test_forecaster_api.py
git commit -m "feat: add local count lag models"
```

### Task 5: Verify Broader Compatibility

**Files:**
- No new files expected

**Step 1: Run targeted lint**

Run: `ruff check src/foresight/models/regression.py src/foresight/models/runtime.py src/foresight/models/catalog/ml.py tests/test_models_optional_deps_ml.py tests/test_models_lag_derived_features.py tests/test_forecaster_api.py`
Expected: PASS

**Step 2: Run broader regression tests**

Run: `pytest tests/test_models_optional_deps_ml.py tests/test_models_lag_derived_features.py tests/test_forecaster_api.py tests/test_models_global_regression_smoke.py -q`
Expected: PASS or only unrelated skips

**Step 3: Review git diff**

Run: `git diff -- src/foresight/models/regression.py src/foresight/models/runtime.py src/foresight/models/catalog/ml.py tests/test_models_optional_deps_ml.py tests/test_models_lag_derived_features.py tests/test_forecaster_api.py docs/plans/2026-03-16-local-count-lag-models-design.md docs/plans/2026-03-16-local-count-lag-models.md`
Expected: Only the intended files changed

**Step 4: Commit**

```bash
git add docs/plans/2026-03-16-local-count-lag-models-design.md docs/plans/2026-03-16-local-count-lag-models.md
git commit -m "docs: add local count lag model design and plan"
```

**Step 5: Final verification before completion**

Run the exact commands above again if any code changed after the prior verification step. Do not claim completion without fresh evidence.
