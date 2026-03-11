# ForeSight Internal Architecture Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor ForeSight's internal package structure so shared data rules, model metadata, runtime builders, and workflow orchestration each live in their own layer while keeping the public Python API and CLI behavior compatible.

**Architecture:** Introduce a `contracts` layer for shared validation and parameter parsing, a split between model specs/catalog and runtime factory logic, and a `services` layer for forecast/evaluation orchestration. Keep existing public modules in place as facades so imports and CLI entrypoints stay stable throughout the migration.

**Tech Stack:** Python 3.10+, NumPy, Pandas, pytest, ruff, mypy, argparse, setuptools

---

### Task 1: Create The Refactor Worktree And Capture A Baseline

**Files:**
- Verify only
- Reference: `docs/plans/2026-03-11-internal-architecture-refactor-design.md`
- Reference: `docs/plans/2026-03-11-internal-architecture-refactor-implementation.md`

**Step 1: Verify the current branch and worktree state**

Run: `git status --short --branch`
Expected: current branch shown, with no unexpected tracked edits before starting the refactor branch

**Step 2: Create a dedicated refactor worktree**

Run: `git worktree add .worktrees/internal-architecture-refactor -b feat/internal-architecture-refactor`
Expected: new worktree created at `.worktrees/internal-architecture-refactor`

**Step 3: Run a focused compatibility baseline in the new worktree**

Run: `PYTHONPATH=src pytest -q tests/test_root_import.py tests/test_cli.py tests/test_forecast_api.py tests/test_models_registry.py tests/test_serialization.py`
Expected: PASS

**Step 4: Commit the design and plan docs if they are not already on the branch**

```bash
git add docs/plans/2026-03-11-internal-architecture-refactor-design.md docs/plans/2026-03-11-internal-architecture-refactor-implementation.md
git commit -m "docs: add internal architecture refactor design and plan"
```

---

### Task 2: Add Shared Contracts Modules

**Files:**
- Create: `src/foresight/contracts/__init__.py`
- Create: `src/foresight/contracts/frames.py`
- Create: `src/foresight/contracts/params.py`
- Create: `src/foresight/contracts/capabilities.py`
- Test: `tests/test_contracts_frames.py`
- Test: `tests/test_contracts_params.py`

**Step 1: Write the failing contracts tests**

```python
import pandas as pd
import pytest

from foresight.contracts.frames import require_long_df, require_future_df
from foresight.contracts.params import normalize_covariate_roles, parse_interval_levels


def test_require_long_df_rejects_missing_y() -> None:
    bad = pd.DataFrame({"unique_id": ["a"], "ds": [1]})
    with pytest.raises(KeyError, match="long_df missing required columns: \\['y'\\]"):
        require_long_df(bad)


def test_require_future_df_fills_nan_y_column() -> None:
    out = require_future_df(pd.DataFrame({"unique_id": ["a"], "ds": [1]}))
    assert "y" in out.columns
    assert out["y"].isna().all()


def test_normalize_covariate_roles_merges_legacy_x_cols() -> None:
    historic, future = normalize_covariate_roles(
        {"x_cols": "promo,price", "historic_x_cols": ("stock",)}
    )
    assert historic == ("stock",)
    assert future == ("promo", "price")


def test_parse_interval_levels_accepts_percent_inputs() -> None:
    assert parse_interval_levels("80,90") == (0.8, 0.9)
```

**Step 2: Run the contracts tests to confirm they fail**

Run: `PYTHONPATH=src pytest -q tests/test_contracts_frames.py tests/test_contracts_params.py`
Expected: FAIL with `ModuleNotFoundError` or missing function errors

**Step 3: Implement the minimal contracts package**

Required functions:

- frame validators with current error wording preserved
- model param normalization helpers
- interval and quantile parsing helpers
- capability-based argument checks

**Step 4: Run the contracts tests again**

Run: `PYTHONPATH=src pytest -q tests/test_contracts_frames.py tests/test_contracts_params.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/contracts/__init__.py src/foresight/contracts/frames.py src/foresight/contracts/params.py src/foresight/contracts/capabilities.py tests/test_contracts_frames.py tests/test_contracts_params.py
git commit -m "refactor: add shared contracts layer"
```

---

### Task 3: Rewire Existing Validation Helpers To The Contracts Layer

**Files:**
- Modify: `src/foresight/forecast.py`
- Modify: `src/foresight/eval_forecast.py`
- Modify: `src/foresight/base.py`
- Modify: `src/foresight/data/prep.py`
- Test: `tests/test_data_contracts_compat.py`
- Test: `tests/test_forecast_api.py`
- Test: `tests/test_eval_local_xreg.py`
- Test: `tests/test_data_prep.py`

**Step 1: Write one failing cross-surface compatibility test**

```python
import pandas as pd
import pytest

from foresight.eval_forecast import eval_model_long_df
from foresight.forecast import forecast_model_long_df


def test_forecast_and_eval_share_long_df_error_message() -> None:
    bad = pd.DataFrame({"unique_id": ["a"], "ds": [1]})

    with pytest.raises(KeyError, match="long_df missing required columns: \\['y'\\]"):
        forecast_model_long_df(model="naive-last", long_df=bad, horizon=1)

    with pytest.raises(KeyError, match="long_df missing required columns: \\['y'\\]"):
        eval_model_long_df(
            model="naive-last",
            long_df=bad,
            horizon=1,
            step=1,
            min_train_size=1,
        )
```

**Step 2: Run the compatibility and targeted regression tests**

Run: `PYTHONPATH=src pytest -q tests/test_data_contracts_compat.py tests/test_forecast_api.py tests/test_eval_local_xreg.py tests/test_data_prep.py`
Expected: FAIL before rewiring, or PASS with duplicated implementations still in place

**Step 3: Replace duplicated private helpers with thin wrappers over `foresight.contracts`**

Rules:

- preserve existing function names in the public modules
- preserve current error strings where possible
- avoid changing public function signatures in this task

**Step 4: Run the targeted suite again**

Run: `PYTHONPATH=src pytest -q tests/test_data_contracts_compat.py tests/test_forecast_api.py tests/test_eval_local_xreg.py tests/test_data_prep.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/forecast.py src/foresight/eval_forecast.py src/foresight/base.py src/foresight/data/prep.py tests/test_data_contracts_compat.py
git commit -m "refactor: route validation helpers through contracts"
```

---

### Task 4: Extract Model Specs And Runtime Builder Functions

**Files:**
- Create: `src/foresight/models/specs.py`
- Create: `src/foresight/models/factories.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/base.py`
- Test: `tests/test_models_registry.py`
- Test: `tests/test_serialization.py`

**Step 1: Write a failing regression test around the new stable runtime surface**

```python
from foresight.models.registry import get_model_spec
from foresight.models.specs import ModelSpec


def test_registry_returns_modelspec_instances() -> None:
    spec = get_model_spec("naive-last")
    assert isinstance(spec, ModelSpec)
    assert spec.interface == "local"
```

**Step 2: Run the targeted tests to confirm the new modules are missing**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py -k modelspec tests/test_serialization.py`
Expected: FAIL with import errors for `foresight.models.specs`

**Step 3: Move `ModelSpec` and builder logic out of `registry.py`**

Required outcomes:

- `ModelSpec` lives in `models/specs.py`
- runtime builders live in `models/factories.py`
- `base.py` rebuilds runtime through the builder layer, not through the registry facade
- `registry.py` keeps the same public helper names

**Step 4: Run the targeted registry and serialization tests**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py tests/test_serialization.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/models/specs.py src/foresight/models/factories.py src/foresight/models/registry.py src/foresight/base.py tests/test_models_registry.py tests/test_serialization.py
git commit -m "refactor: split model specs from runtime builders"
```

---

### Task 5: Introduce Catalog Shards And Keep `registry.py` As A Facade

**Files:**
- Create: `src/foresight/models/catalog/__init__.py`
- Create: `src/foresight/models/catalog/classical.py`
- Create: `src/foresight/models/catalog/ml.py`
- Create: `src/foresight/models/catalog/stats.py`
- Create: `src/foresight/models/catalog/torch_local.py`
- Create: `src/foresight/models/catalog/torch_global.py`
- Create: `src/foresight/models/catalog/multivariate.py`
- Create: `src/foresight/models/catalog/foundation.py`
- Modify: `src/foresight/models/registry.py`
- Test: `tests/test_models_registry.py`
- Test: `tests/test_models_registry_more_models.py`
- Test: `tests/test_cli_models.py`

**Step 1: Add a failing regression test that samples models from multiple families**

```python
from foresight.models.registry import get_model_spec


def test_catalog_shards_preserve_cross_family_lookup() -> None:
    keys = ["naive-last", "ridge-lag", "arima", "torch-dlinear-direct", "var"]
    for key in keys:
        assert get_model_spec(key).key == key
```

**Step 2: Run the targeted tests**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py tests/test_models_registry_more_models.py tests/test_cli_models.py`
Expected: PASS before the move, then used as the refactor safety net

**Step 3: Move registry declarations into catalog shard modules**

Rules:

- `registry.py` should aggregate shard mappings only
- each shard owns metadata only
- public lookup and `make_*` helpers remain in `registry.py`
- delete moved declarations from `registry.py` as each shard lands

**Step 4: Re-run the targeted suite**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py tests/test_models_registry_more_models.py tests/test_cli_models.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/models/catalog/__init__.py src/foresight/models/catalog/classical.py src/foresight/models/catalog/ml.py src/foresight/models/catalog/stats.py src/foresight/models/catalog/torch_local.py src/foresight/models/catalog/torch_global.py src/foresight/models/catalog/multivariate.py src/foresight/models/catalog/foundation.py src/foresight/models/registry.py tests/test_models_registry.py tests/test_models_registry_more_models.py tests/test_cli_models.py
git commit -m "refactor: shard model catalog behind registry facade"
```

---

### Task 6: Extract Forecasting And Evaluation Services

**Files:**
- Create: `src/foresight/services/__init__.py`
- Create: `src/foresight/services/forecasting.py`
- Create: `src/foresight/services/evaluation.py`
- Modify: `src/foresight/forecast.py`
- Modify: `src/foresight/eval_forecast.py`
- Test: `tests/test_forecast_api.py`
- Test: `tests/test_eval_predictions.py`
- Test: `tests/test_cli_eval.py`
- Test: `tests/test_cli_forecast.py`

**Step 1: Write a failing service-level unit test**

```python
import pandas as pd

from foresight.services.forecasting import forecast_long_df


def test_forecast_service_runs_naive_last_on_long_df() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a"],
            "ds": [1, 2, 3],
            "y": [10.0, 11.0, 12.0],
        }
    )
    pred = forecast_long_df(model="naive-last", long_df=long_df, horizon=2)
    assert list(pred["yhat"]) == [12.0, 12.0]
```

**Step 2: Run the targeted suite to confirm the new services do not exist yet**

Run: `PYTHONPATH=src pytest -q tests/test_forecast_api.py tests/test_eval_predictions.py tests/test_cli_eval.py tests/test_cli_forecast.py`
Expected: PASS before the move, then used as the refactor safety net

**Step 3: Extract orchestration logic into `services.forecasting` and `services.evaluation`**

Rules:

- keep `forecast.py` and `eval_forecast.py` importable
- preserve public function names
- move workflow assembly, not package-level re-export behavior

**Step 4: Re-run the targeted suite**

Run: `PYTHONPATH=src pytest -q tests/test_forecast_api.py tests/test_eval_predictions.py tests/test_cli_eval.py tests/test_cli_forecast.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/services/__init__.py src/foresight/services/forecasting.py src/foresight/services/evaluation.py src/foresight/forecast.py src/foresight/eval_forecast.py tests/test_forecast_api.py tests/test_eval_predictions.py tests/test_cli_eval.py tests/test_cli_forecast.py
git commit -m "refactor: extract forecasting and evaluation services"
```

---

### Task 7: Add Architecture Boundary Checks And Developer Docs

**Files:**
- Create: `tools/check_architecture_imports.py`
- Create: `tests/test_architecture_boundaries.py`
- Create: `docs/ARCHITECTURE.md`
- Modify: `tests/test_release_tooling.py`

**Step 1: Write the failing architecture-boundary tests**

```python
from pathlib import Path
import subprocess
import sys


def test_architecture_import_check_passes() -> None:
    result = subprocess.run(
        [sys.executable, "tools/check_architecture_imports.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
```

**Step 2: Run the new boundary test**

Run: `PYTHONPATH=src pytest -q tests/test_architecture_boundaries.py`
Expected: FAIL because the script and docs do not exist yet

**Step 3: Implement a lightweight import-boundary checker and document the architecture**

The checker should enforce at least:

- `foresight.contracts` cannot import `foresight.services`
- `foresight.base` cannot import `foresight.models.registry`
- `foresight.cli` should not define duplicated `require_long_df` or covariate normalization helpers

**Step 4: Re-run the boundary and release-tooling tests**

Run: `PYTHONPATH=src pytest -q tests/test_architecture_boundaries.py tests/test_release_tooling.py`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/check_architecture_imports.py tests/test_architecture_boundaries.py tests/test_release_tooling.py docs/ARCHITECTURE.md
git commit -m "chore: add architecture guardrails"
```

---

### Task 8: Run The Refactor Verification Suite And Clean Up Temporary Wrappers

**Files:**
- Modify: `src/foresight/forecast.py`
- Modify: `src/foresight/eval_forecast.py`
- Modify: `src/foresight/base.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `docs/SOURCE_ENTRYPOINTS.md`

**Step 1: Remove any remaining dead wrappers or duplicate helper bodies discovered during the refactor**

Rules:

- keep public API compatibility
- delete only logic that is now fully owned by contracts, services, or factories

**Step 2: Run a broad verification subset**

Run: `PYTHONPATH=src pytest -q tests/test_root_import.py tests/test_cli.py tests/test_cli_eval.py tests/test_cli_forecast.py tests/test_forecast_api.py tests/test_eval_predictions.py tests/test_models_registry.py tests/test_models_registry_more_models.py tests/test_serialization.py tests/test_data_prep.py tests/test_release_tooling.py tests/test_architecture_boundaries.py`
Expected: PASS

**Step 3: Run static checks**

Run: `ruff check src tests tools`
Expected: PASS

**Step 4: Refresh the source-entrypoint docs**

Run: `sed -n '1,220p' docs/SOURCE_ENTRYPOINTS.md`
Expected: update needed if file paths or entry descriptions changed

**Step 5: Commit**

```bash
git add src/foresight/forecast.py src/foresight/eval_forecast.py src/foresight/base.py src/foresight/models/registry.py docs/SOURCE_ENTRYPOINTS.md
git commit -m "docs: finalize internal architecture refactor"
```

---

### Task 9: Final Package-Level Verification

**Files:**
- Verify only

**Step 1: Run the full fast-package verification bundle**

Run: `PYTHONPATH=src pytest -q`
Expected: PASS, or a clearly scoped list of optional-dependency failures that must be resolved before merge

**Step 2: Run typing on the tracked subset**

Run: `PYTHONPATH=src mypy`
Expected: PASS for the configured file set in `pyproject.toml`

**Step 3: Run the architecture checker directly**

Run: `python tools/check_architecture_imports.py`
Expected: PASS

**Step 4: Record the resulting diff shape**

Run: `git diff --stat main...HEAD`
Expected: contracts, services, model-architecture files, and docs touched; no accidental algorithm churn outside the refactor scope

**Step 5: Commit nothing in this task**

This task is verification only. If any step fails, fix the failure before merge or PR preparation.
