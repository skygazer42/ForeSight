# Package Modularization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor ForeSight into a cleaner internal architecture with canonical contracts, split model lookup/runtime modules, shared execution services, and thinner public facades while keeping public imports, CLI commands, and model keys backward compatible.

**Architecture:** The refactor keeps the package as a single distribution and preserves the current public surface, but moves internal ownership to four explicit layers: contracts, models, services, and facades. The main internal changes are a typed covariate contract, a `models/resolution.py` plus `models/runtime.py` split behind `models/registry.py`, a shared `services/model_execution.py` adapter, and stricter architecture guardrails to keep the codebase from sliding back into mixed responsibilities.

**Tech Stack:** Python 3.10+, NumPy, Pandas, argparse, setuptools, pytest, ruff

---

## Constraints

- Preserve `foresight` CLI command names and existing arguments.
- Preserve public imports from `foresight`.
- Preserve existing model keys and model selection behavior.
- Prefer additive internal modules plus compatibility facades over big-bang rewrites.
- Keep each task independently shippable.

### Task 1: Tighten The Architecture Guardrails

**Files:**
- Modify: `tools/check_architecture_imports.py`
- Modify: `tests/test_architecture_boundaries.py`
- Modify: `docs/ARCHITECTURE.md`

**Step 1: Write the failing architecture test coverage**

Add focused tests for these rules:

- `forecast.py` and `eval_forecast.py` must not export underscore-prefixed helpers in `__all__`
- `services/*` must not import `foresight.forecast`, `foresight.eval_forecast`, or `foresight.cli`
- `models/registry.py` should remain a facade and must not define new catalog helper families that belong in resolution/runtime

Example test shape:

```python
from pathlib import Path


def test_forecast_facade_does_not_export_private_helpers() -> None:
    text = Path("src/foresight/forecast.py").read_text(encoding="utf-8")
    assert '"_require_long_df"' not in text
```

**Step 2: Run the architecture tests to confirm new coverage fails first**

Run: `PYTHONPATH=src pytest -q tests/test_architecture_boundaries.py`
Expected: FAIL because the current facades still re-export private helpers

**Step 3: Extend the checker script**

Implement AST-based checks for the new rules inside `tools/check_architecture_imports.py` and reflect the intended layering in `docs/ARCHITECTURE.md`.

**Step 4: Re-run the architecture tests**

Run: `PYTHONPATH=src pytest -q tests/test_architecture_boundaries.py`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/check_architecture_imports.py tests/test_architecture_boundaries.py docs/ARCHITECTURE.md
git commit -m "test: tighten package architecture guardrails"
```

### Task 2: Introduce A Canonical Covariate Contract

**Files:**
- Create: `src/foresight/contracts/covariates.py`
- Modify: `src/foresight/contracts/__init__.py`
- Modify: `src/foresight/contracts/params.py`
- Modify: `src/foresight/data/format.py`
- Modify: `src/foresight/data/prep.py`
- Modify: `src/foresight/services/forecasting.py`
- Modify: `src/foresight/services/evaluation.py`
- Test: `tests/test_contracts_params.py`
- Create: `tests/test_contracts_covariates.py`

**Step 1: Write the failing covariate tests**

Add tests that define the new canonical behavior:

```python
from foresight.contracts.covariates import CovariateSpec, resolve_covariates


def test_resolve_covariates_merges_legacy_x_cols() -> None:
    spec = resolve_covariates({"x_cols": "promo,price", "historic_x_cols": ("stock",)})
    assert isinstance(spec, CovariateSpec)
    assert spec.historic_x_cols == ("stock",)
    assert spec.future_x_cols == ("promo", "price")
    assert spec.all_x_cols == ("promo", "price", "stock")
```

**Step 2: Run the covariate-focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_contracts_params.py tests/test_contracts_covariates.py`
Expected: FAIL because `covariates.py` and `CovariateSpec` do not exist yet

**Step 3: Implement the minimal canonical contract**

Required behavior:

- normalize legacy `x_cols`
- preserve distinct historic vs future roles
- expose one typed object consumed by services/data helpers
- keep compatibility helpers in `params.py` delegating to the canonical implementation

**Step 4: Rewire service/data callers**

Replace tuple-based covariate-role logic in:

- `data/format.py`
- `data/prep.py`
- `services/forecasting.py`
- `services/evaluation.py`

with direct use of the typed contract or compatibility wrappers backed by it.

**Step 5: Re-run the targeted tests**

Run: `PYTHONPATH=src pytest -q tests/test_contracts_params.py tests/test_contracts_covariates.py tests/test_data_prep.py tests/test_eval_local_xreg.py`
Expected: PASS

**Step 6: Commit**

```bash
git add src/foresight/contracts/covariates.py src/foresight/contracts/__init__.py src/foresight/contracts/params.py src/foresight/data/format.py src/foresight/data/prep.py src/foresight/services/forecasting.py src/foresight/services/evaluation.py tests/test_contracts_params.py tests/test_contracts_covariates.py
git commit -m "refactor: add canonical covariate contract"
```

### Task 3: Make The Frame Contract The Single Source Of Truth

**Files:**
- Modify: `src/foresight/contracts/frames.py`
- Modify: `src/foresight/base.py`
- Modify: `src/foresight/data/prep.py`
- Modify: `src/foresight/services/forecasting.py`
- Modify: `src/foresight/services/evaluation.py`
- Test: `tests/test_contracts_frames.py`
- Test: `tests/test_data_contracts_compat.py`

**Step 1: Add one failing compatibility test across surfaces**

```python
import pandas as pd
import pytest

from foresight.forecast import forecast_model_long_df
from foresight.eval_forecast import eval_model_long_df


def test_forecast_and_eval_share_long_df_contract_errors() -> None:
    bad = pd.DataFrame({"unique_id": ["a"], "ds": [1]})
    with pytest.raises(KeyError):
        forecast_model_long_df(model="naive-last", long_df=bad, horizon=1)
    with pytest.raises(KeyError):
        eval_model_long_df(
            model="naive-last",
            long_df=bad,
            horizon=1,
            step=1,
            min_train_size=1,
        )
```

**Step 2: Run the frame-contract tests**

Run: `PYTHONPATH=src pytest -q tests/test_contracts_frames.py tests/test_data_contracts_compat.py`
Expected: FAIL or expose drift between surfaces

**Step 3: Centralize all shared frame checks in `contracts/frames.py`**

That module should own:

- required column checks
- non-empty handling
- duplicate `(unique_id, ds)` protection
- sort normalization where needed
- future/history merge semantics
- observed-history-only validation

Callers should import the contract directly or use a very thin compatibility shim only when necessary.

**Step 4: Re-run the targeted frame suite**

Run: `PYTHONPATH=src pytest -q tests/test_contracts_frames.py tests/test_data_contracts_compat.py tests/test_forecast_api.py tests/test_eval_local_xreg.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/contracts/frames.py src/foresight/base.py src/foresight/data/prep.py src/foresight/services/forecasting.py src/foresight/services/evaluation.py tests/test_contracts_frames.py tests/test_data_contracts_compat.py
git commit -m "refactor: centralize shared frame contract"
```

### Task 4: Split Model Resolution From Model Runtime

**Files:**
- Create: `src/foresight/models/resolution.py`
- Create: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/base.py`
- Test: `tests/test_models_registry.py`
- Test: `tests/test_serialization.py`

**Step 1: Write the failing lookup/runtime tests**

Add regression coverage that proves the split without changing public behavior:

```python
from foresight.models.registry import get_model_spec, make_forecaster


def test_registry_compatibility_survives_resolution_runtime_split() -> None:
    spec = get_model_spec("naive-last")
    forecaster = make_forecaster("naive-last")
    assert spec.key == "naive-last"
    assert forecaster([1.0, 2.0, 3.0], 1).shape == (1,)
```

**Step 2: Run the targeted model tests**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py tests/test_serialization.py`
Expected: FAIL once the tests start importing `resolution.py` or `runtime.py`

**Step 3: Implement `models/resolution.py`**

Move in:

- catalog assembly
- `list_models`
- `get_model_spec`
- any pure metadata/capability lookup helpers

**Step 4: Implement `models/runtime.py`**

Move in:

- local/global/multivariate constructor dispatch
- object-builder wrappers if they are truly runtime concerns
- rebuild hooks used by `base.py`

**Step 5: Reduce `registry.py` to a compatibility facade**

It should mainly forward the public helpers to `resolution.py` and `runtime.py`.

**Step 6: Re-run the targeted suite**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py tests/test_serialization.py tests/test_cli_models.py`
Expected: PASS

**Step 7: Commit**

```bash
git add src/foresight/models/resolution.py src/foresight/models/runtime.py src/foresight/models/registry.py src/foresight/base.py tests/test_models_registry.py tests/test_serialization.py
git commit -m "refactor: split model resolution from runtime dispatch"
```

### Task 5: Unify Forecast, Eval, And CV Model Execution

**Files:**
- Create: `src/foresight/services/model_execution.py`
- Modify: `src/foresight/services/forecasting.py`
- Modify: `src/foresight/services/evaluation.py`
- Modify: `src/foresight/cv.py`
- Modify: `src/foresight/base.py`
- Test: `tests/test_forecast_api.py`
- Test: `tests/test_eval_local_xreg.py`
- Test: `tests/test_cv_predictions.py`

**Step 1: Write the failing execution-adapter test**

Add a test around one shared execution behavior, for example local xreg or interval-aware local execution, and assert that forecast/eval paths both use the same semantics.

Example skeleton:

```python
def test_local_xreg_execution_path_is_shared() -> None:
    ...
```

**Step 2: Run the targeted execution tests**

Run: `PYTHONPATH=src pytest -q tests/test_forecast_api.py tests/test_eval_local_xreg.py tests/test_cv_predictions.py`
Expected: FAIL after introducing the new shared adapter but before callers are rewired

**Step 3: Implement `services/model_execution.py`**

This module should own:

- local callable execution
- local xreg execution
- global execution dispatch
- runtime reconstruction hooks used by services or artifacts

**Step 4: Rewire callers**

Update:

- `services/forecasting.py`
- `services/evaluation.py`
- `cv.py`

to call the shared execution adapter instead of carrying local helper copies.

**Step 5: Re-run the targeted suite**

Run: `PYTHONPATH=src pytest -q tests/test_forecast_api.py tests/test_eval_local_xreg.py tests/test_cv_predictions.py tests/test_serialization.py`
Expected: PASS

**Step 6: Commit**

```bash
git add src/foresight/services/model_execution.py src/foresight/services/forecasting.py src/foresight/services/evaluation.py src/foresight/cv.py src/foresight/base.py
git commit -m "refactor: share model execution across workflows"
```

### Task 6: Shrink The Public Facades And CLI Orchestration

**Files:**
- Modify: `src/foresight/forecast.py`
- Modify: `src/foresight/eval_forecast.py`
- Modify: `src/foresight/cli.py`
- Create: `src/foresight/services/cli_workflows.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_cli_eval.py`
- Test: `tests/test_cli_forecast.py`

**Step 1: Write the failing facade/CLI compatibility test**

Add a regression that asserts:

- public facade modules expose only supported public entrypoints
- CLI commands still produce the same output shape for representative forecast/eval commands

**Step 2: Run the targeted CLI suite**

Run: `PYTHONPATH=src pytest -q tests/test_cli.py tests/test_cli_eval.py tests/test_cli_forecast.py`
Expected: FAIL until CLI orchestration is moved out of `cli.py`

**Step 3: Remove private helper re-exports from public facades**

Keep only:

- `forecast_model`
- `forecast_model_long_df`
- `eval_model`
- `eval_model_long_df`
- `eval_multivariate_model_df`
- `eval_hierarchical_forecast_df`

**Step 4: Move CLI orchestration into service-owned helpers**

Create `services/cli_workflows.py` for:

- forecast/eval from CSV or long data
- artifact prediction flow
- leaderboard or sweep orchestration helpers when needed

`cli.py` should parse args, call a single helper, and format output.

**Step 5: Re-run the targeted CLI suite**

Run: `PYTHONPATH=src pytest -q tests/test_cli.py tests/test_cli_eval.py tests/test_cli_forecast.py tests/test_cli_leaderboard.py`
Expected: PASS

**Step 6: Commit**

```bash
git add src/foresight/forecast.py src/foresight/eval_forecast.py src/foresight/cli.py src/foresight/services/cli_workflows.py
git commit -m "refactor: reduce public facades and slim CLI orchestration"
```

### Final Verification

Run the focused regression suite:

```bash
PYTHONPATH=src pytest -q \
  tests/test_architecture_boundaries.py \
  tests/test_contracts_frames.py \
  tests/test_contracts_params.py \
  tests/test_contracts_covariates.py \
  tests/test_data_contracts_compat.py \
  tests/test_forecast_api.py \
  tests/test_eval_local_xreg.py \
  tests/test_cv_predictions.py \
  tests/test_models_registry.py \
  tests/test_serialization.py \
  tests/test_cli.py \
  tests/test_cli_eval.py \
  tests/test_cli_forecast.py \
  tests/test_cli_models.py
```

Expected: PASS

Then run broad package verification:

```bash
PYTHONPATH=src pytest -q
```

Expected: PASS
