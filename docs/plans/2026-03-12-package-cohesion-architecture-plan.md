# Package Cohesion Architecture Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make ForeSight feel like one coherent package by tightening layer boundaries, removing duplicated contracts, shrinking the registry and CLI, and standardizing the runtime path for forecasting and evaluation.

**Architecture:** Keep the public API and model keys stable, but finish the internal split into facades, services, contracts/data, and model composition layers. The refactor should move private helpers behind stable internal interfaces, make data contracts single-source, and turn `models/registry.py` into a thin facade instead of a god module.

**Tech Stack:** Python 3.10+, NumPy, Pandas, argparse, setuptools, pytest (verification only when explicitly authorized by the user)

---

## Constraints

- The current worktree is already dirty with in-progress model additions. Do not mix those edits with this refactor. Use a dedicated worktree or branch for implementation.
- Public imports, CLI command names, and registered model keys should remain backward compatible during this wave.
- The user explicitly requested: do **not** run tests unless they later ask for testing. Verification commands are listed below for future use, but execution is gated by user approval.

### Task 1: Freeze The Public Boundary And Stop Re-Exporting Internals

**Files:**
- Modify: `src/foresight/forecast.py`
- Modify: `src/foresight/eval_forecast.py`
- Modify: `src/foresight/cli.py`
- Create: `src/foresight/services/cli_forecasting.py`
- Create: `src/foresight/services/cli_evaluation.py`
- Create: `src/foresight/services/artifacts.py`
- Modify: `docs/ARCHITECTURE.md`

**Step 1:** Remove underscore-helper re-exports from `forecast.py` and `eval_forecast.py`.

Keep only the supported public entrypoints:

- `forecast_model`
- `forecast_model_long_df`
- `eval_model`
- `eval_model_long_df`
- `eval_multivariate_model_df`
- `eval_hierarchical_forecast_df`

Any helper still needed by the CLI should move to a service-owned internal module instead of being re-exported from a public facade.

**Step 2:** Move CLI-only orchestration out of `cli.py`.

Create service functions for:

- forecasting from CSV / DataFrame
- evaluation from CSV / DataFrame
- artifact prediction from serialized forecasters
- leaderboard / sweep orchestration

`cli.py` should parse args, call a single service function, and format output.

**Step 3:** Add a supported artifact runtime API.

Replace direct CLI access to object internals such as `_train_y` with a public method or service helper that can reconstruct the prediction context safely.

**Verification (deferred until user authorizes tests):**

- `PYTHONPATH=src pytest -q tests/test_forecast_api.py tests/test_eval_api.py tests/test_cli.py tests/test_serialization.py`

**Commit suggestion:**

```bash
git add src/foresight/forecast.py src/foresight/eval_forecast.py src/foresight/cli.py src/foresight/services/cli_forecasting.py src/foresight/services/cli_evaluation.py src/foresight/services/artifacts.py docs/ARCHITECTURE.md
git commit -m "refactor: narrow public facade and CLI boundary"
```

### Task 2: Create One Canonical Data Contract

**Files:**
- Modify: `src/foresight/contracts/frames.py`
- Modify: `src/foresight/contracts/params.py`
- Create: `src/foresight/contracts/covariates.py`
- Modify: `src/foresight/data/format.py`
- Modify: `src/foresight/data/prep.py`
- Modify: `src/foresight/services/forecasting.py`
- Modify: `src/foresight/services/evaluation.py`
- Modify: `src/foresight/cli.py`

**Step 1:** Define the actual canonical long-frame contract in one place.

`contracts/frames.py` should own the full shared rules for:

- required columns
- empty-frame handling
- duplicate timestamps per `unique_id`
- monotonic sort requirements
- observed-history-only vs future-tail layout
- history/future merge semantics

Service modules should not re-implement these checks.

**Step 2:** Replace the current split covariate-role logic with one typed contract.

Add a `CovariateSpec`-style object that carries:

- `historic_x_cols`
- `future_x_cols`
- `all_x_cols`
- compatibility handling for legacy `x_cols`

Stop using `DataFrame.attrs` as hidden behavior unless every consumer reads the same object directly.

**Step 3:** Make `data/format.py`, `data/prep.py`, `services/forecasting.py`, `services/evaluation.py`, and `cli.py` all consume the same contract helpers.

That includes interval/quantile parsing and covariate normalization.

**Verification (deferred until user authorizes tests):**

- `PYTHONPATH=src pytest -q tests/test_contracts_frames.py tests/test_contracts_params.py tests/test_data_prep.py tests/test_forecast_api.py tests/test_eval_local_xreg.py`

**Commit suggestion:**

```bash
git add src/foresight/contracts/frames.py src/foresight/contracts/params.py src/foresight/contracts/covariates.py src/foresight/data/format.py src/foresight/data/prep.py src/foresight/services/forecasting.py src/foresight/services/evaluation.py src/foresight/cli.py
git commit -m "refactor: unify data contracts and covariate handling"
```

### Task 3: Unify Forecast And Evaluation Model Execution

**Files:**
- Create: `src/foresight/services/model_execution.py`
- Modify: `src/foresight/services/forecasting.py`
- Modify: `src/foresight/services/evaluation.py`
- Modify: `src/foresight/base.py`
- Modify: `src/foresight/models/factories.py`

**Step 1:** Introduce one execution adapter for registered models.

This layer should own:

- local callable execution
- global forecaster execution
- local xreg execution
- artifact reconstruction hooks
- interval-aware execution shims where supported

**Step 2:** Route both forecasting and evaluation through the same adapter.

The difference between forecast and eval should be workflow shape, not model invocation semantics.

**Step 3:** Remove duplicate helpers once both services share the adapter.

Target duplicates include:

- local xreg execution helpers
- interval/level parsing helpers that do not belong in services
- object-vs-callable branching spread across multiple services

**Verification (deferred until user authorizes tests):**

- `PYTHONPATH=src pytest -q tests/test_forecast_api.py tests/test_eval_local_xreg.py tests/test_eval_multivariate.py tests/test_serialization.py`

**Commit suggestion:**

```bash
git add src/foresight/services/model_execution.py src/foresight/services/forecasting.py src/foresight/services/evaluation.py src/foresight/base.py src/foresight/models/factories.py
git commit -m "refactor: unify model execution across forecast and evaluation"
```

### Task 4: Split The Model Composition Layer By Responsibility

**Files:**
- Create: `src/foresight/models/builders/__init__.py`
- Create: `src/foresight/models/builders/shared.py`
- Create: `src/foresight/models/builders/classical.py`
- Create: `src/foresight/models/builders/ml_local.py`
- Create: `src/foresight/models/builders/ml_global.py`
- Create: `src/foresight/models/builders/torch_local.py`
- Create: `src/foresight/models/builders/torch_global.py`
- Create: `src/foresight/models/builders/multivariate.py`
- Create: `src/foresight/models/builders/foundation.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/catalog/classical.py`
- Modify: `src/foresight/models/catalog/ml.py`
- Modify: `src/foresight/models/catalog/torch_local.py`
- Modify: `src/foresight/models/catalog/torch_global.py`
- Modify: `src/foresight/models/catalog/multivariate.py`
- Modify: `src/foresight/models/catalog/foundation.py`

**Step 1:** Move `_factory_*` functions out of `registry.py` into builder modules grouped by real ownership.

`registry.py` should stop owning model adapters directly.

**Step 2:** Replace `context: Any` catalog building with explicit imports.

Each catalog shard should import:

- `ModelSpec`
- relevant shared defaults/help
- the factory functions it actually uses

Catalog shards should not pull attributes off the registry module object.

**Step 3:** Re-shard mixed files by interface.

Split the current mixed surfaces into:

- local ML vs global ML
- local torch vs global torch
- multivariate-only families in the multivariate shard

Do not leave global specs inside `torch_local.py`.

**Step 4:** Reduce `registry.py` to:

- catalog assembly
- `list_models`
- `get_model_spec`
- `make_forecaster`
- `make_global_forecaster`
- `make_multivariate_forecaster`
- object-builder wrappers

Composite meta-models such as pipelines and ensembles should live in a dedicated composition module if they remain registry-owned.

**Verification (deferred until user authorizes tests):**

- `PYTHONPATH=src pytest -q tests/test_models_registry.py tests/test_models_registry_more_models.py tests/test_serialization.py tests/test_cli_models.py`

**Commit suggestion:**

```bash
git add src/foresight/models/builders src/foresight/models/registry.py src/foresight/models/catalog/classical.py src/foresight/models/catalog/ml.py src/foresight/models/catalog/torch_local.py src/foresight/models/catalog/torch_global.py src/foresight/models/catalog/multivariate.py src/foresight/models/catalog/foundation.py
git commit -m "refactor: split model composition into builders and typed catalog shards"
```

### Task 5: Make ModelSpec Declarative And Stop Inferring Behavior From Docs

**Files:**
- Modify: `src/foresight/models/specs.py`
- Modify: `src/foresight/models/catalog/classical.py`
- Modify: `src/foresight/models/catalog/ml.py`
- Modify: `src/foresight/models/catalog/stats.py`
- Modify: `src/foresight/models/catalog/torch_local.py`
- Modify: `src/foresight/models/catalog/torch_global.py`
- Modify: `src/foresight/models/catalog/multivariate.py`
- Modify: `src/foresight/models/catalog/foundation.py`

**Step 1:** Add explicit declarative fields to `ModelSpec`.

Make the contract explicit for:

- interface
- capabilities
- dependency markers
- artifact support
- xreg requirements

**Step 2:** Remove capability inference from `param_help`.

Documentation should describe behavior, not define it.

**Step 3:** Standardize every spec declaration.

Every catalog entry should explicitly declare interface and capabilities, even when the value matches a default.

**Verification (deferred until user authorizes tests):**

- `PYTHONPATH=src pytest -q tests/test_models_registry.py tests/test_model_capabilities.py tests/test_cli_models.py`

**Commit suggestion:**

```bash
git add src/foresight/models/specs.py src/foresight/models/catalog/classical.py src/foresight/models/catalog/ml.py src/foresight/models/catalog/stats.py src/foresight/models/catalog/torch_local.py src/foresight/models/catalog/torch_global.py src/foresight/models/catalog/multivariate.py src/foresight/models/catalog/foundation.py
git commit -m "refactor: make model specs declarative"
```

### Task 6: Shrink The Public Models Package Surface

**Files:**
- Modify: `src/foresight/models/__init__.py`
- Create: `src/foresight/models/compat.py`
- Modify: `docs/api.md`
- Modify: `docs/ARCHITECTURE.md`

**Step 1:** Decide the primary public boundary for `foresight.models`.

Recommended main exports:

- `ModelSpec`
- `list_models`
- `get_model_spec`
- `make_forecaster`
- `make_forecaster_object`
- `make_global_forecaster`
- `make_global_forecaster_object`
- `make_multivariate_forecaster`

**Step 2:** Move raw implementation re-exports into `models/compat.py` if backward compatibility still requires them.

This keeps the public package stable without making the main entrypoint a barrel of every algorithm function.

**Step 3:** Document the preferred API.

The docs should clearly tell users to prefer registry constructors and service APIs over direct raw-model imports.

**Verification (deferred until user authorizes tests):**

- `PYTHONPATH=src pytest -q tests/test_root_import.py tests/test_models_registry.py tests/test_cli_models.py`

**Commit suggestion:**

```bash
git add src/foresight/models/__init__.py src/foresight/models/compat.py docs/api.md docs/ARCHITECTURE.md
git commit -m "refactor: narrow models package public surface"
```

## Recommended Implementation Order

1. Task 1: freeze the public boundary
2. Task 2: unify the data contract
3. Task 3: unify model execution
4. Task 4: split model composition and shrink registry
5. Task 5: make `ModelSpec` declarative
6. Task 6: shrink the public `foresight.models` surface

This order is intentional. It cuts invalid dependencies first, then standardizes shared contracts, then simplifies model composition once the upper layers stop reaching into private internals.

## Notes For Execution

- Do not implement this plan in the current dirty worktree.
- Do not add more model families while `registry.py` remains the main adapter/factory sink.
- Keep each task independently mergeable.
- Do not run tests until the user explicitly authorizes test execution.
