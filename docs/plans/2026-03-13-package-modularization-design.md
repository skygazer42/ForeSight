# ForeSight Package Modularization Design

**Date:** 2026-03-13

**Status:** Validated with the user

**Supersedes when conflicting:** `docs/plans/2026-03-11-internal-architecture-refactor-design.md`, `docs/plans/2026-03-11-internal-architecture-refactor-implementation.md`, `docs/plans/2026-03-12-package-cohesion-architecture-plan.md`

**Goal:** Turn ForeSight into a more coherent package by finishing the separation between contracts, model resolution/runtime, workflow services, and public facades while keeping the CLI, public imports, and model keys backward compatible.

---

## 1. Problem Statement

ForeSight already has strong package value: broad model coverage, a usable CLI, packaged datasets, and a stable-looking top-level API. The weakness is internal coherence, not missing algorithms.

Three issues are now structural:

1. **Shared rules are duplicated.**
   Long-frame validation, `x_cols` normalization, interval parsing, and related helper behavior appear in multiple layers. That creates drift in error messages, hidden behavior changes, and uneven coding style.

2. **Coordination files still own too much.**
   `src/foresight/models/registry.py` remains a high-coupling module. Public facades such as `forecast.py` and `eval_forecast.py` still forward many internal helpers. `cli.py` still performs orchestration that belongs in service code.

3. **The package shape and the runtime shape do not fully match.**
   The repository already documents layers in `docs/ARCHITECTURE.md`, but several modules still bypass those boundaries through thin wrappers or compatibility shortcuts. That makes the package feel less like one designed library and more like several merged code paths.

This refactor is intended to correct those architectural mismatches without changing the user-facing product surface.

## 2. Explicit Constraints

The user approved the following constraints for this wave:

- Keep CLI command names and argument names backward compatible.
- Keep public imports from `foresight` backward compatible.
- Keep registered model keys backward compatible.
- Internal modules may be reorganized aggressively as long as the public package surface stays stable.

This design is therefore a **moderate modularization** pass, not a rewrite and not a packaging split.

## 3. Target Internal Shape

The target architecture has four internal layers.

### Contracts

Owns canonical validation and normalization rules only.

Primary files:

- `src/foresight/contracts/frames.py`
- `src/foresight/contracts/params.py`
- `src/foresight/contracts/covariates.py`
- `src/foresight/contracts/capabilities.py`

Responsibilities:

- long/future frame validation
- merge semantics for history + future input
- canonical covariate role normalization
- interval and quantile parsing
- capability-based argument checks

The contracts layer must be dependency-light and must not import services, CLI code, or public facades.

### Models

Split into metadata, lookup, and runtime construction.

Primary files:

- `src/foresight/models/catalog/*.py`
- `src/foresight/models/resolution.py`
- `src/foresight/models/runtime.py`
- `src/foresight/models/registry.py`

Responsibilities:

- `catalog`: declarative metadata only
- `resolution`: list, lookup, and resolve model specs/capabilities
- `runtime`: build local/global/multivariate runtime callables and objects
- `registry`: compatibility facade exposing the existing public helpers

This is the core structural change for package cohesion. `registry.py` should stop acting like a god module and become a stable compatibility entrypoint.

### Services

Own workflow orchestration only.

Primary files:

- `src/foresight/services/forecasting.py`
- `src/foresight/services/evaluation.py`
- `src/foresight/services/model_execution.py`
- CLI-oriented service helpers created only if needed

Responsibilities:

- forecast/eval/cv workflow assembly
- common execution semantics across local/global/xreg paths
- payload shaping and result assembly
- no duplicated frame or parameter normalization logic

The service layer may depend on contracts, model resolution/runtime, metrics, splits, datasets, and algorithm modules.

### Facades

Public modules remain import-stable but become thin.

Primary files:

- `src/foresight/__init__.py`
- `src/foresight/forecast.py`
- `src/foresight/eval_forecast.py`
- `src/foresight/cli.py`

Responsibilities:

- expose public entrypoints
- map CLI arguments to service calls
- avoid re-exporting private internal helpers

After this refactor, these files should read like package boundaries, not like secondary implementations.

## 4. Dependency Rules

The following dependency rules are the intended enforcement target:

- `contracts` must not import `services`, `cli`, `forecast`, or `eval_forecast`.
- `models/catalog` may depend on algorithm implementations and model spec types, but not on services.
- `models/resolution` may depend on catalog and model specs, but not on service workflows.
- `models/runtime` may depend on factories/builders and algorithm implementations, but not on public facades.
- `services` may depend on contracts, model resolution/runtime, and lower-level domain helpers.
- `forecast.py`, `eval_forecast.py`, and `cli.py` may depend on services, but services should not depend back on those public facades.

These rules matter because style inconsistency in this package is mostly a symptom of unclear ownership. Stronger import boundaries are the mechanism that forces consistent code placement.

## 5. Canonical Data Contract

The package should have one canonical data contract and one canonical covariate contract.

### Data contract

`contracts/frames.py` should become the single source of truth for:

- required columns for long-format data
- whether empty frames are allowed in each context
- sort guarantees
- duplicate `(unique_id, ds)` handling
- future-tail validation rules
- history/future merge behavior

Today multiple modules expose wrappers around the same validation logic. Those wrappers should be removed or collapsed to direct imports unless backward compatibility requires a tiny shim.

### Covariate contract

Introduce a typed object, for example `CovariateSpec`, to replace scattered tuple-based helper behavior.

It should carry:

- `historic_x_cols`
- `future_x_cols`
- `all_x_cols`
- compatibility handling for legacy `x_cols`

This change is important because covariate handling is currently one of the main sources of duplicated helper logic and implicit behavior.

## 6. Model Resolution And Runtime Split

ForeSight already has `models/specs.py`, `models/factories.py`, and `models/catalog/`, but the effective runtime path is still too registry-centric.

This design therefore does **not** propose another broad factory extraction. Instead it adds a clearer split on top of the current model layer:

- `resolution.py` owns model listing, lookup, and capability resolution
- `runtime.py` owns construction of callable/object runtime instances
- `registry.py` forwards to those modules for compatibility

That allows:

- thinner imports in callers
- better testing of metadata lookup vs runtime construction
- simpler serialization/runtime rebuild semantics in `base.py`
- less pressure on `registry.py` to know everything

## 7. Workflow Unification

Forecasting, evaluation, and CV currently share conceptual execution steps but do not fully share execution code. The moderate modularization path should unify those semantics through a dedicated internal execution adapter in `services/model_execution.py`.

That adapter should own:

- local callable execution
- local xreg execution
- global forecaster execution
- interval-aware execution hooks
- model object reconstruction hooks needed by serialization and artifact usage

Once that exists, `forecasting.py` and `evaluation.py` can differ by workflow shape and output formatting rather than by partially duplicated execution helpers.

## 8. Migration Strategy

The migration should happen in this order:

1. tighten architecture guardrails and tests
2. create the canonical covariate/data contracts
3. split model lookup from model runtime
4. unify service-layer model execution
5. shrink facades and move CLI orchestration behind service helpers

This order is intentionally conservative. It blocks further architecture drift first, then improves the shared internals, then rewires public layers last.

## 9. Acceptance Criteria

This design is successful when all of the following are true:

- public imports and CLI behavior remain compatible
- `models/registry.py` is materially smaller and no longer the default dependency hub
- contracts and covariate handling each have one canonical home
- services no longer duplicate execution helpers
- facades expose supported entrypoints only
- architecture tests fail when a layer reintroduces forbidden imports or duplicated helper ownership

## 10. Risks And Non-Goals

This wave will not solve every package-quality issue.

It will not:

- normalize implementation style across every algorithm module
- split the repository into multiple distributions
- redesign all model declarations
- rewrite `cli.py` into a different framework

Main risks:

- regression risk from changing hidden helper dependencies
- serialization behavior drifting if runtime reconstruction changes without enough tests
- CLI breakage if private helpers are removed before service replacements exist

Those risks are manageable if the refactor is executed in small, verified phases rather than as one large patch.
