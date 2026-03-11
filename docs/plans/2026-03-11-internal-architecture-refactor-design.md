# ForeSight Internal Architecture Refactor Design

**Date:** 2026-03-11

**Goal:** Make ForeSight feel like a coherent Python package again by introducing clear internal layers, removing duplicated data-contract logic, and shrinking the responsibilities of the registry and CLI without breaking the public API or CLI behavior.

**Scope:** This design covers package layering, model catalog/factory separation, shared contract extraction, service extraction, compatibility strategy, and architecture guardrails for the first internal refactor wave.

---

## 1. Context

ForeSight already has substantial user-facing value:

- a large forecasting model surface
- a unified Python API
- a CLI with discovery, evaluation, forecasting, and tuning workflows
- broad test coverage across models and package-level behavior

The current maintenance problem is not lack of features. It is that too many responsibilities are collapsing into a few coordination files:

- `src/foresight/models/registry.py`
- `src/foresight/cli.py`
- `src/foresight/forecast.py`
- `src/foresight/eval_forecast.py`

The result is visible in three ways:

1. Shared data rules are duplicated.
   - `long_df` validation appears in multiple modules.
   - covariate normalization logic is duplicated between forecast and evaluation flows.
   - interval and quantile parsing live near orchestration logic instead of a shared contract layer.

2. Coordination layers are too thick.
   - the registry holds metadata, defaults, capability inference, and factory construction
   - the CLI owns too much runtime coordination instead of remaining a thin interface
   - forecast and evaluation entrypoints mix validation, model wiring, and output shaping

3. Internal dependencies are harder to reason about than they need to be.
   - object wrappers in `base.py` rebuild runtime state by reaching back into the registry facade
   - changes to model registration or covariate semantics can force edits across unrelated files

ForeSight therefore needs an internal architecture pass before more large feature waves land.

---

## 2. Non-Goals

This refactor wave is intentionally conservative.

It does **not** attempt to:

- rename public Python API functions
- rename CLI commands or flags
- rename registered model keys
- change output column names or payload structure
- rewrite individual model implementations for style consistency
- split the repository into multiple packages
- replace `argparse`, `setuptools`, or the existing packaging model

This is an internal structure refactor with compatibility as the first constraint.

---

## 3. Design Principles

The refactor should enforce the following principles:

### One layer, one reason to change

- contracts change when data rules change
- catalog changes when model metadata changes
- factories change when runtime construction changes
- services change when workflow orchestration changes
- interfaces change when CLI or top-level API experience changes

### Public stability, internal mobility

Keep these public entrypoints stable:

- `foresight.__init__`
- `foresight.forecast`
- `foresight.eval_forecast`
- `foresight.cli`
- `foresight.models.registry`

The public modules may become facades, but they should continue to exist and preserve behavior.

### Shared rules must have one source of truth

Data contract rules should not be reimplemented across multiple call sites.

### Prefer facade migration over big-bang rewrite

Each existing public module remains in place while its internals are gradually redirected to new lower-level modules.

---

## 4. Target Layering

The target package shape is:

```text
foresight/
  contracts/
    frames.py
    params.py
    capabilities.py
  services/
    forecasting.py
    evaluation.py
    tuning.py
  models/
    specs.py
    factories.py
    catalog/
      classical.py
      ml.py
      torch_local.py
      torch_global.py
      multivariate.py
      foundation.py
    registry.py
  cli.py
  forecast.py
  eval_forecast.py
  __init__.py
```

Allowed dependency direction:

- `interfaces/facades` may depend on `services`, `contracts`, and `models.registry`
- `services` may depend on `contracts`, `models.registry`, metrics, splits, and existing algorithm modules
- `models.registry` may depend on `models.specs`, `models.factories`, and `models.catalog.*`
- `models.catalog.*` may depend on algorithm implementation modules only
- `contracts` may depend on `numpy`, `pandas`, and standard library only
- `contracts` must not depend on services, CLI, or registry facades

This keeps the lowest layers stable and reusable.

---

## 5. Contracts Layer

### Problem

The same rules currently exist in multiple places:

- `long_df` validation
- `future_df` validation
- covariate role normalization
- quantile and interval-level parsing
- some capability-based argument checks

That duplication causes behavior drift. It also makes simple rule changes expensive.

### Proposed modules

#### `src/foresight/contracts/frames.py`

Owns frame-level validation and shaping:

- `require_long_df(...)`
- `require_future_df(...)`
- `merge_history_and_future_df(...)`
- `require_observed_history_only(...)`
- `sort_long_df(...)`

#### `src/foresight/contracts/params.py`

Owns argument normalization:

- `normalize_model_params(...)`
- `normalize_x_cols(...)`
- `normalize_covariate_roles(...)`
- `parse_interval_levels(...)`
- `parse_quantiles(...)`
- `required_quantiles_for_interval_levels(...)`

#### `src/foresight/contracts/capabilities.py`

Owns runtime checks derived from model capabilities:

- `require_x_cols_if_needed(...)`
- optional small helpers for interval or artifact support gating

### Migration strategy

Existing private helpers in:

- `src/foresight/forecast.py`
- `src/foresight/eval_forecast.py`
- `src/foresight/base.py`
- `src/foresight/data/prep.py`

should initially remain as thin wrappers over the new contract functions. That preserves import paths and minimizes the first regression surface.

---

## 6. Model Metadata And Factory Split

### Problem

`src/foresight/models/registry.py` is currently both a data container and a runtime assembly layer. It mixes:

- `ModelSpec`
- capability inference
- default parameter bundles
- registry population
- local/global/multivariate dispatch
- persistent object wrappers

This is the highest-coupling internal file in the package.

### Proposed split

#### `src/foresight/models/specs.py`

Owns:

- `ModelSpec`
- small type aliases shared by the model registry surface
- capability derivation helpers that belong to the spec abstraction

#### `src/foresight/models/factories.py`

Owns:

- `build_local_forecaster(...)`
- `build_global_forecaster(...)`
- `build_multivariate_forecaster(...)`
- object-wrapper construction helpers used by serialization/runtime restore paths

#### `src/foresight/models/catalog/*.py`

Owns model declarations only.

Suggested sharding:

- `classical.py`
- `ml.py`
- `stats.py`
- `torch_local.py`
- `torch_global.py`
- `multivariate.py`
- `foundation.py`

Each catalog shard should export a mapping of model key to `ModelSpec`. Catalog files are allowed to import the algorithm implementation functions they register. They should not own public runtime dispatch functions.

#### `src/foresight/models/registry.py`

Becomes a compatibility facade that:

- aggregates the catalog shards
- exposes `list_models`, `get_model_spec`, and `make_*` functions
- preserves current public behavior

### `base.py` direction

`src/foresight/base.py` should no longer reach into the registry facade to rebuild runtime state. Instead, it should depend on a stable builder interface from `models.factories`.

That removes a layering inversion and keeps object wrappers dependent on runtime construction, not on registry organization.

---

## 7. Service Layer

### Problem

`forecast.py` and `eval_forecast.py` currently act as both public API modules and orchestration implementations. They perform validation, capability checks, model wiring, prediction shaping, and metric aggregation.

That makes the public entrypoints harder to read and harder to keep consistent.

### Proposed modules

#### `src/foresight/services/forecasting.py`

Owns forecast workflow orchestration:

- prepare inputs
- resolve model specs
- dispatch local/global forecasting paths
- attach intervals or quantiles
- return final forecast frames or arrays

#### `src/foresight/services/evaluation.py`

Owns evaluation orchestration:

- dataset loading or input normalization
- split generation
- walk-forward execution
- metric aggregation
- conformal/hierarchical result assembly

#### `src/foresight/services/tuning.py`

This can be added in the same wave if time allows. If not, it should be the next obvious extraction after forecasting and evaluation.

### Facade behavior

Public modules stay in place:

- `src/foresight/forecast.py`
- `src/foresight/eval_forecast.py`

They become thin facades over the service modules plus any public API compatibility helpers that need to remain importable.

---

## 8. CLI Strategy

`src/foresight/cli.py` should remain the command tree owner for now, but it should lose business rules over time.

Phase 1 CLI goal:

- continue using the existing parser structure
- move reusable workflow logic into services
- keep CLI responsible for argument parsing, formatting, and exit behavior only

This deliberately avoids a risky full CLI rewrite. The change is to make `cli.py` thinner by extraction, not by replacing the interface.

---

## 9. Guardrails

The refactor only holds if the new boundaries become testable.

### Compatibility tests

Add or strengthen tests that lock:

- public factory dispatch behavior
- forecast/evaluation return shapes and column names
- representative error messages for bad `long_df`, bad `future_df`, and missing covariates
- CLI smoke behavior for list/eval/forecast commands

### Boundary checks

Add a lightweight repository script:

- `tools/check_architecture_imports.py`

It should fail when forbidden dependencies appear, for example:

- `contracts` importing `services`
- `cli.py` reintroducing local data-contract helpers
- `base.py` importing `models.registry` instead of the stable builder layer

### Developer documentation

Add:

- `docs/ARCHITECTURE.md`

This document should describe:

- current package layers
- where new helpers belong
- where model metadata belongs
- which modules are facades versus implementation layers

---

## 10. Phased Rollout

### Phase 0: Baseline snapshot

- create a dedicated refactor worktree
- run a focused compatibility baseline
- record the expected green subset before moving code

### Phase 1: Shared contracts extraction

- create `contracts` package
- move duplicated validation and normalization helpers there
- rewire existing callers through thin wrappers

### Phase 2: Registry split

- move `ModelSpec` and builder logic out of `registry.py`
- keep `registry.py` as public facade
- start moving registry declarations into catalog shards

### Phase 3: Service extraction

- create `services.forecasting` and `services.evaluation`
- rewire public modules and CLI handlers to call services

### Phase 4: Guardrails and cleanup

- add architecture boundary checks
- add developer docs
- delete dead wrapper code once all callers use the new layers

This ordering favors low-risk, high-leverage changes first.

---

## 11. Risks And Mitigations

### Risk: behavior drift while extracting helpers

Mitigation:

- add compatibility tests before moving logic
- preserve existing error messages where practical
- keep old private helpers as wrappers during the transition

### Risk: import cycles while adding services

Mitigation:

- keep `contracts` dependency-light
- move runtime construction into `models.factories`
- make public modules facades, not shared dependency sinks

### Risk: optional-dependency regressions

Mitigation:

- run focused optional-dependency tests around registry and model dispatch
- avoid changing algorithm implementation modules in the first wave unless required

### Risk: partial migration leaving two sources of truth

Mitigation:

- every extraction step must delete or thin the old implementation immediately
- do not leave duplicated logic in parallel for more than one task boundary

---

## 12. Success Criteria

This refactor wave is successful when:

- public Python API and CLI behavior remain compatible
- shared data-contract logic exists in one place
- `base.py` no longer depends on the registry facade for runtime restoration
- `registry.py` becomes a public facade rather than the home for every responsibility
- `forecast.py` and `eval_forecast.py` are noticeably thinner and primarily delegate to services
- an architecture boundary check exists and passes in CI-quality local verification

At that point ForeSight will be in a much better position to keep adding model families without making the package harder to maintain.
