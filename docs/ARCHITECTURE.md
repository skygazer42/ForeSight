# Architecture

ForeSight now separates its package internals into a few stable layers so model growth does not keep dragging orchestration and validation concerns back into the same modules.

## Layers

`contracts`

- Shared frame validation, parameter normalization, interval parsing, and capability checks.
- These modules are dependency-light and should not import `services`, CLI code, or registry facades.

`models/specs.py`, `models/factories.py`, `models/catalog/`, `models/registry.py`

- `models/specs.py` owns `ModelSpec` and related typing.
- `models/factories.py` owns runtime object construction.
- `models/catalog/` owns model metadata shards only.
- `models/registry.py` is the public facade that composes catalog shards and exposes lookup / `make_*` helpers.

`services`

- `services/forecasting.py` owns forecast workflow orchestration.
- `services/evaluation.py` owns evaluation workflow orchestration.
- Services may depend on `contracts`, `models.registry`, metrics, splits, and existing algorithm modules.

`facades`

- `forecast.py` and `eval_forecast.py` remain import-stable public entrypoints.
- They should stay thin and delegate implementation to `services`.
- Temporary module-level compatibility shims are acceptable during migrations, but private helpers must not be part of the public export surface in `__all__`.
- New logic should not accumulate in these modules.

`cli.py`

- Owns parser construction, argument mapping, formatting, and exit behavior.
- Business rules should live in `services` or `contracts`, not in duplicated local helper functions.

## Dependency Rules

- `contracts` must not import `services`.
- `base.py` must rebuild runtime state through `models.factories`, not through `models.registry`.
- `cli.py` should not redefine long-frame validators or covariate normalization helpers.
- `services` may call `models.registry`, but `models.registry` should not depend on `services`.
- `services` must not import `foresight.forecast`, `foresight.eval_forecast`, or `foresight.cli`.
- Once dedicated `models/resolution.py` and `models/runtime.py` modules exist, `models.registry` should remain a facade and stop defining private helper families that belong in those modules.

## Where New Code Belongs

Add new code based on ownership, not convenience:

- Put shared validation or normalization in `contracts`.
- Put workflow assembly in `services`.
- Put model metadata in `models/catalog/`.
- Put runtime constructor behavior in `models/factories.py`.
- Keep `forecast.py`, `eval_forecast.py`, and `models/registry.py` as facades unless compatibility explicitly requires a small wrapper.

## Model Metadata

All registered model metadata belongs in `src/foresight/models/catalog/`.

- Group related families into shard modules such as `classical.py`, `ml.py`, `stats.py`, `torch_local.py`, `torch_global.py`, `multivariate.py`, and `foundation.py`.
- `catalog/__init__.py` is responsible for composing shards and rejecting key collisions.

## Facade vs. Implementation

Use this rule of thumb:

- If the module exists mainly because users import it directly, it is a facade.
- If the module exists mainly to hold the actual workflow or shared behavior, it is an implementation layer.

Today the main facades are:

- `foresight.forecast`
- `foresight.eval_forecast`
- `foresight.models.registry`

The main implementation layers are:

- `foresight.contracts.*`
- `foresight.services.*`
- `foresight.models.catalog.*`
- `foresight.models.factories`
- algorithm modules under `foresight.models.*`
