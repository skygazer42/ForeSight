# Source Entrypoints

This note maps the fastest way to read the ForeSight codebase from the outside in.

## 1. CLI startup path

The command-line path starts here:

1. `python -m foresight`
2. `src/foresight/__main__.py`
3. `src/foresight/cli.py:main()`
4. `build_parser()` creates the root parser and subcommands
5. `main()` resolves either:
   - a root shortcut such as `--list` / `--list-datasets`
   - or a subcommand handler stored in `args._handler`
6. the selected handler imports only the modules it needs and executes the requested operation

When you want to add or debug CLI behavior, start with:

- `src/foresight/__main__.py`
- `src/foresight/cli.py`

The CLI is intentionally lazy. Most heavy imports happen inside handlers, not at module import time.

## 2. Python import path

The package import path starts here:

1. `import foresight`
2. `src/foresight/__init__.py`
3. `__all__` defines the stable public surface
4. `__getattr__()` lazily imports submodules on first access

This means:

- `import foresight` stays lightweight
- public functions such as `eval_model`, `forecast_model`, and `make_forecaster` are re-exported from one place
- registry and model modules are not imported until needed

If you are tracing a Python API call, begin at `src/foresight/__init__.py` and then follow the `__getattr__()` branch for the symbol you care about.

## 3. Model resolution path

Most runtime behavior eventually flows through the model facade, but the implementation is now split across multiple layers:

1. public API or CLI handler requests a model key
2. `src/foresight/models/registry.py`
3. `registry.py` delegates metadata assembly to `src/foresight/models/catalog/`
4. `get_model_spec()` resolves metadata such as:
   - description
   - interface type (`local`, `global`, `multivariate`)
   - optional dependency requirements
   - parameter help and capability flags
5. runtime construction flows through `src/foresight/models/factories.py`
6. object-style calls still combine factory output with `src/foresight/base.py`

Read these files in this order when working on model registration:

- `src/foresight/models/registry.py`
- `src/foresight/models/catalog/__init__.py`
- the relevant shard under `src/foresight/models/catalog/`
- `src/foresight/models/factories.py`

If you add a new model family, start with the appropriate catalog shard rather than stuffing more declarations back into `registry.py`.

## 4. Forecast and evaluation path

There are two main runtime paths after a model is resolved.

The public modules stay stable, but they are now thin facades over `src/foresight/services/`.

### Forecast path

1. CLI `forecast ...` or Python `forecast_model(...)`
2. `src/foresight/forecast.py`
3. `forecast.py` forwards to `src/foresight/services/forecasting.py`
4. service-level validation reuses `src/foresight/contracts/`
5. registry lookup and callable/object creation happen through the model facade and factories
6. prediction frame assembly, optional interval generation, optional artifact save/load happen inside the forecasting service

### Evaluation path

1. CLI `eval ...` / `cv ...` / `leaderboard ...` or Python `eval_model(...)`
2. `src/foresight/eval_forecast.py`
3. `eval_forecast.py` forwards to `src/foresight/services/evaluation.py`
4. dataset loading or DataFrame validation happens in the evaluation service, with shared rules from `src/foresight/contracts/`
5. rolling split generation still uses `src/foresight/splits.py`
6. walk-forward execution still uses `src/foresight/backtesting.py`
7. metric aggregation still uses `src/foresight/metrics.py`

For debugging metric regressions or shape mismatches, start with:

- `src/foresight/eval_forecast.py`
- `src/foresight/services/evaluation.py`
- `src/foresight/backtesting.py`
- `src/foresight/metrics.py`

## 5. Data and dataset path

ForeSight has two related but separate data layers.

### Built-in dataset registry

- `src/foresight/datasets/registry.py` defines dataset metadata and path resolution
- `src/foresight/datasets/loaders.py` loads those datasets

### Generic formatting helpers

- `src/foresight/data_processing/format.py` converts raw frames into the package's long format
- `src/foresight/data_processing/prep.py` and workflow helpers support preprocessing workflows
- `src/foresight/data/format.py`, `src/foresight/data/prep.py`, and `src/foresight/data/workflows.py`
  remain as compatibility shims for legacy imports.

If a dataset command fails, check the dataset registry first. If a custom CSV forecast/eval path fails, inspect the format helpers.

## 6. Practical reading order

For most contributors, the most efficient top-down reading order is:

1. `README.md`
2. `src/foresight/__main__.py`
3. `src/foresight/cli.py`
4. `src/foresight/__init__.py`
5. `src/foresight/models/registry.py`
6. `src/foresight/base.py`
7. `src/foresight/forecast.py`
8. `src/foresight/services/forecasting.py`
9. `src/foresight/eval_forecast.py`
10. `src/foresight/services/evaluation.py`
11. `src/foresight/backtesting.py`
12. `src/foresight/datasets/registry.py`

That sequence gets you from the public entrypoints to the facades, then into the service and model layers, without reading the entire model zoo first.
