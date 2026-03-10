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

Most runtime behavior eventually flows through the registry:

1. public API or CLI handler requests a model key
2. `src/foresight/models/registry.py`
3. `get_model_spec()` resolves metadata such as:
   - description
   - interface type (`local`, `global`, `multivariate`)
   - optional dependency requirements
   - parameter help and capability flags
4. `make_forecaster()` / `make_global_forecaster()` / `make_multivariate_forecaster()` return a callable
5. object-style calls use `make_forecaster_object()` or `make_global_forecaster_object()` in combination with `src/foresight/base.py`

If you add a new model family, the registry is the first file to inspect.

## 4. Forecast and evaluation path

There are two main runtime paths after a model is resolved.

### Forecast path

1. CLI `forecast ...` or Python `forecast_model(...)`
2. `src/foresight/forecast.py`
3. input validation and long-format preparation
4. registry lookup and callable/object creation
5. prediction frame assembly, optional interval generation, optional artifact save/load

### Evaluation path

1. CLI `eval ...` / `cv ...` / `leaderboard ...` or Python `eval_model(...)`
2. `src/foresight/eval_forecast.py`
3. dataset loading or DataFrame validation
4. rolling split generation via `src/foresight/splits.py`
5. walk-forward execution via `src/foresight/backtesting.py`
6. metric aggregation via `src/foresight/metrics.py`

For debugging metric regressions or shape mismatches, start with:

- `src/foresight/eval_forecast.py`
- `src/foresight/backtesting.py`
- `src/foresight/metrics.py`

## 5. Data and dataset path

ForeSight has two related but separate data layers.

### Built-in dataset registry

- `src/foresight/datasets/registry.py` defines dataset metadata and path resolution
- `src/foresight/datasets/loaders.py` loads those datasets

### Generic formatting helpers

- `src/foresight/data/format.py` converts raw frames into the package's long format
- `src/foresight/data/prep.py` and feature helpers support preprocessing workflows

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
8. `src/foresight/eval_forecast.py`
9. `src/foresight/backtesting.py`
10. `src/foresight/datasets/registry.py`

That sequence gets you from the public entrypoints to the model registry and then into the forecast/evaluation core without reading the entire model zoo first.
