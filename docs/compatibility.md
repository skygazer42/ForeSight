# Compatibility Guide

## Runtime Baseline

- Python: `>=3.10`
- Core install: `numpy` + `pandas`
- Optional extras: `ml`, `xgb`, `lgbm`, `catboost`, `stats`, `torch`, `transformers`
- Default expectation: CPU-only workflows must remain usable without any heavy optional backend installed

## Installation Decision Tree

Use the smallest install that matches your workflow:

```bash
pip install foresight-ts
pip install "foresight-ts[stats]"
pip install "foresight-ts[ml]"
pip install "foresight-ts[torch]"
pip install "foresight-ts[all]"
```

- Choose core when you only need classical models, CLI dataset utilities, and basic backtesting.
- Choose `stats` for ARIMA / ETS / SARIMAX-style workflows.
- Choose `ml` for sklearn-style lag models.
- Choose `torch` for neural local/global/multivariate models.
- Choose `all` only when you need the full mixed stack.

## Stability Levels

ForeSight surfaces a model stability level in `foresight models list` and `foresight models info`:

- `stable`: default public workflows that are expected to remain import-stable and documentation-backed.
- `beta`: broader torch-based local/global model families that are supported but still evolving.
- `experimental`: frontier / wrapper / paper-zoo style models that may change faster than the core API.

## Environment Diagnostics

Use `foresight doctor` to inspect the current runtime before filing an issue or debugging an install:

```bash
foresight doctor
foresight --data-dir /path/to/root doctor
```

The report includes:

- installed package version and module path
- Python executable and version
- optional dependency status and detected extras
- packaged dataset resolution previews
- `--data-dir` and `FORESIGHT_DATA_DIR` inputs

## Dataset Resolution Order

When a workflow needs external datasets, ForeSight resolves paths in this order:

1. `--data-dir`
2. `FORESIGHT_DATA_DIR`
3. packaged demo datasets under `foresight/data/`
4. repo-root fallback when running from source

Use `foresight doctor` and `foresight datasets path <key>` to confirm which location is being used.
