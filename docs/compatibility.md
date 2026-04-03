# Compatibility Guide

## Runtime Baseline

- Python: `>=3.10`
- Core install: `numpy` + `pandas`
- Optional extras: `ml`, `xgb`, `lgbm`, `catboost`, `stats`, `torch`, `transformers`
- Default expectation: CPU-only workflows must remain usable without any heavy optional backend installed

## Supported Public Surface

ForeSight exposes many internal modules and experimental model families, but the
compatibility promise is anchored to a smaller stable public surface:

- the root `foresight` Python API exported through `foresight.__all__`
- model registry metadata and discovery fields surfaced by `foresight models list/info`
- the documented core CLI workflows, especially `doctor`, `models`, `forecast`,
  `eval`, `leaderboard`, and `artifact`
- the persisted forecaster artifact schema contract documented in `docs/artifacts.md`

Everything else should be treated as implementation detail unless it is
explicitly documented as a stable entry point. Beta and experimental model
families remain supported for use, but not with the same compatibility promise
as the stable public surface.

## CI-Backed Support Matrix

The project should only claim support where the repository actually runs checks:

| surface | matrix | expectation |
| --- | --- | --- |
| Core package + public contract suite | Python `3.10`, `3.11` | Supported and checked in CI |
| Release build / publish path | Python `3.10` | Supported and checked before publishing |
| Optional `stats` extra smoke | Python `3.10` | Checked in CI |
| Optional `ml` extra smoke | Python `3.10` | Checked in CI |
| Torch / transformers / frontier wrappers | best-effort | Availability is communicated through stability levels, not the core compatibility guarantee |

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

Stable docs and examples should prefer the `stable` surface first, then opt into
`beta` or `experimental` intentionally.

## Artifact Compatibility Contract

Serialized forecaster artifacts are part of the supported platform surface:

- schema version `1` is the current artifact payload contract
- legacy payloads without an explicit schema version are still loaded as version `0`
- releases that change artifact payload structure must update
  `src/foresight/serialization.py`, keep serialization tests green, and document
  whether re-saving artifacts is required after upgrade
- forward compatibility with unknown future schema versions is intentionally not promised

## Environment Diagnostics

Use `foresight doctor` to inspect the current runtime before filing an issue or debugging an install:

```bash
foresight doctor
foresight --data-dir /path/to/root doctor
foresight doctor --format text
foresight --data-dir /path/to/root doctor --format text --strict
foresight doctor --require-extra torch --strict
```

The report includes:

- installed package version and module path
- Python executable and version
- optional dependency status and detected extras
- packaged dataset resolution previews
- `--data-dir` and `FORESIGHT_DATA_DIR` inputs
- text output for human triage and JSON for machine-readable automation
- strict mode that returns exit code `1` when warnings are present
- required-extra checks that promote missing requested extras to errors

## Dataset Resolution Order

When a workflow needs external datasets, ForeSight resolves paths in this order:

1. `--data-dir`
2. `FORESIGHT_DATA_DIR`
3. packaged demo datasets under `foresight/data/`
4. repo-root fallback when running from source

Use `foresight doctor` and `foresight datasets path <key>` to confirm which location is being used.
