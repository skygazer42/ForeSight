# Install

This project is distributed on PyPI as **`foresight-ts`** and imported as **`foresight`**.

## Basic

```bash
pip install foresight-ts
```

## TestPyPI (optional)

```bash
# Use TestPyPI for pre-release smoke (may be missing some wheels).
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple foresight-ts
```

## Optional extras

```bash
# ML models (scikit-learn)
pip install "foresight-ts[ml]"

# XGBoost models
pip install "foresight-ts[xgb]"

# Statsmodels wrappers
pip install "foresight-ts[stats]"

# Torch models (including the RNN Paper Zoo / RNN Zoo)
pip install "foresight-ts[torch]"

# Everything above
pip install "foresight-ts[all]"
```

## Quick smoke

```bash
python -m foresight --help
python -m foresight --version
python -m foresight doctor
python -m foresight doctor --format text
python -m foresight --data-dir /path/to/root doctor --format text --strict
python -m foresight cv --help
python -m foresight forecast --help
python -m foresight detect --help
python -m foresight tuning --help
python -m foresight cv csv --help
python -m foresight forecast csv --help
python -m foresight models list --prefix torch-rnnpaper
python -m foresight datasets preview catfish --nrows 10
python -m foresight eval run --model naive-last --dataset catfish --y-col Total --horizon 3 --step 3 --min-train-size 12
python -m foresight detect run --dataset catfish --y-col Total --model naive-last --score-method forecast-residual --min-train-size 24
python -m foresight tuning run --model moving-average --dataset catfish --y-col Total --horizon 1 --step 1 --min-train-size 24 --max-windows 4 --grid-param window=1,3
```

Runtime logs for long-running commands are written to `stderr` by default. Use
`--no-progress`, `--log-style plain`, or `--log-file /tmp/run-log.jsonl` to
control the CLI logging behavior without affecting `stdout`.

Use `doctor --format text` for a human-readable environment summary and
`doctor --strict` when you want warnings to return exit code `1` in automation.

## High-level Python helpers

After install, the package root exposes the common forecasting workflow helpers directly:

```python
from foresight import (
    bootstrap_intervals,
    detect_anomalies,
    eval_model,
    forecast_model,
    load_forecaster,
    make_forecaster,
    make_forecaster_object,
    prepare_long_df,
    save_forecaster,
    tune_model,
)
```

## Datasets

- Small demo datasets are bundled in the wheel (e.g. `catfish`, `ice_cream_interest`).
- Larger/local datasets are not guaranteed to be bundled. Use one of:
  - `FORESIGHT_DATA_DIR=/path/to/root`
  - `python -m foresight --data-dir /path/to/root ...`

The base directory is expected to contain files at the same relative paths as in
`src/foresight/datasets/registry.py` (for example `data/store_sales.csv`).

## Picking A Model

After installation, use the model selection guide plus capability filters
instead of scanning the full registry by hand:

- [How to choose a model](model-selection.md)
- `python -m foresight models list --stability stable --format json`
- `python -m foresight models list --capability supports_x_cols=true --format json`
