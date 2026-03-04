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

# Statsmodels wrappers
pip install "foresight-ts[stats]"

# Torch models (including the RNN Paper Zoo / RNN Zoo)
pip install "foresight-ts[torch]"

# Everything above
pip install "foresight-ts[all]"
```

## Quick smoke

```bash
python -m foresight --version
python -m foresight models list --prefix torch-rnnpaper
python -m foresight datasets preview catfish --nrows 10
```

## Datasets

- Small demo datasets are bundled in the wheel (e.g. `catfish`, `ice_cream_interest`).
- Larger/local datasets are not guaranteed to be bundled. Use one of:
  - `FORESIGHT_DATA_DIR=/path/to/root`
  - `python -m foresight --data-dir /path/to/root ...`

The base directory is expected to contain files at the same relative paths as in
`src/foresight/datasets/registry.py` (for example `data/store_sales.csv`).
