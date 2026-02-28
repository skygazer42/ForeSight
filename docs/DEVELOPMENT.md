# Development Guide

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

## Quality

```bash
ruff check src tests tools
ruff format src tests tools
pytest -q
```

## CLI smoke tests

```bash
foresight --help
foresight --version

foresight datasets list
foresight datasets list --with-path
foresight datasets preview store_sales --nrows 20
foresight datasets validate
foresight datasets validate --dataset catfish
foresight datasets path catfish

foresight eval naive-last --dataset catfish --y-col Total --horizon 3 --step 3 --min-train-size 12
foresight eval naive-last --dataset catfish --y-col Total --horizon 3 --step 3 --min-train-size 12 --format md
foresight leaderboard naive --dataset catfish --y-col Total --horizon 3 --step 3 --min-train-size 12 --season-length 12
```

## Datasets location

By default, dataset loaders resolve files relative to the repository root using registry metadata.

If you want to run experiments with datasets stored somewhere else, you can override the base directory:

- **Env var:** `FORESIGHT_DATA_DIR=/path/to/root`
- **CLI flag:** `foresight --data-dir /path/to/root ...`

The base directory is expected to contain the dataset files at the same relative paths as in
`src/foresight/datasets/registry.py` (for example `data/store_sales.csv`).

