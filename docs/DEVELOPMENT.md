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

## Build / packaging smoke

```bash
# Build wheel + sdist
python -m build

# Optional: build + install wheel into a clean venv and run CLI smoke
python tools/smoke_build_install.py

# Optional: also test installing from sdist (slower; may require network)
python tools/smoke_build_install.py --sdist
```

## Generated docs

Some docs are generated from the model registries to stay in sync with code.

```bash
# (Optional) Refresh paper metadata (titles / DOI / arXiv) for the RNN Paper Zoo table.
# This hits public APIs; please be polite with `--sleep` if you run it often.
python tools/fetch_rnn_paper_metadata.py --refresh --sleep 0.02

# Re-generate the markdown docs from registries + metadata.
python tools/generate_rnn_docs.py
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

# Multi-dataset sweep + summary (dependency-free core models)
foresight leaderboard sweep --datasets catfish,ice_cream_interest --horizon 3 --step 3 --min-train-size 12 --max-windows 2 --models naive-last,mean --jobs 2 --backend process --progress --chunk-size 0 --output /tmp/sweep.json --summary-output /tmp/summary.md --summary-format md --failures-output /tmp/failures.txt
foresight leaderboard summarize --input /tmp/sweep.json --format md --min-datasets 2
```

## Datasets location

Resolution order:

1) CLI flag `--data-dir` (or function arg `data_dir=...`)
2) Env var `FORESIGHT_DATA_DIR`
3) Packaged demo datasets bundled in the wheel (e.g. `catfish`, `ice_cream_interest`)
4) Repo-root fallback when running from source (dev only)

If you want to run experiments with datasets stored somewhere else, you can override the base directory:

- **Env var:** `FORESIGHT_DATA_DIR=/path/to/root`
- **CLI flag:** `foresight --data-dir /path/to/root ...`

The base directory is expected to contain the dataset files at the same relative paths as in
`src/foresight/datasets/registry.py` (for example `data/store_sales.csv`).
