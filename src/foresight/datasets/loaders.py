from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .registry import DatasetSpec, get_dataset_spec


def _repo_root(*, data_dir: str | Path | None = None) -> Path:
    # src/foresight/datasets/loaders.py -> repo root is 3 levels up from `src/`
    if isinstance(data_dir, str) and not data_dir.strip():
        data_dir = None
    if data_dir is not None:
        return Path(data_dir).expanduser()

    env_dir = os.environ.get("FORESIGHT_DATA_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return Path(__file__).resolve().parents[3]


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}. Are you running from the repo root?")
    return path


def _load_csv_spec(*, spec: DatasetSpec, nrows: int | None, data_dir: str | Path | None) -> pd.DataFrame:
    path = _require_file(_repo_root(data_dir=data_dir) / spec.rel_path)
    read_kwargs = {"nrows": nrows}
    if spec.parse_dates:
        read_kwargs["parse_dates"] = list(spec.parse_dates)
    return pd.read_csv(path, **read_kwargs)

def load_store_sales(*, nrows: int | None = None, data_dir: str | Path | None = None) -> pd.DataFrame:
    return load_dataset("store_sales", nrows=nrows, data_dir=data_dir)

def load_promotion_data(*, nrows: int | None = None, data_dir: str | Path | None = None) -> pd.DataFrame:
    return load_dataset("promotion_data", nrows=nrows, data_dir=data_dir)


def load_cashflow_data(*, nrows: int | None = None, data_dir: str | Path | None = None) -> pd.DataFrame:
    return load_dataset("cashflow_data", nrows=nrows, data_dir=data_dir)


def load_catfish(*, nrows: int | None = None, data_dir: str | Path | None = None) -> pd.DataFrame:
    return load_dataset("catfish", nrows=nrows, data_dir=data_dir)


def load_ice_cream_interest(*, nrows: int | None = None, data_dir: str | Path | None = None) -> pd.DataFrame:
    return load_dataset("ice_cream_interest", nrows=nrows, data_dir=data_dir)


def load_dataset(
    key: str,
    *,
    nrows: int | None = None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Convenience loader by dataset key for CLI and quick experiments.

    Note: keyword support is intentionally minimal and dataset-specific.
    """
    spec = get_dataset_spec(key)
    return _load_csv_spec(spec=spec, nrows=nrows, data_dir=data_dir)
