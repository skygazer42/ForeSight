from __future__ import annotations

from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    # src/foresight/datasets/loaders.py -> repo root is 3 levels up from `src/`
    return Path(__file__).resolve().parents[3]


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}. Are you running from the repo root?")
    return path


def load_store_sales(*, nrows: int | None = None) -> pd.DataFrame:
    path = _require_file(_repo_root() / "data" / "store_sales.csv")
    return pd.read_csv(path, parse_dates=["week"], nrows=nrows)


def load_promotion_data(*, nrows: int | None = None) -> pd.DataFrame:
    path = _require_file(_repo_root() / "data" / "promotion_data.csv")
    return pd.read_csv(path, parse_dates=["week"], nrows=nrows)


def load_cashflow_data(*, nrows: int | None = None) -> pd.DataFrame:
    path = _require_file(_repo_root() / "data" / "cashflow_data.csv")
    # Keep parsing conservative; some CSVs might not have a standard date column.
    return pd.read_csv(path, nrows=nrows)


def load_catfish(*, nrows: int | None = None) -> pd.DataFrame:
    path = _require_file(_repo_root() / "statistics time series" / "catfish.csv")
    return pd.read_csv(path, parse_dates=["Date"], nrows=nrows)


def load_ice_cream_interest(*, nrows: int | None = None) -> pd.DataFrame:
    path = _require_file(_repo_root() / "statistics time series" / "ice_cream_interest.csv")
    return pd.read_csv(path, parse_dates=["month"], nrows=nrows)


def load_dataset(key: str, **kwargs) -> pd.DataFrame:
    """
    Convenience loader by dataset key for CLI and quick experiments.

    Note: keyword support is intentionally minimal and dataset-specific.
    """
    if key == "store_sales":
        return load_store_sales(nrows=kwargs.get("nrows"))
    if key == "promotion_data":
        return load_promotion_data(nrows=kwargs.get("nrows"))
    if key == "cashflow_data":
        return load_cashflow_data(nrows=kwargs.get("nrows"))
    if key == "catfish":
        return load_catfish(nrows=kwargs.get("nrows"))
    if key == "ice_cream_interest":
        return load_ice_cream_interest(nrows=kwargs.get("nrows"))
    raise KeyError(f"Unknown dataset key: {key!r}")
