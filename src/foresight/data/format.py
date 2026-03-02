from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _build_unique_id(df: pd.DataFrame, *, id_cols: tuple[str, ...]) -> pd.Series:
    if not id_cols:
        return pd.Series(["series=0"] * len(df), index=df.index, dtype="string")

    parts: list[pd.Series] = []
    for col in id_cols:
        if col not in df.columns:
            raise KeyError(f"id col not found: {col!r}")
        parts.append(col + "=" + df[col].astype("string"))
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def to_long(
    df: pd.DataFrame,
    *,
    time_col: str,
    y_col: str,
    id_cols: Iterable[str] = (),
    x_cols: Iterable[str] = (),
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Convert an arbitrary DataFrame into a canonical long format:

      unique_id | ds | y

    Commonly used by forecasting toolkits (Prophet/Nixtla-style).
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col not found: {time_col!r}")
    if y_col not in df.columns:
        raise KeyError(f"y_col not found: {y_col!r}")

    id_cols_tup = tuple(id_cols)
    x_cols_tup = tuple(x_cols)
    out = pd.DataFrame(index=df.index)
    out["unique_id"] = _build_unique_id(df, id_cols=id_cols_tup)
    out["ds"] = df[time_col]
    out["y"] = df[y_col]

    for col in x_cols_tup:
        if col in {"unique_id", "ds", "y"}:
            raise ValueError(f"x_cols cannot include reserved column name: {col!r}")
        if col not in df.columns:
            raise KeyError(f"x col not found: {col!r}")
        out[col] = df[col]

    if dropna:
        out = out.dropna(subset=["ds", "y", *x_cols_tup])

    return out.reset_index(drop=True)


def validate_long_df(
    df: pd.DataFrame,
    *,
    require_sorted: bool = True,
    require_unique_ds: bool = True,
) -> None:
    """
    Validate a canonical long-format DataFrame (unique_id, ds, y).
    """
    required = {"unique_id", "ds", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Long DataFrame is empty")

    if df["unique_id"].isna().any():
        raise ValueError("unique_id contains NA values")
    if df["ds"].isna().any():
        raise ValueError("ds contains NA values")

    for uid, g in df.groupby("unique_id", sort=False):
        if require_sorted and not g["ds"].is_monotonic_increasing:
            raise ValueError(f"ds is not sorted for unique_id={uid!r}")
        if require_unique_ds and g["ds"].duplicated().any():
            raise ValueError(f"ds contains duplicates for unique_id={uid!r}")
