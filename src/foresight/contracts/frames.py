from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def require_long_df(
    long_df: Any,
    *,
    require_non_empty: bool = False,
) -> pd.DataFrame:
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df must be a pandas DataFrame")

    required = {"unique_id", "ds", "y"}
    missing = required.difference(long_df.columns)
    if missing:
        raise KeyError(f"long_df missing required columns: {sorted(missing)}")

    if require_non_empty and long_df.empty:
        raise ValueError("long_df is empty")

    if long_df.loc[:, ["unique_id", "ds"]].duplicated().any():
        raise ValueError("long_df contains duplicate unique_id/ds rows")

    return long_df


def require_future_df(
    future_df: Any,
    *,
    require_non_empty: bool = True,
) -> pd.DataFrame:
    if not isinstance(future_df, pd.DataFrame):
        raise TypeError("future_df must be a pandas DataFrame")

    required = {"unique_id", "ds"}
    missing = required.difference(future_df.columns)
    if missing:
        raise KeyError(f"future_df missing required columns: {sorted(missing)}")

    if require_non_empty and future_df.empty:
        raise ValueError("future_df is empty")

    out = future_df.copy()
    if "y" not in out.columns:
        out["y"] = np.nan
    elif out["y"].notna().any():
        raise ValueError("future_df must not contain observed y values")

    return out


def merge_history_and_future_df(long_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    overlap = (
        long_df.loc[:, ["unique_id", "ds"]]
        .merge(future_df.loc[:, ["unique_id", "ds"]], on=["unique_id", "ds"], how="inner")
        .drop_duplicates()
    )
    if not overlap.empty:
        raise ValueError("future_df overlaps with long_df on unique_id/ds")

    cols = list(long_df.columns)
    for col in future_df.columns:
        if col not in cols:
            cols.append(col)

    left = long_df.copy()
    right = future_df.copy()
    for col in cols:
        if col not in left.columns:
            left[col] = np.nan
        if col not in right.columns:
            right[col] = np.nan

    merged = pd.concat(
        [left.loc[:, cols], right.loc[:, cols]],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    return sort_long_df(merged)


def require_observed_history_only(df: pd.DataFrame) -> pd.DataFrame:
    if df["y"].isna().any():
        raise ValueError("long_df contains missing y values; provide observed history only")
    return df


def sort_long_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
