from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..contracts.covariates import resolve_covariate_roles as _contracts_resolve_covariate_roles


def resolve_covariate_roles(
    *,
    x_cols: Iterable[str] | str | None = (),
    historic_x_cols: Iterable[str] | str | None = (),
    future_x_cols: Iterable[str] | str | None = (),
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    spec = _contracts_resolve_covariate_roles(
        x_cols=x_cols,
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
    )
    return spec.historic_x_cols, spec.future_x_cols, spec.all_x_cols


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


def build_hierarchy_spec(
    df: pd.DataFrame,
    *,
    id_cols: Iterable[str],
    root: str = "total",
) -> dict[str, tuple[str, ...]]:
    """
    Build a simple parent->children hierarchy spec from ordered ID columns.

    Example:
      id_cols=("region", "store")
      total -> region=north, region=south
      region=north -> region=north|store=a, region=north|store=b
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    id_cols_tup = tuple(id_cols)
    if not id_cols_tup:
        raise ValueError("id_cols must be non-empty")

    for col in id_cols_tup:
        if col not in df.columns:
            raise KeyError(f"id col not found: {col!r}")

    out: dict[str, tuple[str, ...]] = {}
    root_s = str(root).strip()
    if not root_s:
        raise ValueError("root must be non-empty")

    first_level = sorted(pd.unique(_build_unique_id(df, id_cols=(id_cols_tup[0],))).tolist())
    out[root_s] = tuple(str(v) for v in first_level)

    for depth in range(1, len(id_cols_tup)):
        parent_ids = _build_unique_id(df, id_cols=id_cols_tup[:depth])
        child_ids = _build_unique_id(df, id_cols=id_cols_tup[: depth + 1])
        pairs = pd.DataFrame({"parent": parent_ids, "child": child_ids}).drop_duplicates()
        for parent, g in pairs.groupby("parent", sort=True):
            children = tuple(sorted(str(v) for v in g["child"].tolist()))
            out[str(parent)] = children

    return out


def to_long(  # NOSONAR - public API intentionally exposes explicit conversion knobs
    df: pd.DataFrame,
    *,
    time_col: str,
    y_col: str,
    id_cols: Iterable[str] = (),
    x_cols: Iterable[str] = (),
    historic_x_cols: Iterable[str] = (),
    future_x_cols: Iterable[str] = (),
    dropna: bool = True,
    prepare: bool = False,
    freq: str | None = None,
    strict_freq: bool = False,
    y_missing: str = "error",
    x_missing: str = "error",
    historic_x_missing: str | None = None,
    future_x_missing: str | None = None,
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
    historic_x_cols_tup, future_x_cols_tup, all_x_cols_tup = resolve_covariate_roles(
        x_cols=x_cols,
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
    )
    out = pd.DataFrame(index=df.index)
    out["unique_id"] = _build_unique_id(df, id_cols=id_cols_tup)
    out["ds"] = df[time_col]
    out["y"] = df[y_col]

    for col in all_x_cols_tup:
        if col in {"unique_id", "ds", "y"}:
            raise ValueError(f"x_cols cannot include reserved column name: {col!r}")
        if col not in df.columns:
            raise KeyError(f"x col not found: {col!r}")
        out[col] = df[col]

    out.attrs["historic_x_cols"] = historic_x_cols_tup
    out.attrs["future_x_cols"] = future_x_cols_tup
    if dropna:
        out = out.dropna(subset=["ds", "y", *all_x_cols_tup])

    if prepare:
        from .prep import prepare_long_df

        out = prepare_long_df(
            out,
            freq=freq,
            strict_freq=bool(strict_freq),
            y_missing=str(y_missing),
            x_missing=str(x_missing),
            historic_x_cols=historic_x_cols_tup,
            future_x_cols=future_x_cols_tup,
            historic_x_missing=historic_x_missing,
            future_x_missing=future_x_missing,
        )

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


def long_to_wide(
    long_df: Any,
    *,
    id_col: str = "unique_id",
    ds_col: str = "ds",
    value_col: str = "y",
    freq: str | None = None,
    strict_freq: bool = False,
    missing: str = "error",
    sort_ids: bool = True,
) -> pd.DataFrame:
    """
    Pivot a canonical long-format DataFrame into wide format.

    Input long format:
      - one row per (id_col, ds_col)
      - value_col holds the observed target

    Output wide format:
      - one row per timestamp (ds_col)
      - one column per series id (id_col)

    Optionally regularizes the timestamp axis via `freq` and fills / rejects missing
    target values via `missing`.
    """
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df must be a pandas DataFrame")

    df = long_df.copy()
    if df.empty:
        raise ValueError("long_df is empty")

    id_col_s = str(id_col).strip()
    ds_col_s = str(ds_col).strip()
    value_col_s = str(value_col).strip()
    if not id_col_s:
        raise ValueError("id_col must be non-empty")
    if not ds_col_s:
        raise ValueError("ds_col must be non-empty")
    if not value_col_s:
        raise ValueError("value_col must be non-empty")

    required = {id_col_s, ds_col_s, value_col_s}
    missing_cols = required.difference(df.columns)
    if missing_cols:
        raise KeyError(f"long_df missing required columns: {sorted(missing_cols)}")

    df = df.loc[:, [id_col_s, ds_col_s, value_col_s]].copy()
    if df[[id_col_s, ds_col_s]].duplicated().any():
        raise ValueError("long_df contains duplicate id/ds pairs")

    # Normalize ids to strings to keep CLI and downstream behavior deterministic.
    df[id_col_s] = df[id_col_s].astype("string")

    wide = (
        df.pivot(index=ds_col_s, columns=id_col_s, values=value_col_s)
        .sort_index(axis=0)
        .reset_index()
    )

    # Stable column ordering: ds first, then id columns.
    id_cols_out = [c for c in wide.columns if c != ds_col_s]
    if sort_ids:
        id_cols_out = sorted(id_cols_out, key=lambda x: str(x).lower())
    wide = wide.loc[:, [ds_col_s, *id_cols_out]]

    from .prep import prepare_wide_df

    return prepare_wide_df(
        wide,
        ds_col=ds_col_s,
        freq=freq,
        strict_freq=bool(strict_freq),
        missing=str(missing),
        target_cols=tuple(str(c) for c in id_cols_out),
    )
