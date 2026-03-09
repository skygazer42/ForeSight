from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _normalize_covariate_arg(values: Iterable[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        s = values.strip()
        return tuple([p.strip() for p in s.split(",") if p.strip()]) if s else ()
    return tuple([str(v).strip() for v in values if str(v).strip()])


def _merge_unique_columns(*groups: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for col in group:
            if col not in seen:
                seen.add(col)
                out.append(col)
    return tuple(out)


def resolve_covariate_roles(
    *,
    x_cols: Iterable[str] | str | None = (),
    historic_x_cols: Iterable[str] | str | None = (),
    future_x_cols: Iterable[str] | str | None = (),
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """
    Normalize covariate-role arguments.

    `x_cols` is kept as a compatibility alias and is merged into `future_x_cols`.
    """
    historic = _normalize_covariate_arg(historic_x_cols)
    future = _merge_unique_columns(
        _normalize_covariate_arg(future_x_cols),
        _normalize_covariate_arg(x_cols),
    )
    all_x = _merge_unique_columns(historic, future)
    return historic, future, all_x


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


def to_long(
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
