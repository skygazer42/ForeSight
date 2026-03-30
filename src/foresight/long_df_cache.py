from __future__ import annotations

import weakref
from typing import Any

import numpy as np
import pandas as pd

from .contracts.frames import coerce_sorted_long_df
from .splits import rolling_origin_split_sequence

_LONG_DF_CACHE_REGISTRY: dict[int, dict[str, Any]] = {}
_LONG_DF_CACHE_REFS: dict[int, weakref.ReferenceType[pd.DataFrame]] = {}


def _register_cache(df: pd.DataFrame, cache: dict[str, Any]) -> dict[str, Any]:
    key = id(df)

    def _cleanup(
        ref: weakref.ReferenceType[pd.DataFrame],
        *,
        cache_key: int = key,
        refs: dict[int, weakref.ReferenceType[pd.DataFrame]] = _LONG_DF_CACHE_REFS,
        registry: dict[int, dict[str, Any]] = _LONG_DF_CACHE_REGISTRY,
    ) -> None:
        current = refs.get(cache_key)
        if current is ref:
            refs.pop(cache_key, None)
            registry.pop(cache_key, None)

    _LONG_DF_CACHE_REGISTRY[key] = cache
    _LONG_DF_CACHE_REFS[key] = weakref.ref(df, _cleanup)
    return cache


def long_df_cache(df: pd.DataFrame) -> dict[str, Any]:
    key = id(df)
    ref = _LONG_DF_CACHE_REFS.get(key)
    cache = _LONG_DF_CACHE_REGISTRY.get(key)
    if ref is not None and ref() is df and isinstance(cache, dict):
        return cache
    return _register_cache(df, {})


def sorted_long_df(long_df: pd.DataFrame, *, reset_index: bool) -> pd.DataFrame:
    cache = long_df_cache(long_df)
    key = ("sorted_long_df", bool(reset_index))
    cached = cache.get(key)
    if isinstance(cached, pd.DataFrame):
        return cached

    out = coerce_sorted_long_df(long_df, reset_index=bool(reset_index))
    if out is not long_df:
        _register_cache(out, cache)
    cache[key] = out
    return out


def cached_series_slices(df: pd.DataFrame) -> tuple[tuple[str, int, int], ...]:
    cache = long_df_cache(df)
    cached = cache.get("series_slices")
    if isinstance(cached, tuple):
        return cached
    if df.empty:
        return ()

    uid_arr = df["unique_id"].to_numpy(copy=False)
    boundaries = np.flatnonzero(uid_arr[1:] != uid_arr[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=int), boundaries.astype(int, copy=False)))
    stops = np.concatenate((boundaries.astype(int, copy=False), np.array([len(df)], dtype=int)))
    out = tuple(
        (str(uid_arr[int(start)]), int(start), int(stop))
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True)
    )
    cache["series_slices"] = out
    return out


def cached_ds_array(df: pd.DataFrame) -> np.ndarray:
    cache = long_df_cache(df)
    cached = cache.get("ds_array")
    if isinstance(cached, np.ndarray):
        return cached

    out = df["ds"].to_numpy(copy=False)
    cache["ds_array"] = out
    return out


def cached_y_array(df: pd.DataFrame) -> np.ndarray:
    cache = long_df_cache(df)
    cached = cache.get("y_array")
    if isinstance(cached, np.ndarray):
        return cached

    out = df["y"].to_numpy(dtype=float, copy=False)
    cache["y_array"] = out
    return out


def cached_y_lookup(df: pd.DataFrame) -> pd.Series:
    cache = long_df_cache(df)
    cached = cache.get("y_lookup")
    if isinstance(cached, pd.Series):
        return cached

    out = df.set_index(["unique_id", "ds"])["y"]
    cache["y_lookup"] = out
    return out


def cached_x_matrix(df: pd.DataFrame, *, x_cols: tuple[str, ...]) -> np.ndarray:
    missing_x_cols = [col for col in x_cols if col not in df.columns]
    if missing_x_cols:
        raise KeyError(f"long_df missing required x_cols: {missing_x_cols}")

    cache = long_df_cache(df)
    x_matrices = cache.setdefault("x_matrices", {})
    key = tuple(x_cols)
    cached = x_matrices.get(key)
    if isinstance(cached, np.ndarray):
        return cached

    out = df.loc[:, list(key)].to_numpy(dtype=float, copy=False)
    x_matrices[key] = out
    return out


def cached_split_sequence(
    df: pd.DataFrame,
    *,
    namespace: str,
    n_obs: int,
    horizon: int,
    step_size: int,
    min_train_size: int,
    max_train_size: int | None,
    limit: int | None,
    keep: str,
    limit_error: str,
) -> tuple[Any, ...]:
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")

    cache = long_df_cache(df)
    split_cache = cache.setdefault("split_sequences", {})
    key = (
        str(namespace),
        int(n_obs),
        int(horizon),
        int(step_size),
        int(min_train_size),
        None if max_train_size is None else int(max_train_size),
        None if limit is None else int(limit),
        str(keep),
    )
    cached = split_cache.get(key)
    if isinstance(cached, tuple):
        return cached

    out = rolling_origin_split_sequence(
        int(n_obs),
        horizon=int(horizon),
        step_size=int(step_size),
        min_train_size=int(min_train_size),
        max_train_size=max_train_size,
        limit=limit,
        keep=str(keep),
        limit_error=str(limit_error),
    )
    split_cache[key] = out
    return out


__all__ = [
    "cached_ds_array",
    "cached_series_slices",
    "cached_split_sequence",
    "cached_x_matrix",
    "cached_y_array",
    "cached_y_lookup",
    "long_df_cache",
    "sorted_long_df",
]
