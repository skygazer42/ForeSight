from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .contracts.params import normalize_covariate_roles as _normalize_covariate_roles
from .contracts.params import normalize_static_cols as _normalize_static_cols
from .data.format import to_long
from .datasets.loaders import load_dataset
from .datasets.registry import get_dataset_spec

_DATASET_LONG_DF_CACHE: dict[tuple[Any, ...], pd.DataFrame] = {}
_DATASET_RAW_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_DATASET_SPEC_CACHE: dict[str, Any] = {}


def _normalize_dataset_long_df_request(
    *,
    dataset: str,
    y_col: str | None,
    data_dir: str | Path | None,
    model_params: dict[str, Any] | None,
) -> dict[str, Any]:
    dataset_key = str(dataset)
    data_dir_s = "" if data_dir is None else str(data_dir).strip()
    data_dir_arg = data_dir_s or None
    spec = _get_cached_dataset_spec(dataset_key)
    y_col_final = (
        str(y_col).strip() if (y_col is not None and str(y_col).strip()) else str(spec.default_y)
    )
    historic_x_cols, future_x_cols = _normalize_covariate_roles(model_params)
    static_cols = _normalize_static_cols(model_params)
    return {
        "dataset_key": dataset_key,
        "data_dir_s": data_dir_s,
        "data_dir_arg": data_dir_arg,
        "spec": spec,
        "y_col_final": y_col_final,
        "historic_x_cols": historic_x_cols,
        "future_x_cols": future_x_cols,
        "static_cols": static_cols,
    }


def _get_dataset_long_df_cache_lock() -> Any:
    import threading

    lock = getattr(get_or_build_dataset_long_df, "_cache_lock", None)
    if lock is None:
        lock = threading.Lock()
        setattr(get_or_build_dataset_long_df, "_cache_lock", lock)
    return lock


def _get_cached_dataset_spec(dataset_key: str) -> Any:
    lock = _get_dataset_long_df_cache_lock()
    with lock:
        spec = _DATASET_SPEC_CACHE.get(str(dataset_key))
    if spec is not None:
        return spec

    spec = get_dataset_spec(str(dataset_key))
    with lock:
        _DATASET_SPEC_CACHE[str(dataset_key)] = spec
        return _DATASET_SPEC_CACHE[str(dataset_key)]


def get_or_build_dataset_frame(
    *,
    dataset: str,
    data_dir: str | Path | None,
) -> dict[str, Any]:
    import time

    dataset_key = str(dataset)
    data_dir_s = "" if data_dir is None else str(data_dir).strip()
    data_dir_arg = data_dir_s or None
    raw_key = (dataset_key, data_dir_s)
    lock = _get_dataset_long_df_cache_lock()

    with lock:
        raw_bundle = _DATASET_RAW_CACHE.get(raw_key)

    load_seconds = 0.0
    if raw_bundle is None:
        started = time.perf_counter()
        raw_bundle = {
            "spec": _get_cached_dataset_spec(dataset_key),
            "df": load_dataset(dataset_key, data_dir=data_dir_arg),
        }
        load_seconds = float(time.perf_counter() - started)
        with lock:
            _DATASET_RAW_CACHE[raw_key] = raw_bundle
            raw_bundle = _DATASET_RAW_CACHE[raw_key]

    return {
        "spec": raw_bundle["spec"],
        "df": raw_bundle["df"],
        "load_seconds": round(load_seconds, 6),
    }


def get_or_build_dataset_long_df(
    *,
    dataset: str,
    y_col: str | None,
    data_dir: str | Path | None,
    model_params: dict[str, Any] | None,
) -> dict[str, Any]:
    import time

    request = _normalize_dataset_long_df_request(
        dataset=dataset,
        y_col=y_col,
        data_dir=data_dir,
        model_params=model_params,
    )
    dataset_key = str(request["dataset_key"])
    data_dir_s = str(request["data_dir_s"])
    data_dir_arg = request["data_dir_arg"]
    spec = request["spec"]
    y_col_final = str(request["y_col_final"])
    historic_x_cols = tuple(request["historic_x_cols"])
    future_x_cols = tuple(request["future_x_cols"])
    static_cols = tuple(request["static_cols"])

    prepared_key = (
        dataset_key,
        y_col_final,
        data_dir_s,
        historic_x_cols,
        future_x_cols,
        static_cols,
    )
    lock = _get_dataset_long_df_cache_lock()
    with lock:
        long_df = _DATASET_LONG_DF_CACHE.get(prepared_key)
    if long_df is not None:
        return {
            "spec": spec,
            "y_col_final": y_col_final,
            "long_df": long_df,
            "load_seconds": 0.0,
            "prepare_seconds": 0.0,
        }

    raw_bundle = get_or_build_dataset_frame(
        dataset=dataset_key,
        data_dir=data_dir_arg,
    )
    load_seconds = float(raw_bundle["load_seconds"])

    prepare_seconds = 0.0
    started = time.perf_counter()
    long_df = to_long(
        raw_bundle["df"],
        time_col=spec.time_col,
        y_col=y_col_final,
        id_cols=tuple(spec.group_cols),
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
        static_cols=static_cols,
        dropna=True,
    )
    prepare_seconds = float(time.perf_counter() - started)
    if long_df.empty:
        raise ValueError("Loaded 0 rows after to_long(dropna=True). Check dataset and y_col.")
    with lock:
        _DATASET_LONG_DF_CACHE[prepared_key] = long_df
        long_df = _DATASET_LONG_DF_CACHE[prepared_key]

    return {
        "spec": spec,
        "y_col_final": y_col_final,
        "long_df": long_df,
        "load_seconds": round(load_seconds, 6),
        "prepare_seconds": round(prepare_seconds, 6),
    }


__all__ = ["get_or_build_dataset_frame", "get_or_build_dataset_long_df"]
