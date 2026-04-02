from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cli_runtime import compact_log_payload, emit_cli_event
from .contracts.capabilities import require_x_cols_if_needed as _contracts_require_x_cols_if_needed
from .contracts.params import normalize_covariate_roles as _normalize_covariate_roles
from .contracts.params import normalize_static_cols as _normalize_static_cols
from .dataset_long_df_cache import get_or_build_dataset_long_df
from .long_df_cache import (
    cached_ds_array,
    cached_series_slices,
    cached_split_sequence,
    cached_x_matrix,
    cached_y_array,
    cached_y_lookup,
    long_df_cache,
    sorted_long_df,
)
from .services import model_execution as _model_execution

N_WINDOWS_MIN_ERROR = "n_windows must be >= 1"


def _prediction_columns(*, pred_cols: tuple[str, ...] = ()) -> dict[str, list[Any]]:
    pred_cols_effective = tuple(dict.fromkeys(("yhat", *pred_cols)))
    columns = {
        "unique_id": [],
        "ds": [],
        "cutoff": [],
        "step": [],
        "y": [],
        "yhat": [],
        "model": [],
    }
    for col in pred_cols_effective:
        if col not in columns:
            columns[str(col)] = []
    return columns


def _append_prediction_rows(
    columns: dict[str, list[Any]],
    *,
    uid: str,
    ds_true: np.ndarray,
    cutoff: Any,
    y_true: np.ndarray,
    yhat: np.ndarray,
    model: str,
) -> None:
    horizon = int(y_true.shape[0])
    columns["unique_id"].extend([str(uid)] * horizon)
    columns["ds"].extend(_index_values_list(ds_true))
    columns["cutoff"].extend([cutoff] * horizon)
    columns["step"].extend(range(1, horizon + 1))
    columns["y"].extend(np.asarray(y_true, dtype=float).tolist())
    columns["yhat"].extend(np.asarray(yhat, dtype=float).tolist())
    columns["model"].extend([str(model)] * horizon)


def _index_values_list(values: Any) -> list[Any]:
    return pd.Index(values).tolist()


def _prediction_frame_from_columns(
    columns: dict[str, list[Any]],
    *,
    pred_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    pred_cols_effective = tuple(dict.fromkeys(("yhat", *pred_cols)))
    ordered_cols = ["unique_id", "ds", "cutoff", "step", "y", *pred_cols_effective, "model"]
    return pd.DataFrame(columns, columns=ordered_cols)


def _append_prediction_frame_rows(
    columns: dict[str, list[Any]],
    frame: pd.DataFrame,
    *,
    pred_cols: tuple[str, ...],
) -> None:
    pred_cols_effective = tuple(dict.fromkeys(("yhat", *pred_cols)))
    ordered_cols = ("unique_id", "ds", "cutoff", "step", "y", *pred_cols_effective, "model")
    for col in ordered_cols:
        columns[str(col)].extend(frame[str(col)].tolist())


def _normalize_cv_x_cols(
    model_spec: Any,
    model_params: dict[str, Any] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    interface = str(getattr(model_spec, "interface", "")).strip().lower()
    if interface == "multivariate":
        return (), (), ()
    return (
        *_normalize_covariate_roles(model_params),
        _normalize_static_cols(model_params or {}),
    )


def _normalize_cv_covariate_roles(
    model_spec: Any,
    model_params: dict[str, Any] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return _normalize_cv_x_cols(model_spec, model_params)


def _require_x_cols_if_needed(
    *,
    model: str,
    capabilities: dict[str, Any],
    x_cols: tuple[str, ...],
    context: str,
) -> None:
    _contracts_require_x_cols_if_needed(
        model=str(model),
        capabilities=capabilities,
        x_cols=x_cols,
        context=str(context),
    )


def _trim_cv_splits(
    df: pd.DataFrame,
    *,
    namespace: str,
    n_obs: int,
    horizon: int,
    step_size: int,
    min_train_size: int,
    max_train_size: int | None,
    n_windows: int | None,
    keep: str,
) -> tuple[Any, ...]:
    return cached_split_sequence(
        df,
        namespace=str(namespace),
        n_obs=int(n_obs),
        horizon=int(horizon),
        step_size=int(step_size),
        min_train_size=int(min_train_size),
        max_train_size=max_train_size,
        limit=n_windows,
        keep=str(keep),
        limit_error=N_WINDOWS_MIN_ERROR,
    )


def _local_cv_split_rows(
    *,
    columns: dict[str, list[Any]],
    uid: str,
    ds_arr: np.ndarray,
    y_arr: np.ndarray,
    split: Any,
    forecaster: Any,
    horizon: int,
    model: str,
) -> None:
    train = y_arr[split.train_start : split.train_end]
    yhat = np.asarray(forecaster(train, int(horizon)), dtype=float)
    if yhat.shape != (int(horizon),):
        raise ValueError(f"forecaster must return shape ({int(horizon)},), got {yhat.shape}")

    y_true = y_arr[split.test_start : split.test_end]
    ds_true = ds_arr[split.test_start : split.test_end]
    if y_true.shape != (int(horizon),) or ds_true.shape != (int(horizon),):
        raise RuntimeError("Internal error: unexpected slice length for horizon.")

    cutoff = ds_arr[split.train_end - 1]
    _append_prediction_rows(
        columns,
        uid=uid,
        ds_true=ds_true,
        cutoff=cutoff,
        y_true=y_true,
        yhat=yhat,
        model=model,
    )


def _local_cv_xreg_arrays(
    df: pd.DataFrame,
    *,
    x_cols: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    y_all = cached_y_array(df)
    x_all = cached_x_matrix(df, x_cols=x_cols)
    if np.isnan(y_all).any():
        raise ValueError("cross_validation_predictions_long_df does not support missing y values")
    if np.isnan(x_all).any():
        raise ValueError("cross_validation_predictions_long_df does not support missing x_cols values")
    return y_all, x_all


def _local_cv_xreg_split_rows(
    *,
    columns: dict[str, list[Any]],
    uid: str,
    ds_arr: np.ndarray,
    y_arr: np.ndarray,
    x_arr: np.ndarray,
    split: Any,
    horizon: int,
    model: str,
    model_params: dict[str, Any],
) -> None:
    yhat = _model_execution.call_local_xreg_forecaster(
        model=str(model),
        train_y=y_arr[split.train_start : split.train_end],
        horizon=int(horizon),
        train_exog=x_arr[split.train_start : split.train_end, :],
        future_exog=x_arr[split.test_start : split.test_end, :],
        model_params=model_params,
    )

    y_true = y_arr[split.test_start : split.test_end]
    ds_true = ds_arr[split.test_start : split.test_end]
    if y_true.shape != (int(horizon),) or ds_true.shape != (int(horizon),):
        raise RuntimeError("Internal error: unexpected slice length for horizon.")

    cutoff = ds_arr[split.train_end - 1]
    _append_prediction_rows(
        columns,
        uid=uid,
        ds_true=ds_true,
        cutoff=cutoff,
        y_true=y_true,
        yhat=yhat,
        model=model,
    )


def _local_cv_prediction_rows(
    df: pd.DataFrame,
    *,
    forecaster: Any,
    horizon: int,
    step_size: int,
    min_train_size: int,
    max_train_size: int | None,
    n_windows: int | None,
    model: str,
) -> tuple[pd.DataFrame, int, int, int]:
    columns = _prediction_columns()
    n_series = 0
    n_series_skipped = 0
    total_windows = 0
    series_slices = cached_series_slices(df)
    ds_all = cached_ds_array(df)
    y_all = cached_y_array(df)

    for uid, start, stop in series_slices:
        n_series += 1
        ds_arr = ds_all[start:stop]
        y_arr = y_all[start:stop]

        splits = _trim_cv_splits(
            df,
            namespace="cv_local",
            n_obs=int(y_arr.size),
            horizon=int(horizon),
            step_size=int(step_size),
            min_train_size=int(min_train_size),
            max_train_size=max_train_size,
            n_windows=n_windows,
            keep="last",
        )
        if not splits:
            n_series_skipped += 1
            emit_cli_event(
                "CV skip",
                event="cv_series_skipped",
                payload=compact_log_payload(unique_id=str(uid)),
                progress=True,
            )
            continue

        series_windows = 0
        for split in splits:
            _local_cv_split_rows(
                columns=columns,
                uid=str(uid),
                ds_arr=ds_arr,
                y_arr=y_arr,
                split=split,
                forecaster=forecaster,
                horizon=horizon,
                model=model,
            )
            series_windows += 1
            total_windows += 1
        emit_cli_event(
            "CV series",
            event="cv_series_completed",
            payload=compact_log_payload(
                unique_id=str(uid),
                windows=int(series_windows),
            ),
            progress=True,
        )

    return _prediction_frame_from_columns(columns), n_series, n_series_skipped, total_windows


def _local_cv_xreg_prediction_rows(
    df: pd.DataFrame,
    *,
    model: str,
    model_params: dict[str, Any],
    x_cols: tuple[str, ...],
    horizon: int,
    step_size: int,
    min_train_size: int,
    max_train_size: int | None,
    n_windows: int | None,
) -> tuple[pd.DataFrame, int, int, int]:
    columns = _prediction_columns()
    n_series = 0
    n_series_skipped = 0
    total_windows = 0
    local_xreg_params = dict(model_params)
    local_xreg_params.pop("x_cols", None)
    series_slices = cached_series_slices(df)
    ds_all = cached_ds_array(df)
    y_all, x_all = _local_cv_xreg_arrays(df, x_cols=x_cols)

    for uid, start, stop in series_slices:
        n_series += 1
        ds_arr = ds_all[start:stop]
        y_arr = y_all[start:stop]
        x_arr = x_all[start:stop, :]

        splits = _trim_cv_splits(
            df,
            namespace="cv_local_xreg",
            n_obs=int(y_arr.size),
            horizon=int(horizon),
            step_size=int(step_size),
            min_train_size=int(min_train_size),
            max_train_size=max_train_size,
            n_windows=n_windows,
            keep="last",
        )
        if not splits:
            n_series_skipped += 1
            emit_cli_event(
                "CV skip",
                event="cv_series_skipped",
                payload=compact_log_payload(unique_id=str(uid)),
                progress=True,
            )
            continue

        series_windows = 0
        for split in splits:
            _local_cv_xreg_split_rows(
                columns=columns,
                uid=str(uid),
                ds_arr=ds_arr,
                y_arr=y_arr,
                x_arr=x_arr,
                split=split,
                horizon=horizon,
                model=model,
                model_params=local_xreg_params,
            )
            series_windows += 1
            total_windows += 1
        emit_cli_event(
            "CV series",
            event="cv_series_completed",
            payload=compact_log_payload(
                unique_id=str(uid),
                windows=int(series_windows),
            ),
            progress=True,
        )

    return _prediction_frame_from_columns(columns), n_series, n_series_skipped, total_windows


def _global_cv_cutoffs(
    df: pd.DataFrame,
    *,
    horizon: int,
    step_size: int,
    min_train_size: int,
    max_train_size: int | None,
    n_windows: int | None,
) -> tuple[str, list[Any]]:
    series_slices = cached_series_slices(df)
    if not series_slices:
        raise ValueError("long_df is empty")

    ref_uid, start, stop = series_slices[0]
    ref_ds = cached_ds_array(df)[start:stop]
    splits = _trim_cv_splits(
        df,
        namespace="cv_global_cutoffs",
        n_obs=int(ref_ds.size),
        horizon=int(horizon),
        step_size=int(step_size),
        min_train_size=int(min_train_size),
        max_train_size=max_train_size,
        n_windows=n_windows,
        keep="last",
    )
    cutoffs = [ref_ds[split.train_end - 1] for split in splits]
    return str(ref_uid), cutoffs


def _global_cv_context_cache_key(
    *,
    horizon: int,
    step_size: int,
    min_train_size: int,
    max_train_size: int | None,
    n_windows: int | None,
) -> tuple[int, int, int, int | None, int | None]:
    return (
        int(horizon),
        int(step_size),
        int(min_train_size),
        None if max_train_size is None else int(max_train_size),
        None if n_windows is None else int(n_windows),
    )


def _get_cached_global_cv_context(
    df: pd.DataFrame,
    *,
    horizon: int,
    step_size: int,
    min_train_size: int,
    max_train_size: int | None,
    n_windows: int | None,
) -> dict[str, Any]:
    cache = long_df_cache(df)
    global_cache = cache.setdefault("global_cv_context", {})
    key = _global_cv_context_cache_key(
        horizon=horizon,
        step_size=step_size,
        min_train_size=min_train_size,
        max_train_size=max_train_size,
        n_windows=n_windows,
    )
    cached = global_cache.get(key)
    if isinstance(cached, dict):
        return cached

    ref_uid, cutoffs = _global_cv_cutoffs(
        df,
        horizon=int(horizon),
        step_size=int(step_size),
        min_train_size=int(min_train_size),
        max_train_size=max_train_size,
        n_windows=n_windows,
    )
    context = {
        "ref_uid": str(ref_uid),
        "cutoffs": list(cutoffs),
        "total_series": int(len(cached_series_slices(df))),
        "y_lookup": cached_y_lookup(df),
    }
    global_cache[key] = context
    return context


def _validated_global_cv_prediction_table(pred: Any) -> pd.DataFrame:
    if not isinstance(pred, pd.DataFrame):
        raise TypeError(
            f"Global forecaster must return a pandas DataFrame, got: {type(pred).__name__}"
        )

    required = {"unique_id", "ds", "yhat"}
    missing = required.difference(pred.columns)
    if missing:
        raise KeyError(f"Global prediction table missing columns: {sorted(missing)}")
    return pred


def _prepared_global_cv_columns(
    pred: pd.DataFrame,
    *,
    y_lookup: pd.Series,
    horizon: int,
    total_series: int,
    cutoff: Any,
    model: str,
) -> tuple[dict[str, list[Any]] | None, int, tuple[str, ...], int]:
    pred_cols = tuple(str(col) for col in pred.columns if col not in {"unique_id", "ds"})
    pred_index = pd.MultiIndex.from_frame(pred.loc[:, ["unique_id", "ds"]], names=["unique_id", "ds"])
    y_values = y_lookup.reindex(pred_index).to_numpy(copy=False)
    ds_values = pred["ds"].to_numpy(copy=False)
    uid_values = pred["unique_id"].astype("string").to_numpy(copy=False)
    pred_arrays = {col: pred[col].to_numpy(copy=False) for col in pred_cols}

    valid_mask = pd.notna(uid_values) & pd.notna(ds_values) & pd.notna(y_values)
    for col in pred_cols:
        valid_mask &= pd.notna(pred_arrays[col])
    if not bool(np.any(valid_mask)):
        return None, total_series, pred_cols, 0

    uid_filtered = np.asarray(uid_values[valid_mask], dtype=str)
    ds_filtered = ds_values[valid_mask]
    y_filtered = np.asarray(y_values[valid_mask], dtype=float)
    pred_filtered = {
        col: np.asarray(pred_arrays[col][valid_mask], dtype=float if np.issubdtype(np.asarray(pred_arrays[col]).dtype, np.number) else object)
        for col in pred_cols
    }

    ds_sort_key = np.asarray(ds_filtered).astype("datetime64[ns]").view("int64")
    sort_index = np.lexsort((ds_sort_key, uid_filtered))
    uid_sorted = uid_filtered[sort_index]
    ds_sorted = ds_filtered[sort_index]
    y_sorted = y_filtered[sort_index]
    pred_sorted = {col: values[sort_index] for col, values in pred_filtered.items()}

    if uid_sorted.size == 0:
        return None, total_series, pred_cols, 0

    boundaries = np.flatnonzero(uid_sorted[1:] != uid_sorted[:-1]) + 1 if uid_sorted.size > 1 else np.array([], dtype=int)
    starts = np.concatenate((np.array([0], dtype=int), boundaries.astype(int, copy=False)))
    stops = np.concatenate((boundaries.astype(int, copy=False), np.array([uid_sorted.size], dtype=int)))
    row_keep = np.zeros(uid_sorted.size, dtype=bool)
    steps = np.zeros(uid_sorted.size, dtype=int)
    valid_series = 0
    for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
        width = int(stop - start)
        if width != int(horizon):
            continue
        row_keep[start:stop] = True
        steps[start:stop] = np.arange(1, width + 1, dtype=int)
        valid_series += 1

    if not bool(np.any(row_keep)):
        return None, total_series, pred_cols, 0

    columns = _prediction_columns(pred_cols=pred_cols)
    row_count = int(np.count_nonzero(row_keep))
    kept = row_keep.tolist()
    columns["unique_id"].extend(uid_sorted[kept].tolist())
    columns["ds"].extend(_index_values_list(ds_sorted[kept]))
    columns["cutoff"].extend([cutoff] * row_count)
    columns["step"].extend(steps[kept].tolist())
    columns["y"].extend(y_sorted[kept].tolist())
    for col in pred_cols:
        columns[col].extend(pred_sorted[col][kept].tolist())
    columns["model"].extend([str(model)] * row_count)

    skipped_here = total_series - int(valid_series)
    return columns, skipped_here, pred_cols, row_count


def _prepared_global_cv_frame(
    pred: pd.DataFrame,
    *,
    y_lookup: pd.Series,
    horizon: int,
    total_series: int,
    cutoff: Any,
    model: str,
) -> tuple[pd.DataFrame | None, int, tuple[str, ...]]:
    columns, skipped_here, pred_cols, _row_count = _prepared_global_cv_columns(
        pred,
        y_lookup=y_lookup,
        horizon=horizon,
        total_series=total_series,
        cutoff=cutoff,
        model=model,
    )
    if columns is None:
        return None, skipped_here, pred_cols
    return _prediction_frame_from_columns(columns, pred_cols=pred_cols), skipped_here, pred_cols


def _global_cv_prediction_table(
    df: pd.DataFrame,
    *,
    model: str,
    model_params: dict[str, Any] | None,
    horizon: int,
    step_size: int,
    min_train_size: int,
    max_train_size: int | None,
    n_windows: int | None,
) -> tuple[pd.DataFrame, int, str, int]:
    global_params = dict(model_params or {})
    global_params["max_train_size"] = max_train_size
    global_forecaster = _model_execution.make_global_forecaster_runner(
        str(model),
        global_params,
    )
    context = _get_cached_global_cv_context(
        df,
        horizon=horizon,
        step_size=step_size,
        min_train_size=min_train_size,
        max_train_size=max_train_size,
        n_windows=n_windows,
    )
    ref_uid = str(context["ref_uid"])
    cutoffs = list(context["cutoffs"])

    total_series = int(context["total_series"])
    y_lookup = context["y_lookup"]
    pred_cols: tuple[str, ...] | None = None
    columns: dict[str, list[Any]] | None = None
    series_skipped_any = 0

    for cutoff_idx, cutoff in enumerate(cutoffs, start=1):
        pred = _validated_global_cv_prediction_table(global_forecaster(df, cutoff, int(horizon)))
        frame_columns, skipped_here, frame_pred_cols, row_count = _prepared_global_cv_columns(
            pred,
            y_lookup=y_lookup,
            horizon=horizon,
            total_series=total_series,
            cutoff=cutoff,
            model=model,
        )
        if frame_columns is None:
            continue
        if pred_cols is None:
            pred_cols = frame_pred_cols
            columns = _prediction_columns(pred_cols=pred_cols)
        elif frame_pred_cols != pred_cols:
            raise ValueError("Global prediction columns changed across CV cutoffs.")
        if skipped_here > 0:
            series_skipped_any += int(skipped_here)
        for col, values in frame_columns.items():
            columns[col].extend(values)
        emit_cli_event(
            f"CV cutoff {cutoff_idx}/{len(cutoffs)}",
            event="cv_cutoff_completed",
            payload=compact_log_payload(
                cutoff=cutoff,
                rows=int(row_count),
                series_skipped=int(skipped_here),
            ),
            progress=True,
        )

    if columns is None or pred_cols is None:
        raise ValueError("Global model produced 0 predictions for the requested CV parameters.")

    return (
        _prediction_frame_from_columns(columns, pred_cols=pred_cols),
        series_skipped_any,
        ref_uid,
        len(cutoffs),
    )


def cross_validation_predictions(
    *,
    model: str,
    dataset: str,
    horizon: int,
    step_size: int,
    min_train_size: int,
    y_col: str | None = None,
    model_params: dict[str, Any] | None = None,
    data_dir: str | Path | None = None,
    max_train_size: int | None = None,
    n_windows: int | None = None,
) -> pd.DataFrame:
    """
    Rolling-origin cross-validation that returns a tidy predictions table.

    Output columns:
      unique_id, ds, cutoff, step, y, yhat, model

    This mirrors the "predictions table" style used by many TS toolkits, and is
    a good foundation for interval calibration (e.g. conformal) and analysis.
    """
    model_spec = _model_execution.get_model_spec(str(model))
    historic_x_cols, future_x_cols, static_cols = _normalize_cv_x_cols(
        model_spec,
        model_params,
    )
    frame_bundle = get_or_build_dataset_long_df(
        dataset=str(dataset),
        y_col=y_col,
        data_dir=data_dir,
        model_params={
            "historic_x_cols": historic_x_cols,
            "future_x_cols": future_x_cols,
            "static_cols": static_cols,
        },
    )

    return cross_validation_predictions_long_df(
        model=str(model),
        long_df=frame_bundle["long_df"],
        horizon=int(horizon),
        step_size=int(step_size),
        min_train_size=int(min_train_size),
        model_params=model_params,
        max_train_size=max_train_size,
        n_windows=n_windows,
    )


def cross_validation_predictions_long_df(
    *,
    model: str,
    long_df: pd.DataFrame,
    horizon: int,
    step_size: int,
    min_train_size: int,
    model_params: dict[str, Any] | None = None,
    max_train_size: int | None = None,
    n_windows: int | None = None,
) -> pd.DataFrame:
    """
    Cross-validation predictions table for a canonical long DataFrame.

    Dispatches based on model interface:
      - local: per-series training with (train_1d, horizon) -> yhat
      - global: panel training with (long_df, cutoff, horizon) -> pred_df
    """
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df must be a pandas DataFrame")
    if long_df.empty:
        raise ValueError("long_df is empty")

    model_spec = _model_execution.get_model_spec(str(model))
    interface = str(model_spec.interface).lower().strip()
    params = dict(model_params or {})
    capabilities = dict(getattr(model_spec, "capabilities", {}))
    historic_x_cols, x_cols, static_cols = _normalize_cv_covariate_roles(model_spec, params)

    _require_x_cols_if_needed(
        model=str(model),
        capabilities=capabilities,
        x_cols=x_cols,
        context="cross_validation_predictions_long_df",
    )
    if interface == "local":
        if historic_x_cols:
            raise ValueError(
                "historic_x_cols are not yet supported in cross_validation_predictions_long_df"
            )
        if static_cols:
            if not bool(capabilities.get("supports_static_cols", False)):
                raise ValueError(
                    f"Model {model!r} does not support static_cols in cross_validation_predictions_long_df"
                )
            raise ValueError(
                "static_cols are not yet supported for local models in "
                "cross_validation_predictions_long_df"
            )
    elif interface == "global":
        if static_cols and not bool(capabilities.get("supports_static_cols", False)):
            raise ValueError(
                f"Model {model!r} does not support static_cols in cross_validation_predictions_long_df"
            )
    elif interface == "multivariate":
        raise ValueError(
            f"Model {model!r} is multivariate and cannot be used with "
            "`cross_validation_predictions_long_df()`."
        )

    df = sorted_long_df(long_df, reset_index=False)
    emit_cli_event(
        "CV start",
        event="cv_started",
        payload=compact_log_payload(
            model=str(model),
            interface=interface,
            horizon=int(horizon),
            step_size=int(step_size),
            min_train_size=int(min_train_size),
            n_series=int(df["unique_id"].nunique()),
        ),
    )

    if interface == "local":
        if x_cols:
            if not bool(capabilities.get("supports_x_cols", False)):
                raise ValueError(
                    f"Model {model!r} does not support x_cols in cross_validation_predictions_long_df"
                )
            out, n_series, n_series_skipped, n_windows = _local_cv_xreg_prediction_rows(
                df,
                model=str(model),
                model_params=params,
                x_cols=x_cols,
                horizon=int(horizon),
                step_size=int(step_size),
                min_train_size=int(min_train_size),
                max_train_size=max_train_size,
                n_windows=n_windows,
            )
        else:
            forecaster = _model_execution.make_local_forecaster_runner(str(model), params)
            out, n_series, n_series_skipped, n_windows = _local_cv_prediction_rows(
                df,
                forecaster=forecaster,
                horizon=int(horizon),
                step_size=int(step_size),
                min_train_size=int(min_train_size),
                max_train_size=max_train_size,
                n_windows=n_windows,
                model=str(model),
            )

        if out.empty:
            raise ValueError("No series had enough data for the requested CV parameters.")

        out.attrs["n_series"] = int(n_series)
        out.attrs["n_series_skipped"] = int(n_series_skipped)
        out.attrs["n_windows"] = int(n_windows)
        emit_cli_event(
            "CV done",
            event="cv_completed",
            payload=compact_log_payload(
                model=str(model),
                rows=int(len(out)),
                n_series=int(n_series),
                n_series_skipped=int(n_series_skipped),
                n_windows=int(n_windows),
            ),
        )
        return out

    if interface == "global":
        out, series_skipped_any, ref_uid, n_windows = _global_cv_prediction_table(
            df,
            model=str(model),
            model_params=params,
            horizon=int(horizon),
            step_size=int(step_size),
            min_train_size=int(min_train_size),
            max_train_size=max_train_size,
            n_windows=n_windows,
        )

        out.attrs["n_series"] = int(df["unique_id"].nunique())
        out.attrs["n_series_skipped"] = int(series_skipped_any)
        out.attrs["reference_unique_id"] = str(ref_uid)
        out.attrs["n_windows"] = int(n_windows)
        emit_cli_event(
            "CV done",
            event="cv_completed",
            payload=compact_log_payload(
                model=str(model),
                rows=int(len(out)),
                n_series=int(out.attrs["n_series"]),
                n_series_skipped=int(series_skipped_any),
                n_windows=int(n_windows),
            ),
        )
        return out

    raise ValueError(f"Unknown model interface: {model_spec.interface!r}")
