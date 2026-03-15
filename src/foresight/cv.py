from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.format import to_long
from .datasets.loaders import load_dataset
from .datasets.registry import get_dataset_spec
from .splits import rolling_origin_splits
from .services import model_execution as _model_execution

N_WINDOWS_MIN_ERROR = "n_windows must be >= 1"


def _normalize_cv_x_cols(model_spec: Any, model_params: dict[str, Any] | None) -> tuple[str, ...]:
    if model_spec.interface != "global" or not model_params or "x_cols" not in model_params:
        return ()

    raw = model_params.get("x_cols")
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(part).strip() for part in raw if str(part).strip())

    value = str(raw).strip()
    return (value,) if value else ()


def _trim_cv_splits(splits: list[Any], *, n_windows: int | None) -> list[Any]:
    if n_windows is None:
        return splits
    if int(n_windows) <= 0:
        raise ValueError(N_WINDOWS_MIN_ERROR)
    return splits[-int(n_windows) :]


def _local_cv_split_rows(
    *,
    uid: str,
    ds_arr: np.ndarray,
    y_arr: np.ndarray,
    split: Any,
    forecaster: Any,
    horizon: int,
    model: str,
) -> list[dict[str, Any]]:
    train = y_arr[split.train_start : split.train_end]
    yhat = np.asarray(forecaster(train, int(horizon)), dtype=float)
    if yhat.shape != (int(horizon),):
        raise ValueError(f"forecaster must return shape ({int(horizon)},), got {yhat.shape}")

    y_true = y_arr[split.test_start : split.test_end]
    ds_true = ds_arr[split.test_start : split.test_end]
    if y_true.shape != (int(horizon),) or ds_true.shape != (int(horizon),):
        raise RuntimeError("Internal error: unexpected slice length for horizon.")

    cutoff = ds_arr[split.train_end - 1]
    return [
        {
            "unique_id": uid,
            "ds": ds_true[idx],
            "cutoff": cutoff,
            "step": int(idx + 1),
            "y": float(y_true[idx]),
            "yhat": float(yhat[idx]),
            "model": str(model),
        }
        for idx in range(int(horizon))
    ]


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
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    n_series = 0
    n_series_skipped = 0

    for uid, group in df.groupby("unique_id", sort=False):
        n_series += 1
        ds_arr = group["ds"].to_numpy(copy=False)
        y_arr = group["y"].to_numpy(dtype=float, copy=False)

        try:
            splits = list(
                rolling_origin_splits(
                    y_arr.size,
                    horizon=int(horizon),
                    step_size=int(step_size),
                    min_train_size=int(min_train_size),
                    max_train_size=max_train_size,
                )
            )
        except ValueError:
            n_series_skipped += 1
            continue

        for split in _trim_cv_splits(splits, n_windows=n_windows):
            rows.extend(
                _local_cv_split_rows(
                    uid=str(uid),
                    ds_arr=ds_arr,
                    y_arr=y_arr,
                    split=split,
                    forecaster=forecaster,
                    horizon=horizon,
                    model=model,
                )
            )

    return rows, n_series, n_series_skipped


def _global_cv_cutoffs(
    df: pd.DataFrame,
    *,
    horizon: int,
    step_size: int,
    min_train_size: int,
    max_train_size: int | None,
    n_windows: int | None,
) -> tuple[str, list[Any]]:
    ref_uid, ref_group = next(iter(df.groupby("unique_id", sort=False)))
    ref_ds = ref_group["ds"].to_numpy(copy=False)
    splits = list(
        rolling_origin_splits(
            int(ref_ds.size),
            horizon=int(horizon),
            step_size=int(step_size),
            min_train_size=int(min_train_size),
            max_train_size=max_train_size,
        )
    )
    trimmed_splits = _trim_cv_splits(splits, n_windows=n_windows)
    cutoffs = [ref_ds[split.train_end - 1] for split in trimmed_splits]
    return str(ref_uid), cutoffs


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


def _prepared_global_cv_frame(
    pred: pd.DataFrame,
    *,
    y_lookup: pd.DataFrame,
    horizon: int,
    total_series: int,
    cutoff: Any,
    model: str,
) -> tuple[pd.DataFrame | None, int]:
    pred_cols = [col for col in pred.columns if col not in {"unique_id", "ds"}]
    merged = pred.merge(y_lookup, on=["unique_id", "ds"], how="left", validate="one_to_one")
    merged = merged.dropna(subset=["y", "ds", *pred_cols])
    if merged.empty:
        return None, total_series

    merged = merged.sort_values(["unique_id", "ds"], kind="mergesort")
    sizes = merged.groupby("unique_id", sort=False).size()
    valid_uids = sizes.index[sizes.to_numpy(dtype=int, copy=False) == int(horizon)]
    if len(valid_uids) == 0:
        return None, total_series

    merged = merged[merged["unique_id"].isin(valid_uids)].copy()
    merged["step"] = merged.groupby("unique_id", sort=False).cumcount() + 1
    merged["cutoff"] = cutoff
    merged["model"] = str(model)
    skipped_here = total_series - int(merged["unique_id"].nunique())

    cols = ["unique_id", "ds", "cutoff", "step", "y", *pred_cols, "model"]
    return merged.loc[:, cols], skipped_here


def _global_cv_prediction_frames(
    df: pd.DataFrame,
    *,
    model: str,
    model_params: dict[str, Any] | None,
    horizon: int,
    step_size: int,
    min_train_size: int,
    max_train_size: int | None,
    n_windows: int | None,
) -> tuple[list[pd.DataFrame], int, str]:
    global_params = dict(model_params or {})
    global_params["max_train_size"] = max_train_size
    global_forecaster = _model_execution.make_global_forecaster_runner(
        str(model),
        global_params,
    )
    ref_uid, cutoffs = _global_cv_cutoffs(
        df,
        horizon=horizon,
        step_size=step_size,
        min_train_size=min_train_size,
        max_train_size=max_train_size,
        n_windows=n_windows,
    )

    total_series = int(df["unique_id"].nunique())
    y_lookup = df[["unique_id", "ds", "y"]]
    frames: list[pd.DataFrame] = []
    series_skipped_any = 0

    for cutoff in cutoffs:
        pred = _validated_global_cv_prediction_table(global_forecaster(df, cutoff, int(horizon)))
        frame, skipped_here = _prepared_global_cv_frame(
            pred,
            y_lookup=y_lookup,
            horizon=horizon,
            total_series=total_series,
            cutoff=cutoff,
            model=model,
        )
        if frame is None:
            continue
        if skipped_here > 0:
            series_skipped_any += int(skipped_here)
        frames.append(frame)

    return frames, series_skipped_any, ref_uid


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
    spec = get_dataset_spec(str(dataset))
    y_col_final = (
        str(y_col).strip() if (y_col is not None and str(y_col).strip()) else spec.default_y
    )

    df = load_dataset(str(dataset), data_dir=data_dir)

    model_spec = _model_execution.get_model_spec(str(model))
    x_cols = _normalize_cv_x_cols(model_spec, model_params)

    long_df = to_long(
        df,
        time_col=spec.time_col,
        y_col=y_col_final,
        id_cols=tuple(spec.group_cols),
        x_cols=x_cols,
        dropna=True,
    )
    if long_df.empty:
        raise ValueError("Loaded 0 rows after to_long(dropna=True). Check dataset and y_col.")

    return cross_validation_predictions_long_df(
        model=str(model),
        long_df=long_df,
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

    df = long_df.sort_values(["unique_id", "ds"], kind="mergesort")

    if interface == "local":
        forecaster = _model_execution.make_local_forecaster_runner(str(model), model_params)
        rows, n_series, n_series_skipped = _local_cv_prediction_rows(
            df,
            forecaster=forecaster,
            horizon=int(horizon),
            step_size=int(step_size),
            min_train_size=int(min_train_size),
            max_train_size=max_train_size,
            n_windows=n_windows,
            model=str(model),
        )

        if not rows:
            raise ValueError("No series had enough data for the requested CV parameters.")

        out = pd.DataFrame(rows)
        out.attrs["n_series"] = int(n_series)
        out.attrs["n_series_skipped"] = int(n_series_skipped)
        return out

    if interface == "global":
        frames, series_skipped_any, ref_uid = _global_cv_prediction_frames(
            df,
            model=str(model),
            model_params=model_params,
            horizon=int(horizon),
            step_size=int(step_size),
            min_train_size=int(min_train_size),
            max_train_size=max_train_size,
            n_windows=n_windows,
        )

        if not frames:
            raise ValueError("Global model produced 0 predictions for the requested CV parameters.")

        out = pd.concat(frames, axis=0, ignore_index=True)
        out.attrs["n_series"] = int(df["unique_id"].nunique())
        out.attrs["n_series_skipped"] = int(series_skipped_any)
        out.attrs["reference_unique_id"] = str(ref_uid)
        return out

    raise ValueError(f"Unknown model interface: {model_spec.interface!r}")
