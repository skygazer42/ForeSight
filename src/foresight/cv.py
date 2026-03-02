from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.format import to_long
from .datasets.loaders import load_dataset
from .datasets.registry import get_dataset_spec
from .models.registry import get_model_spec, make_forecaster, make_global_forecaster
from .splits import rolling_origin_splits


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

    model_spec = get_model_spec(str(model))
    x_cols: tuple[str, ...] = ()
    if model_spec.interface == "global" and model_params and "x_cols" in model_params:
        raw = model_params.get("x_cols")
        if raw is not None:
            if isinstance(raw, str):
                x_cols = tuple([p.strip() for p in raw.split(",") if p.strip()])
            elif isinstance(raw, list | tuple):
                x_cols = tuple([str(p).strip() for p in raw if str(p).strip()])
            else:
                s = str(raw).strip()
                x_cols = (s,) if s else ()

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

    model_spec = get_model_spec(str(model))
    interface = str(model_spec.interface).lower().strip()

    df = long_df.sort_values(["unique_id", "ds"], kind="mergesort")

    if interface == "local":
        forecaster = make_forecaster(str(model), **(model_params or {}))

        rows: list[dict[str, Any]] = []
        n_series = 0
        n_series_skipped = 0

        for uid, g in df.groupby("unique_id", sort=False):
            n_series += 1
            ds_arr = g["ds"].to_numpy(copy=False)
            y_arr = g["y"].to_numpy(dtype=float, copy=False)

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

            if n_windows is not None:
                if int(n_windows) <= 0:
                    raise ValueError("n_windows must be >= 1")
                splits = splits[-int(n_windows) :]

            for split in splits:
                train = y_arr[split.train_start : split.train_end]
                yhat = np.asarray(forecaster(train, int(horizon)), dtype=float)
                if yhat.shape != (int(horizon),):
                    raise ValueError(
                        f"forecaster must return shape ({int(horizon)},), got {yhat.shape}"
                    )

                y_true = y_arr[split.test_start : split.test_end]
                ds_true = ds_arr[split.test_start : split.test_end]
                if y_true.shape != (int(horizon),) or ds_true.shape != (int(horizon),):
                    raise RuntimeError("Internal error: unexpected slice length for horizon.")

                cutoff = ds_arr[split.train_end - 1]
                for i in range(int(horizon)):
                    rows.append(
                        {
                            "unique_id": str(uid),
                            "ds": ds_true[i],
                            "cutoff": cutoff,
                            "step": int(i + 1),
                            "y": float(y_true[i]),
                            "yhat": float(yhat[i]),
                            "model": str(model),
                        }
                    )

        if not rows:
            raise ValueError("No series had enough data for the requested CV parameters.")

        out = pd.DataFrame(rows)
        out.attrs["n_series"] = int(n_series)
        out.attrs["n_series_skipped"] = int(n_series_skipped)
        return out

    if interface == "global":
        global_forecaster = make_global_forecaster(
            str(model), **(model_params or {}), max_train_size=max_train_size
        )

        # Choose global cutoffs from a reference series (first group). This matches many panel datasets
        # where series share a common timestamp grid, while still allowing missing series to be skipped.
        ref_uid, ref_g = next(iter(df.groupby("unique_id", sort=False)))
        ref_ds = ref_g["ds"].to_numpy(copy=False)
        splits = list(
            rolling_origin_splits(
                int(ref_ds.size),
                horizon=int(horizon),
                step_size=int(step_size),
                min_train_size=int(min_train_size),
                max_train_size=max_train_size,
            )
        )
        if n_windows is not None:
            if int(n_windows) <= 0:
                raise ValueError("n_windows must be >= 1")
            splits = splits[-int(n_windows) :]

        cutoffs = [ref_ds[sp.train_end - 1] for sp in splits]

        total_series = int(df["unique_id"].nunique())
        series_skipped_any = 0
        frames: list[pd.DataFrame] = []

        y_lookup = df[["unique_id", "ds", "y"]]

        for cutoff in cutoffs:
            pred = global_forecaster(df, cutoff, int(horizon))
            if not isinstance(pred, pd.DataFrame):
                raise TypeError(
                    f"Global forecaster must return a pandas DataFrame, got: {type(pred).__name__}"
                )
            required = {"unique_id", "ds", "yhat"}
            missing = required.difference(pred.columns)
            if missing:
                raise KeyError(f"Global prediction table missing columns: {sorted(missing)}")

            # Keep any extra prediction columns (e.g. quantiles like yhat_p10/yhat_p90).
            pred_cols = [c for c in pred.columns if c not in {"unique_id", "ds"}]

            merged = pred.merge(y_lookup, on=["unique_id", "ds"], how="left", validate="one_to_one")
            # Some series may be skipped by the model; keep only rows with observed y for metrics.
            merged = merged.dropna(subset=["y", "ds", *pred_cols])
            if merged.empty:
                continue

            merged = merged.sort_values(["unique_id", "ds"], kind="mergesort")
            sizes = merged.groupby("unique_id", sort=False).size()
            valid_uids = sizes.index[sizes.to_numpy(dtype=int, copy=False) == int(horizon)]
            if len(valid_uids) == 0:
                continue

            merged = merged[merged["unique_id"].isin(valid_uids)].copy()
            merged["step"] = merged.groupby("unique_id", sort=False).cumcount() + 1
            merged["cutoff"] = cutoff
            merged["model"] = str(model)

            skipped_here = total_series - int(merged["unique_id"].nunique())
            if skipped_here > 0:
                series_skipped_any += int(skipped_here)

            cols = ["unique_id", "ds", "cutoff", "step", "y", *pred_cols, "model"]
            frames.append(merged.loc[:, cols])

        if not frames:
            raise ValueError("Global model produced 0 predictions for the requested CV parameters.")

        out = pd.concat(frames, axis=0, ignore_index=True)
        out.attrs["n_series"] = total_series
        out.attrs["n_series_skipped"] = int(series_skipped_any)
        out.attrs["reference_unique_id"] = str(ref_uid)
        return out

    raise ValueError(f"Unknown model interface: {model_spec.interface!r}")

    rows: list[dict[str, Any]] = []
    n_series = 0
    n_series_skipped = 0

    for uid, g in long_df.groupby("unique_id", sort=False):
        n_series += 1
        ds_arr = g["ds"].to_numpy(copy=False)
        y_arr = g["y"].to_numpy(dtype=float, copy=False)

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

        if n_windows is not None:
            if int(n_windows) <= 0:
                raise ValueError("n_windows must be >= 1")
            splits = splits[-int(n_windows) :]

        for split in splits:
            train = y_arr[split.train_start : split.train_end]
            yhat = np.asarray(forecaster(train, int(horizon)), dtype=float)
            if yhat.shape != (int(horizon),):
                raise ValueError(
                    f"forecaster must return shape ({int(horizon)},), got {yhat.shape}"
                )

            y_true = y_arr[split.test_start : split.test_end]
            ds_true = ds_arr[split.test_start : split.test_end]
            if y_true.shape != (int(horizon),) or ds_true.shape != (int(horizon),):
                raise RuntimeError("Internal error: unexpected slice length for horizon.")

            cutoff = ds_arr[split.train_end - 1]
            for i in range(int(horizon)):
                rows.append(
                    {
                        "unique_id": str(uid),
                        "ds": ds_true[i],
                        "cutoff": cutoff,
                        "step": int(i + 1),
                        "y": float(y_true[i]),
                        "yhat": float(yhat[i]),
                        "model": str(model),
                    }
                )

    if not rows:
        raise ValueError("No series had enough data for the requested CV parameters.")

    out = pd.DataFrame(rows)
    out.attrs["n_series"] = int(n_series)
    out.attrs["n_series_skipped"] = int(n_series_skipped)
    return out
