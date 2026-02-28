from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.format import to_long
from .datasets.loaders import load_dataset
from .datasets.registry import get_dataset_spec
from .models.registry import make_forecaster
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
    long_df = to_long(
        df,
        time_col=spec.time_col,
        y_col=y_col_final,
        id_cols=tuple(spec.group_cols),
        dropna=True,
    )
    if long_df.empty:
        raise ValueError("Loaded 0 rows after to_long(dropna=True). Check dataset and y_col.")

    long_df = long_df.sort_values(["unique_id", "ds"], kind="mergesort")
    forecaster = make_forecaster(str(model), **(model_params or {}))

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
