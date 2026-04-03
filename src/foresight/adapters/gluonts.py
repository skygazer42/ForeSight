from __future__ import annotations

from typing import Any

import pandas as pd

from ..contracts.frames import require_long_df

__all__ = ["to_gluonts_list_dataset"]


def _require_gluonts() -> Any:
    try:
        import gluonts
    except Exception as e:  # noqa: BLE001
        from ..optional_deps import missing_dependency_message

        raise ImportError(missing_dependency_message("gluonts", subject="gluonts adapter")) from e
    return gluonts


def _infer_series_frequency(ds: pd.Series) -> str:
    if len(ds) < 2:
        raise ValueError("freq must be provided when a series has fewer than 2 timestamps")

    values = pd.to_datetime(ds, errors="raise")
    if len(values) >= 3:
        inferred = pd.infer_freq(values)
        if inferred is not None:
            return str(inferred)

    delta = values.iloc[1] - values.iloc[0]
    synthetic = pd.DatetimeIndex([values.iloc[0], values.iloc[1], values.iloc[1] + delta])
    inferred = pd.infer_freq(synthetic)
    if inferred is not None:
        return str(inferred)

    offset = pd.tseries.frequencies.to_offset(delta)
    if offset is None:
        raise ValueError("Could not infer a regular frequency; provide freq explicitly")
    return str(offset.freqstr)


def to_gluonts_list_dataset(
    data: Any,
    *,
    freq: str | None = None,
) -> Any:
    gluonts_mod = _require_gluonts()
    ListDataset = gluonts_mod.dataset.common.ListDataset

    if isinstance(data, pd.Series):
        series = data.astype(float, copy=False)
        freq_s = str(freq or _infer_series_frequency(series.index.to_series()))
        return ListDataset(
            [{"start": series.index[0], "target": series.tolist()}],
            freq=freq_s,
        )

    long_df = require_long_df(data, require_non_empty=False)
    rows = []
    for unique_id, group in long_df.groupby("unique_id", sort=True):
        group_ds = pd.Series(group["ds"]).reset_index(drop=True)
        freq_s = str(freq or _infer_series_frequency(group_ds))
        rows.append(
            {
                "start": pd.Timestamp(group["ds"].iloc[0]),
                "target": group["y"].to_numpy(dtype=float, copy=False).tolist(),
                "item_id": str(unique_id),
            }
        )
    return ListDataset(rows, freq=freq_s)
