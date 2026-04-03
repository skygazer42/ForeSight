from __future__ import annotations

from typing import Any

import pandas as pd

from ..contracts.frames import require_long_df

__all__ = [
    "from_darts_timeseries",
    "to_darts_timeseries",
]


def _require_darts() -> Any:
    try:
        import darts
    except Exception as e:  # noqa: BLE001
        from ..optional_deps import missing_dependency_message

        raise ImportError(missing_dependency_message("darts", subject="darts adapter")) from e
    return darts


def _to_darts_series(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(float, copy=False)
    raise TypeError("Darts conversion expects a pandas Series or a canonical long DataFrame")


def to_darts_timeseries(data: Any) -> Any:
    darts_mod = _require_darts()

    if isinstance(data, pd.Series):
        return darts_mod.TimeSeries.from_series(_to_darts_series(data))

    long_df = require_long_df(data, require_non_empty=False)
    out: dict[str, Any] = {}
    for unique_id, group in long_df.groupby("unique_id", sort=True):
        series = pd.Series(
            group["y"].to_numpy(dtype=float, copy=False),
            index=pd.Index(group["ds"]),
            name="y",
        )
        out[str(unique_id)] = darts_mod.TimeSeries.from_series(series)
    return out


def _pandas_series_from_timeseries(timeseries: Any) -> pd.Series:
    if hasattr(timeseries, "pd_series"):
        series = timeseries.pd_series()
        if isinstance(series, pd.Series):
            return series.astype(float, copy=False)
    if hasattr(timeseries, "to_series"):
        series = timeseries.to_series()
        if isinstance(series, pd.Series):
            return series.astype(float, copy=False)
    raise TypeError("Unsupported Darts TimeSeries object; expected pd_series() or to_series()")


def from_darts_timeseries(data: Any) -> Any:
    if isinstance(data, dict):
        rows: list[dict[str, Any]] = []
        for unique_id, timeseries in sorted(data.items(), key=lambda item: str(item[0])):
            series = _pandas_series_from_timeseries(timeseries)
            for ds, value in series.items():
                rows.append({"unique_id": str(unique_id), "ds": ds, "y": float(value)})
        return pd.DataFrame(rows, columns=["unique_id", "ds", "y"])

    return _pandas_series_from_timeseries(data)
