from __future__ import annotations

from typing import Any

import pandas as pd

from ..contracts.frames import require_long_df
from .shared import AdapterFrameBundle, require_adapter_frame_bundle

__all__ = [
    "from_darts_bundle",
    "from_darts_timeseries",
    "to_darts_bundle",
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


def _timeseries_from_frame(darts_mod: Any, frame: pd.DataFrame) -> Any:
    index = pd.Index(pd.to_datetime(frame["ds"], errors="raise"), name="ds")
    value_cols = [col for col in frame.columns if col != "ds"]
    if len(value_cols) != 1:
        if hasattr(darts_mod.TimeSeries, "from_dataframe"):
            df = frame.loc[:, value_cols].copy()
            df.index = index
            return darts_mod.TimeSeries.from_dataframe(df)
        raise TypeError("Darts beta bundle currently requires TimeSeries.from_dataframe for multi-column covariates")

    series = pd.Series(
        frame[value_cols[0]].to_numpy(dtype=float, copy=False),
        index=index,
        name=value_cols[0],
        dtype=float,
    )
    return darts_mod.TimeSeries.from_series(series)


def _set_timeseries_unique_id(timeseries: Any, *, unique_id: str) -> Any:
    setattr(timeseries, "_foresight_unique_id", str(unique_id))
    return timeseries


def _apply_static_covariates(timeseries: Any, static_frame: pd.DataFrame | None) -> Any:
    if static_frame is None or static_frame.empty:
        return timeseries
    if hasattr(timeseries, "with_static_covariates"):
        return timeseries.with_static_covariates(static_frame)
    setattr(timeseries, "static_covariates", static_frame.copy())
    return timeseries


def _extract_static_covariates(timeseries: Any) -> pd.DataFrame | None:
    static_covariates = getattr(timeseries, "static_covariates", None)
    if isinstance(static_covariates, pd.DataFrame):
        return static_covariates.copy()
    return None


def _timeseries_to_frame(timeseries: Any, *, default_name: str) -> pd.DataFrame:
    if hasattr(timeseries, "pd_dataframe"):
        frame = timeseries.pd_dataframe()
        if isinstance(frame, pd.DataFrame):
            out = frame.copy()
            out = out.reset_index()
            first_col = str(out.columns[0])
            return out.rename(columns={first_col: "ds"})
    series = _pandas_series_from_timeseries(timeseries)
    return pd.DataFrame(
        {
            "ds": pd.Index(series.index),
            str(series.name or default_name): series.to_numpy(dtype=float, copy=False),
        }
    )


def _bundle_payload_for_group(
    darts_mod: Any,
    *,
    bundle: AdapterFrameBundle,
    unique_id: str,
) -> tuple[Any, Any | None, Any | None]:
    payload = bundle.payloads[str(unique_id)]
    target = _timeseries_from_frame(darts_mod, payload.target)
    target = _set_timeseries_unique_id(
        _apply_static_covariates(target, payload.static_covariates),
        unique_id=str(unique_id),
    )

    past = None
    if payload.historic_covariates is not None:
        past = _set_timeseries_unique_id(
            _timeseries_from_frame(darts_mod, payload.historic_covariates),
            unique_id=str(unique_id),
        )

    future = None
    if payload.future_covariates is not None:
        future = _set_timeseries_unique_id(
            _timeseries_from_frame(darts_mod, payload.future_covariates),
            unique_id=str(unique_id),
        )

    return target, past, future


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


def to_darts_bundle(data: Any) -> dict[str, Any]:
    darts_mod = _require_darts()
    bundle = require_adapter_frame_bundle(data)

    target_out: dict[str, Any] = {}
    past_out: dict[str, Any] = {}
    future_out: dict[str, Any] = {}
    freq_out: dict[str, str] = {}
    for unique_id in bundle.unique_ids:
        target, past, future = _bundle_payload_for_group(
            darts_mod,
            bundle=bundle,
            unique_id=unique_id,
        )
        target_out[str(unique_id)] = target
        if isinstance(bundle.freq, dict):
            freq_out[str(unique_id)] = str(bundle.freq[str(unique_id)])
        else:
            freq_out[str(unique_id)] = str(bundle.freq)
        if past is not None:
            past_out[str(unique_id)] = past
        if future is not None:
            future_out[str(unique_id)] = future
    return {
        "target": target_out,
        "past_covariates": past_out,
        "future_covariates": future_out,
        "freq": freq_out,
    }


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


def _bundle_items(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [(str(unique_id), obj) for unique_id, obj in sorted(value.items(), key=lambda item: str(item[0]))]
    unique_id = str(getattr(value, "_foresight_unique_id", "series=0"))
    return [(unique_id, value)]


def from_darts_bundle(data: Any) -> pd.DataFrame:
    if not isinstance(data, dict):
        raise TypeError("Darts beta bundle conversion expects a dict-like bundle")
    if "target" not in data or data.get("target") is None:
        raise ValueError("Darts beta bundle must include a non-empty 'target' payload")

    target_items = _bundle_items(data.get("target"))
    past_lookup = {str(unique_id): value for unique_id, value in _bundle_items(data.get("past_covariates", {}))}
    future_lookup = {str(unique_id): value for unique_id, value in _bundle_items(data.get("future_covariates", {}))}

    frames: list[pd.DataFrame] = []
    historic_cols: list[str] = []
    future_cols: list[str] = []
    static_cols: list[str] = []

    for unique_id, target_ts in target_items:
        target_frame = _timeseries_to_frame(target_ts, default_name="y")
        target_frame = target_frame.rename(columns={target_frame.columns[1]: "y"})
        target_frame.insert(0, "unique_id", str(unique_id))

        past_ts = past_lookup.get(str(unique_id))
        if past_ts is not None:
            past_frame = _timeseries_to_frame(past_ts, default_name="x")
            historic_cols.extend([str(col) for col in past_frame.columns if str(col) != "ds"])
            target_frame = target_frame.merge(past_frame, on="ds", how="left")

        future_ts = future_lookup.get(str(unique_id))
        if future_ts is not None:
            future_frame = _timeseries_to_frame(future_ts, default_name="x")
            future_cols.extend([str(col) for col in future_frame.columns if str(col) != "ds"])
            target_frame = target_frame.merge(future_frame, on="ds", how="left")

        static_frame = _extract_static_covariates(target_ts)
        if static_frame is not None:
            for col in static_frame.columns:
                target_frame[str(col)] = static_frame.iloc[0][col]
                static_cols.append(str(col))

        frames.append(target_frame)

    out = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    out = out.loc[:, ["unique_id", "ds", "y", *[col for col in out.columns if col not in {"unique_id", "ds", "y"}]]]
    out.attrs["historic_x_cols"] = tuple(dict.fromkeys(historic_cols))
    out.attrs["future_x_cols"] = tuple(dict.fromkeys(future_cols))
    out.attrs["static_cols"] = tuple(dict.fromkeys(static_cols))
    return out
