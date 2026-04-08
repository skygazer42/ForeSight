from __future__ import annotations

from typing import Any

import pandas as pd

from ..contracts.frames import require_long_df
from ..optional_deps import require_dependency
from .shared import require_adapter_frame_bundle

__all__ = ["from_gluonts_bundle", "to_gluonts_bundle", "to_gluonts_list_dataset"]
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
    gluonts_mod = require_dependency("gluonts", subject="gluonts adapter")
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


def _frame_to_feature_major_lists(frame: pd.DataFrame | None) -> list[list[float]]:
    if frame is None:
        return []
    value_cols = [str(col) for col in frame.columns if str(col) != "ds"]
    return [frame[str(col)].to_numpy(dtype=float, copy=False).tolist() for col in value_cols]


def to_gluonts_bundle(data: Any) -> dict[str, Any]:
    require_dependency("gluonts", subject="gluonts adapter")
    bundle = require_adapter_frame_bundle(data)

    target: dict[str, dict[str, object]] = {}
    past_feat_dynamic_real: dict[str, list[list[float]]] = {}
    feat_dynamic_real: dict[str, list[list[float]]] = {}
    feat_static_real: dict[str, list[float]] = {}

    for unique_id in bundle.unique_ids:
        payload = bundle.payloads[str(unique_id)]
        target[str(unique_id)] = {
            "start": pd.Timestamp(payload.target["ds"].iloc[0]),
            "target": payload.target["y"].to_numpy(dtype=float, copy=False).tolist(),
            "item_id": str(unique_id),
        }
        if payload.historic_covariates is not None:
            past_feat_dynamic_real[str(unique_id)] = _frame_to_feature_major_lists(
                payload.historic_covariates
            )
        if payload.future_covariates is not None:
            feat_dynamic_real[str(unique_id)] = _frame_to_feature_major_lists(
                payload.future_covariates
            )
        if payload.static_covariates is not None:
            feat_static_real[str(unique_id)] = [
                float(payload.static_covariates.iloc[0][col])
                for col in payload.static_covariates.columns
            ]

    return {
        "target": target,
        "past_feat_dynamic_real": past_feat_dynamic_real,
        "feat_dynamic_real": feat_dynamic_real,
        "feat_static_real": feat_static_real,
        "feature_names": {
            "historic_x_cols": bundle.covariates.historic_x_cols,
            "future_x_cols": bundle.covariates.future_x_cols,
            "static_cols": bundle.covariates.static_cols,
        },
        "freq": (
            bundle.freq
            if isinstance(bundle.freq, dict)
            else {str(bundle.unique_ids[0]): str(bundle.freq)}
        ),
    }


def from_gluonts_bundle(data: Any) -> pd.DataFrame:
    if not isinstance(data, dict):
        raise TypeError("GluonTS beta bundle conversion expects a dict-like bundle")
    if "target" not in data or data.get("target") is None:
        raise ValueError("GluonTS beta bundle must include a non-empty 'target' payload")

    target = dict(data.get("target", {}))
    past_feat_dynamic_real = {
        str(unique_id): value
        for unique_id, value in dict(data.get("past_feat_dynamic_real", {})).items()
    }
    feat_dynamic_real = {
        str(unique_id): value
        for unique_id, value in dict(data.get("feat_dynamic_real", {})).items()
    }
    feat_static_real = {
        str(unique_id): value for unique_id, value in dict(data.get("feat_static_real", {})).items()
    }
    feature_names = dict(data.get("feature_names", {}))
    historic_cols = tuple(str(col) for col in feature_names.get("historic_x_cols", ()) or ())
    future_cols = tuple(str(col) for col in feature_names.get("future_x_cols", ()) or ())
    static_cols = tuple(str(col) for col in feature_names.get("static_cols", ()) or ())

    rows: list[dict[str, object]] = []
    for unique_id, payload in sorted(target.items(), key=lambda item: str(item[0])):
        start = pd.Timestamp(payload["start"])
        target_values = [float(value) for value in payload["target"]]
        freq_map = dict(data.get("freq", {}))
        freq = str(freq_map.get(str(unique_id), "D"))
        ds_values = pd.date_range(start=start, periods=len(target_values), freq=freq)

        historic_values = past_feat_dynamic_real.get(str(unique_id), [])
        future_values = feat_dynamic_real.get(str(unique_id), [])
        static_values = feat_static_real.get(str(unique_id), [])
        static_lookup = {col: static_values[idx] for idx, col in enumerate(static_cols)}

        for idx, (ds, y_value) in enumerate(zip(ds_values, target_values)):
            row: dict[str, object] = {
                "unique_id": str(unique_id),
                "ds": pd.Timestamp(ds),
                "y": float(y_value),
            }
            for col_idx, col in enumerate(historic_cols):
                row[str(col)] = float(historic_values[col_idx][idx])
            for col_idx, col in enumerate(future_cols):
                row[str(col)] = float(future_values[col_idx][idx])
            for col in static_cols:
                row[str(col)] = float(static_lookup[str(col)])
            rows.append(row)

    out = pd.DataFrame(rows)
    out = out.loc[
        :,
        [
            "unique_id",
            "ds",
            "y",
            *[col for col in out.columns if col not in {"unique_id", "ds", "y"}],
        ],
    ]
    out.attrs["historic_x_cols"] = historic_cols
    out.attrs["future_x_cols"] = future_cols
    out.attrs["static_cols"] = static_cols
    return out
