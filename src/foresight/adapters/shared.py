from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..contracts.covariates import CovariateSpec, resolve_covariate_roles
from ..contracts.frames import require_long_df


@dataclass(frozen=True)
class AdapterSeriesPayload:
    target: pd.DataFrame
    historic_covariates: pd.DataFrame | None = None
    future_covariates: pd.DataFrame | None = None
    static_covariates: pd.DataFrame | None = None
    freq: str = "D"


@dataclass(frozen=True)
class AdapterFrameBundle:
    covariates: CovariateSpec
    freq: str | dict[str, str]
    unique_ids: tuple[str, ...]
    payloads: dict[str, AdapterSeriesPayload]


def _normalize_bundle_freq(bundle: AdapterFrameBundle) -> dict[str, str]:
    if isinstance(bundle.freq, dict):
        return {str(unique_id): str(value) for unique_id, value in bundle.freq.items()}
    if not bundle.unique_ids:
        return {}
    return {str(bundle.unique_ids[0]): str(bundle.freq)}


def infer_adapter_frequency(ds: pd.Series) -> str:
    values = pd.to_datetime(ds, errors="raise")
    if len(values) < 2:
        raise ValueError(
            "Darts beta bundle conversion requires explicit freq or at least 2 timestamps per series"
        )

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
    return str(offset.freqstr)


def _build_static_covariates_frame(
    group: pd.DataFrame,
    *,
    static_cols: tuple[str, ...],
    unique_id: str,
) -> pd.DataFrame | None:
    if not static_cols:
        return None

    values: dict[str, Any] = {}
    for col in static_cols:
        observed = pd.Series(group[col]).dropna()
        if observed.empty:
            raise ValueError(
                f"static_cols column {col!r} has no observed value for unique_id={unique_id!r}"
            )
        unique_values = pd.unique(observed.to_numpy(copy=False))
        if len(unique_values) != 1:
            raise ValueError(
                f"static_cols column {col!r} must be constant within unique_id={unique_id!r}"
            )
        values[str(col)] = unique_values[0]
    return pd.DataFrame([values])


def require_adapter_frame_bundle(data: Any) -> AdapterFrameBundle:
    long_df = require_long_df(data, require_non_empty=False).copy()
    long_df = long_df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    covariates = resolve_covariate_roles(
        historic_x_cols=tuple(long_df.attrs.get("historic_x_cols", ()) or ()),
        future_x_cols=tuple(long_df.attrs.get("future_x_cols", ()) or ()),
        static_cols=tuple(long_df.attrs.get("static_cols", ()) or ()),
    )
    unique_ids = tuple(str(uid) for uid in long_df["unique_id"].astype("string").unique().tolist())

    payloads: dict[str, AdapterSeriesPayload] = {}
    if long_df.empty:
        freq: str | dict[str, str] = "D"
    elif len(unique_ids) == 1:
        uid_series = long_df["unique_id"].astype("string")
        unique_id = unique_ids[0]
        group = long_df.loc[uid_series == str(unique_id)].reset_index(drop=True)
        group_freq = infer_adapter_frequency(pd.Series(group["ds"]).reset_index(drop=True))
        payloads[str(unique_id)] = AdapterSeriesPayload(
            target=group.loc[:, ["ds", "y"]].copy(),
            historic_covariates=(
                None
                if not covariates.historic_x_cols
                else group.loc[:, ["ds", *covariates.historic_x_cols]].copy()
            ),
            future_covariates=(
                None
                if not covariates.future_x_cols
                else group.loc[:, ["ds", *covariates.future_x_cols]].copy()
            ),
            static_covariates=_build_static_covariates_frame(
                group,
                static_cols=covariates.static_cols,
                unique_id=str(unique_id),
            ),
            freq=group_freq,
        )
        freq = group_freq
    else:
        freq = {}
        uid_series = long_df["unique_id"].astype("string")
        for unique_id in unique_ids:
            group = long_df.loc[uid_series == str(unique_id)].reset_index(drop=True)
            group_freq = infer_adapter_frequency(pd.Series(group["ds"]).reset_index(drop=True))
            freq[str(unique_id)] = group_freq
            payloads[str(unique_id)] = AdapterSeriesPayload(
                target=group.loc[:, ["ds", "y"]].copy(),
                historic_covariates=(
                    None
                    if not covariates.historic_x_cols
                    else group.loc[:, ["ds", *covariates.historic_x_cols]].copy()
                ),
                future_covariates=(
                    None
                    if not covariates.future_x_cols
                    else group.loc[:, ["ds", *covariates.future_x_cols]].copy()
                ),
                static_covariates=_build_static_covariates_frame(
                    group,
                    static_cols=covariates.static_cols,
                    unique_id=str(unique_id),
                ),
                freq=group_freq,
            )

    return AdapterFrameBundle(
        covariates=covariates,
        freq=freq,
        unique_ids=unique_ids,
        payloads=payloads,
    )


def to_beta_bundle(data: Any) -> dict[str, Any]:
    bundle = require_adapter_frame_bundle(data)

    target: dict[str, pd.DataFrame] = {}
    historic_covariates: dict[str, pd.DataFrame] = {}
    future_covariates: dict[str, pd.DataFrame] = {}
    static_covariates: dict[str, pd.DataFrame] = {}

    for unique_id in bundle.unique_ids:
        payload = bundle.payloads[str(unique_id)]
        target[str(unique_id)] = payload.target.copy()
        if payload.historic_covariates is not None:
            historic_covariates[str(unique_id)] = payload.historic_covariates.copy()
        if payload.future_covariates is not None:
            future_covariates[str(unique_id)] = payload.future_covariates.copy()
        if payload.static_covariates is not None:
            static_covariates[str(unique_id)] = payload.static_covariates.copy()

    return {
        "target": target,
        "historic_covariates": historic_covariates,
        "future_covariates": future_covariates,
        "static_covariates": static_covariates,
        "freq": _normalize_bundle_freq(bundle),
    }


def from_beta_bundle(data: Any) -> pd.DataFrame:
    if not isinstance(data, dict):
        raise TypeError("shared beta bundle conversion expects a dict-like bundle")
    if "target" not in data or data.get("target") is None:
        raise ValueError("shared beta bundle must include a non-empty 'target' payload")

    target = dict(data.get("target", {}))
    historic_covariates = {
        str(unique_id): value
        for unique_id, value in dict(data.get("historic_covariates", {})).items()
    }
    future_covariates = {
        str(unique_id): value
        for unique_id, value in dict(data.get("future_covariates", {})).items()
    }
    static_covariates = {
        str(unique_id): value
        for unique_id, value in dict(data.get("static_covariates", {})).items()
    }

    frames: list[pd.DataFrame] = []
    historic_cols: list[str] = []
    future_cols: list[str] = []
    static_cols: list[str] = []

    for unique_id, target_frame in sorted(target.items(), key=lambda item: str(item[0])):
        if not isinstance(target_frame, pd.DataFrame):
            raise TypeError("shared beta bundle target payloads must be pandas DataFrames")
        if list(target_frame.columns) != ["ds", "y"]:
            raise ValueError("shared beta bundle target payloads must use columns ['ds', 'y']")
        frame = target_frame.copy()
        frame.insert(0, "unique_id", str(unique_id))

        historic_frame = historic_covariates.get(str(unique_id))
        if historic_frame is not None:
            if not isinstance(historic_frame, pd.DataFrame):
                raise TypeError(
                    "shared beta bundle historic_covariates payloads must be DataFrames"
                )
            historic_cols.extend([str(col) for col in historic_frame.columns if str(col) != "ds"])
            frame = frame.merge(historic_frame.copy(), on="ds", how="left")

        future_frame = future_covariates.get(str(unique_id))
        if future_frame is not None:
            if not isinstance(future_frame, pd.DataFrame):
                raise TypeError("shared beta bundle future_covariates payloads must be DataFrames")
            future_cols.extend([str(col) for col in future_frame.columns if str(col) != "ds"])
            frame = frame.merge(future_frame.copy(), on="ds", how="left")

        static_frame = static_covariates.get(str(unique_id))
        if static_frame is not None:
            if not isinstance(static_frame, pd.DataFrame):
                raise TypeError("shared beta bundle static_covariates payloads must be DataFrames")
            for col in static_frame.columns:
                frame[str(col)] = static_frame.iloc[0][col]
                static_cols.append(str(col))

        frames.append(frame)

    out = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    out = out.loc[
        :,
        [
            "unique_id",
            "ds",
            "y",
            *[col for col in out.columns if col not in {"unique_id", "ds", "y"}],
        ],
    ]
    out.attrs["historic_x_cols"] = tuple(dict.fromkeys(historic_cols))
    out.attrs["future_x_cols"] = tuple(dict.fromkeys(future_cols))
    out.attrs["static_cols"] = tuple(dict.fromkeys(static_cols))
    return out
