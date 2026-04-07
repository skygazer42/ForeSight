from __future__ import annotations

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
            raise ValueError(f"static_cols column {col!r} has no observed value for unique_id={unique_id!r}")
        unique_values = pd.unique(observed.to_numpy(copy=False))
        if len(unique_values) != 1:
            raise ValueError(f"static_cols column {col!r} must be constant within unique_id={unique_id!r}")
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
