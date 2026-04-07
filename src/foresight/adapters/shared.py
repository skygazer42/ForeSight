from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..contracts.covariates import CovariateSpec, resolve_covariate_roles
from ..contracts.frames import require_long_df


@dataclass(frozen=True)
class AdapterFrameBundle:
    long_df: pd.DataFrame
    covariates: CovariateSpec
    freq: str
    unique_ids: tuple[str, ...]


def infer_adapter_frequency(ds: pd.Series) -> str:
    values = pd.to_datetime(ds, errors="raise")
    if len(values) < 2:
        raise ValueError("adapter conversion requires at least 2 timestamps to infer freq")

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


def require_adapter_frame_bundle(data: Any) -> AdapterFrameBundle:
    long_df = require_long_df(data, require_non_empty=False).copy()
    long_df = long_df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    covariates = resolve_covariate_roles(
        historic_x_cols=tuple(long_df.attrs.get("historic_x_cols", ()) or ()),
        future_x_cols=tuple(long_df.attrs.get("future_x_cols", ()) or ()),
        static_cols=tuple(long_df.attrs.get("static_cols", ()) or ()),
    )
    unique_ids = tuple(str(uid) for uid in long_df["unique_id"].astype("string").unique().tolist())

    if long_df.empty:
        freq = "D"
    else:
        first_uid = unique_ids[0]
        first_group = long_df.loc[long_df["unique_id"].astype("string") == first_uid, "ds"]
        freq = infer_adapter_frequency(pd.Series(first_group).reset_index(drop=True))

    return AdapterFrameBundle(
        long_df=long_df,
        covariates=covariates,
        freq=freq,
        unique_ids=unique_ids,
    )
