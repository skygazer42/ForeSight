from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.tseries.frequencies import to_offset

_MISSING_POLICIES = {"error", "drop", "ffill", "zero", "interpolate"}


def _require_long_df(long_df: Any) -> pd.DataFrame:
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df must be a pandas DataFrame")
    required = {"unique_id", "ds", "y"}
    missing = required.difference(long_df.columns)
    if missing:
        raise KeyError(f"long_df missing required columns: {sorted(missing)}")
    if long_df.empty:
        raise ValueError("long_df is empty")
    return long_df


def _coerce_datetime_index(ds: Any) -> pd.DatetimeIndex:
    idx = pd.Index(ds)
    parsed = pd.to_datetime(idx, errors="coerce")
    dt_idx = pd.DatetimeIndex(parsed)
    if dt_idx.isna().any():
        raise ValueError("ds must contain only valid datetime-like values")
    return dt_idx.sort_values()


def infer_series_frequency(ds: Any, *, strict: bool = True) -> str | None:
    """
    Infer a regular datetime frequency from series timestamps.

    Returns a pandas-style frequency string such as ``"D"`` or ``"W-WED"``.
    """
    dt_idx = _coerce_datetime_index(ds)
    if len(dt_idx) < 2:
        raise ValueError("Need at least 2 timestamps to infer a frequency")
    if dt_idx.duplicated().any():
        raise ValueError("Cannot infer frequency with duplicate ds values")

    try:
        inferred = pd.infer_freq(dt_idx)
    except ValueError:
        inferred = None
    if inferred:
        return str(to_offset(inferred).freqstr)

    diffs = dt_idx.to_series(index=range(len(dt_idx))).diff().dropna()
    if not diffs.empty and diffs.nunique() == 1:
        return str(to_offset(diffs.iloc[0]).freqstr)

    if strict:
        raise ValueError("Could not infer a regular frequency from ds")
    return None


def _validate_missing_policy(policy: str, *, arg_name: str) -> str:
    p = str(policy).strip().lower()
    if p not in _MISSING_POLICIES:
        raise ValueError(f"{arg_name} must be one of: {sorted(_MISSING_POLICIES)}")
    return p


def _apply_missing_policy(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    policy: str,
    unique_id: str,
) -> pd.DataFrame:
    if not columns:
        return df

    p = _validate_missing_policy(policy, arg_name="missing policy")
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df

    out = df.copy()
    if p == "error":
        if out.loc[:, cols].isna().any().any():
            raise ValueError(f"Missing values remain for unique_id={unique_id!r} in columns: {cols}")
        return out

    if p == "drop":
        return out.dropna(subset=cols).reset_index(drop=True)

    if p == "zero":
        out.loc[:, cols] = out.loc[:, cols].fillna(0.0)
    elif p == "ffill":
        out.loc[:, cols] = out.loc[:, cols].ffill()
    elif p == "interpolate":
        for col in cols:
            series = out[col]
            if not is_numeric_dtype(series):
                out[col] = series.ffill()
            else:
                out[col] = series.interpolate(method="linear", limit_direction="both")

    if out.loc[:, cols].isna().any().any():
        raise ValueError(
            f"Policy {policy!r} could not fully resolve missing values for unique_id={unique_id!r}"
        )
    return out


def prepare_long_df(
    long_df: Any,
    *,
    freq: str | None = None,
    strict_freq: bool = False,
    y_missing: str = "error",
    x_missing: str = "error",
) -> pd.DataFrame:
    """
    Regularize and validate a canonical long-format DataFrame.

    - Infers per-series frequency from ``ds`` when ``freq`` is not provided.
    - Optionally enforces a single shared frequency across all series.
    - Fills or rejects missing ``y``/covariate values with configurable policies.
    """
    df = _require_long_df(long_df).copy()
    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    x_cols = tuple(c for c in df.columns if c not in {"unique_id", "ds", "y"})
    y_policy = _validate_missing_policy(y_missing, arg_name="y_missing")
    x_policy = _validate_missing_policy(x_missing, arg_name="x_missing")

    freq_by_uid: dict[str, str | None] = {}
    if freq is not None:
        base_freq = str(to_offset(freq).freqstr)
        for uid in df["unique_id"].astype(str).unique().tolist():
            freq_by_uid[str(uid)] = base_freq
    else:
        for uid, g in df.groupby("unique_id", sort=False):
            uid_s = str(uid)
            try:
                freq_by_uid[uid_s] = infer_series_frequency(g["ds"], strict=bool(strict_freq))
            except ValueError:
                if strict_freq:
                    raise
                freq_by_uid[uid_s] = None

        inferred = {v for v in freq_by_uid.values() if v is not None}
        if strict_freq and len(inferred) > 1:
            raise ValueError(f"Mixed frequencies detected across series: {sorted(inferred)}")

    frames: list[pd.DataFrame] = []
    for uid, g in df.groupby("unique_id", sort=False):
        uid_s = str(uid)
        g = g.copy().sort_values("ds", kind="mergesort").reset_index(drop=True)
        if g["ds"].duplicated().any():
            raise ValueError(f"ds contains duplicates for unique_id={uid_s!r}")

        group_freq = freq_by_uid.get(uid_s)
        if group_freq is not None:
            full_ds = pd.date_range(start=g["ds"].iloc[0], end=g["ds"].iloc[-1], freq=group_freq)
            out = g.set_index("ds").reindex(full_ds).reset_index()
            out = out.rename(columns={"index": "ds"})
            out["unique_id"] = out["unique_id"].fillna(uid_s)
        else:
            out = g

        out = _apply_missing_policy(out, columns=("y",), policy=y_policy, unique_id=uid_s)
        out = _apply_missing_policy(out, columns=x_cols, policy=x_policy, unique_id=uid_s)
        frames.append(out)

    prepared = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    prepared = prepared.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    cols = ["unique_id", "ds", "y", *x_cols]
    return prepared.loc[:, cols]
