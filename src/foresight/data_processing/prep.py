from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.tseries.frequencies import to_offset

from ..contracts.covariates import resolve_covariate_roles as _resolve_covariate_roles
from ..contracts.frames import coerce_sorted_long_df as _contracts_coerce_sorted_long_df
from ..contracts.frames import require_long_df as _contracts_require_long_df

_MISSING_POLICIES = {"error", "drop", "ffill", "zero", "interpolate"}


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
            raise ValueError(
                f"Missing values remain for unique_id={unique_id!r} in columns: {cols}"
            )
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


def _resolve_long_covariate_columns(
    df: pd.DataFrame,
    *,
    historic_x_cols: tuple[str, ...],
    future_x_cols: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    hist_input = historic_x_cols or tuple(df.attrs.get("historic_x_cols", ()))
    fut_input = future_x_cols or tuple(df.attrs.get("future_x_cols", ()))
    covariates = _resolve_covariate_roles(
        historic_x_cols=hist_input,
        future_x_cols=fut_input,
    )
    historic_cols = covariates.historic_x_cols
    future_cols = covariates.future_x_cols
    all_x_cols = covariates.all_x_cols
    if all_x_cols:
        return historic_cols, future_cols, all_x_cols

    inferred_future_cols = tuple(col for col in df.columns if col not in {"unique_id", "ds", "y"})
    return (), inferred_future_cols, inferred_future_cols


def _resolve_long_missing_policies(
    *,
    y_missing: str,
    x_missing: str,
    historic_x_missing: str | None,
    future_x_missing: str | None,
    historic_x_cols: tuple[str, ...],
    future_x_cols: tuple[str, ...],
) -> tuple[str, str, str, set[str]]:
    y_policy = _validate_missing_policy(y_missing, arg_name="y_missing")
    x_policy = _validate_missing_policy(x_missing, arg_name="x_missing")
    historic_policy = (
        x_policy
        if historic_x_missing is None
        else _validate_missing_policy(historic_x_missing, arg_name="historic_x_missing")
    )
    future_policy = (
        x_policy
        if future_x_missing is None
        else _validate_missing_policy(future_x_missing, arg_name="future_x_missing")
    )
    overlap = set(historic_x_cols).intersection(future_x_cols)
    if overlap and historic_policy != future_policy:
        raise ValueError("same covariate columns cannot use different missing policies")
    return y_policy, historic_policy, future_policy, overlap


def _frequency_by_unique_id(
    df: pd.DataFrame,
    *,
    freq: str | None,
    strict_freq: bool,
) -> dict[str, str | None]:
    if freq is not None:
        base_freq = str(to_offset(freq).freqstr)
        return {str(uid): base_freq for uid in df["unique_id"].astype(str).unique().tolist()}

    freq_by_uid: dict[str, str | None] = {}
    for uid, group in df.groupby("unique_id", sort=False):
        uid_s = str(uid)
        try:
            freq_by_uid[uid_s] = infer_series_frequency(group["ds"], strict=bool(strict_freq))
        except ValueError:
            if strict_freq:
                raise
            freq_by_uid[uid_s] = None

    inferred = {value for value in freq_by_uid.values() if value is not None}
    if strict_freq and len(inferred) > 1:
        raise ValueError(f"Mixed frequencies detected across series: {sorted(inferred)}")
    return freq_by_uid


def _prepare_long_group_frame(
    group: pd.DataFrame,
    *,
    uid_s: str,
    group_freq: str | None,
    y_policy: str,
    historic_x_cols: tuple[str, ...],
    future_x_cols: tuple[str, ...],
    overlap: set[str],
    historic_policy: str,
    future_policy: str,
    assume_sorted: bool,
) -> pd.DataFrame:
    out = (
        group.reset_index(drop=True)
        if assume_sorted
        else group.sort_values("ds", kind="mergesort").reset_index(drop=True)
    )
    if out["ds"].duplicated().any():
        raise ValueError(f"ds contains duplicates for unique_id={uid_s!r}")

    if group_freq is not None:
        full_ds = pd.date_range(start=out["ds"].iloc[0], end=out["ds"].iloc[-1], freq=group_freq)
        out = out.set_index("ds").reindex(full_ds).reset_index()
        out = out.rename(columns={"index": "ds"})
        out["unique_id"] = out["unique_id"].fillna(uid_s)

    overlap_tup = tuple(col for col in historic_x_cols if col in overlap)
    historic_only = tuple(col for col in historic_x_cols if col not in overlap)
    future_only = tuple(col for col in future_x_cols if col not in overlap)

    out = _apply_missing_policy(out, columns=("y",), policy=y_policy, unique_id=uid_s)
    out = _apply_missing_policy(
        out,
        columns=historic_only,
        policy=historic_policy,
        unique_id=uid_s,
    )
    out = _apply_missing_policy(
        out,
        columns=future_only,
        policy=future_policy,
        unique_id=uid_s,
    )
    return _apply_missing_policy(
        out,
        columns=overlap_tup,
        policy=historic_policy,
        unique_id=uid_s,
    )


def _coerce_wide_frame_ds_column(
    df: pd.DataFrame,
    *,
    ds_col: str | None,
) -> tuple[pd.DataFrame, str]:
    if ds_col is None:
        out_ds_col = "ds"
        if out_ds_col in df.columns:
            raise ValueError("wide_df already contains a 'ds' column; pass ds_col='ds' instead")
        if pd.Index(df.index).duplicated().any():
            raise ValueError("ds contains duplicates")
        idx_name = df.index.name or "index"
        return df.reset_index().rename(columns={idx_name: out_ds_col}), out_ds_col

    out_ds_col = str(ds_col).strip()
    if not out_ds_col:
        raise ValueError("ds_col must be non-empty")
    if out_ds_col not in df.columns:
        raise KeyError(f"ds_col not found: {out_ds_col!r}")
    return df, out_ds_col


def _resolve_wide_target_columns(
    df: pd.DataFrame,
    *,
    ds_col: str,
    target_cols: tuple[str, ...],
) -> tuple[str, ...]:
    if target_cols:
        targets = tuple(str(col).strip() for col in target_cols if str(col).strip())
        if not targets:
            raise ValueError("target_cols must be non-empty when provided")
    else:
        targets = tuple(col for col in df.columns if col != ds_col)

    if not targets:
        raise ValueError("No target columns found in wide_df")
    for col in targets:
        if col == ds_col:
            raise ValueError("target_cols cannot include ds_col")
        if col not in df.columns:
            raise KeyError(f"target col not found: {col!r}")
    return targets


def _wide_group_frequency(
    df: pd.DataFrame,
    *,
    ds_col: str,
    freq: str | None,
    strict_freq: bool,
) -> str | None:
    if freq is not None:
        return str(to_offset(freq).freqstr)

    try:
        return infer_series_frequency(df[ds_col], strict=bool(strict_freq))
    except ValueError:
        if strict_freq:
            raise
        return None


def _regularize_wide_frame(
    df: pd.DataFrame,
    *,
    ds_col: str,
    targets: tuple[str, ...],
    group_freq: str | None,
) -> pd.DataFrame:
    out = df.loc[:, [ds_col, *targets]].copy()
    out = out.sort_values(ds_col, kind="mergesort").reset_index(drop=True)
    if out[ds_col].isna().any():
        raise ValueError("ds contains NA values")
    if out[ds_col].duplicated().any():
        raise ValueError("ds contains duplicates")
    if group_freq is None:
        return out

    full_ds = pd.date_range(start=out[ds_col].iloc[0], end=out[ds_col].iloc[-1], freq=group_freq)
    out = out.set_index(ds_col).reindex(full_ds).reset_index()
    return out.rename(columns={"index": ds_col})


def prepare_long_df(
    long_df: Any,
    *,
    freq: str | None = None,
    strict_freq: bool = False,
    y_missing: str = "error",
    x_missing: str = "error",
    historic_x_cols: tuple[str, ...] = (),
    future_x_cols: tuple[str, ...] = (),
    historic_x_missing: str | None = None,
    future_x_missing: str | None = None,
) -> pd.DataFrame:
    """
    Regularize and validate a canonical long-format DataFrame.

    - Infers per-series frequency from ``ds`` when ``freq`` is not provided.
    - Optionally enforces a single shared frequency across all series.
    - Fills or rejects missing ``y``/covariate values with configurable policies.
    """
    df = _contracts_coerce_sorted_long_df(
        _contracts_require_long_df(long_df, require_non_empty=True),
        reset_index=True,
    )

    historic_x_cols_tup, future_x_cols_tup, all_x_cols = _resolve_long_covariate_columns(
        df,
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
    )
    y_policy, historic_policy, future_policy, overlap = _resolve_long_missing_policies(
        y_missing=y_missing,
        x_missing=x_missing,
        historic_x_missing=historic_x_missing,
        future_x_missing=future_x_missing,
        historic_x_cols=historic_x_cols_tup,
        future_x_cols=future_x_cols_tup,
    )
    freq_by_uid = _frequency_by_unique_id(df, freq=freq, strict_freq=strict_freq)

    frames: list[pd.DataFrame] = []
    for uid, group in df.groupby("unique_id", sort=False):
        uid_s = str(uid)
        frames.append(
            _prepare_long_group_frame(
                group,
                uid_s=uid_s,
                group_freq=freq_by_uid.get(uid_s),
                y_policy=y_policy,
                historic_x_cols=historic_x_cols_tup,
                future_x_cols=future_x_cols_tup,
                overlap=overlap,
                historic_policy=historic_policy,
                future_policy=future_policy,
                assume_sorted=True,
            )
        )

    prepared = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    prepared.attrs["historic_x_cols"] = historic_x_cols_tup
    prepared.attrs["future_x_cols"] = future_x_cols_tup
    cols = ["unique_id", "ds", "y", *all_x_cols]
    return prepared.loc[:, cols]


def prepare_wide_df(
    wide_df: Any,
    *,
    ds_col: str | None = "ds",
    freq: str | None = None,
    strict_freq: bool = False,
    missing: str = "error",
    target_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Regularize and validate a canonical wide-format DataFrame.

    Wide format is defined as:
      - one row per timestamp
      - one or more numeric target columns

    This helper mirrors `prepare_long_df` (panel) for multivariate / wide use-cases:
      - optional regular frequency insertion via `date_range` reindexing
      - configurable missing-value policy per target column

    Parameters:
      - ds_col: timestamp column name. If None, uses the DataFrame index and emits a
        new `ds` column in the output.
      - freq: optional pandas frequency string (e.g. "D"). If omitted, attempts to
        infer a regular frequency from ds; when inference fails and `strict_freq=False`,
        the function skips reindexing.
      - missing: missing value policy for target columns:
          error | drop | ffill | zero | interpolate
      - target_cols: explicit target columns; when empty, defaults to all columns
        except ds_col.
    """
    if not isinstance(wide_df, pd.DataFrame):
        raise TypeError("wide_df must be a pandas DataFrame")
    if wide_df.empty:
        raise ValueError("wide_df is empty")

    df, out_ds_col = _coerce_wide_frame_ds_column(wide_df.copy(), ds_col=ds_col)
    targets = _resolve_wide_target_columns(df, ds_col=out_ds_col, target_cols=target_cols)
    group_freq = _wide_group_frequency(
        df,
        ds_col=out_ds_col,
        freq=freq,
        strict_freq=strict_freq,
    )
    out = _regularize_wide_frame(df, ds_col=out_ds_col, targets=targets, group_freq=group_freq)

    out = _apply_missing_policy(out, columns=targets, policy=str(missing), unique_id="__wide__")
    return out.loc[:, [out_ds_col, *targets]]
