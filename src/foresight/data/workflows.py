from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.tseries.frequencies import to_offset

from ..features.lag import build_seasonal_lag_features, make_lagged_xy, make_lagged_xy_multi
from ..features.tabular import (
    build_lag_derived_features,
    normalize_int_tuple,
    normalize_lag_steps,
    normalize_str_tuple,
)
from ..features.time import build_fourier_features, build_time_features
from .prep import infer_series_frequency

_ALLOWED_ALIGN_AGG = frozenset({"first", "last", "max", "mean", "min", "sum"})
_ALLOWED_OUTLIER_METHODS = frozenset({"iqr", "zscore"})


def _coerce_long_df(long_df: Any, *, require_non_empty: bool = True) -> pd.DataFrame:
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df must be a pandas DataFrame")
    required = {"unique_id", "ds", "y"}
    missing = required.difference(long_df.columns)
    if missing:
        raise KeyError(f"long_df missing required columns: {sorted(missing)}")
    if require_non_empty and long_df.empty:
        raise ValueError("long_df is empty")

    out = long_df.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="raise")
    return out


def _normalize_freq(freq: str | None) -> str | None:
    if freq is None:
        return None
    return str(to_offset(str(freq)).freqstr)


def _normalize_align_agg(agg: str) -> str:
    normalized = str(agg).strip().lower()
    if normalized not in _ALLOWED_ALIGN_AGG:
        raise ValueError(f"agg must be one of: {sorted(_ALLOWED_ALIGN_AGG)}")
    return normalized


def _resolve_numeric_columns(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...] | None,
    default_columns: tuple[str, ...],
) -> tuple[str, ...]:
    if columns is None:
        selected = default_columns
    else:
        selected = tuple(str(col).strip() for col in columns if str(col).strip())
    if not selected:
        raise ValueError("columns must be non-empty")
    missing = [col for col in selected if col not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    non_numeric = [col for col in selected if not is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"columns must be numeric: {non_numeric}")
    return selected


def _aggregate_group_rows(group: pd.DataFrame, *, value_cols: tuple[str, ...], agg: str) -> pd.DataFrame:
    ordered = group.loc[:, ["ds", *value_cols]].sort_values("ds", kind="mergesort")
    aggregated = ordered.groupby("ds", sort=True, as_index=False).agg(agg)
    return aggregated.loc[:, ["ds", *value_cols]]


def _group_alignment_frequency(
    group: pd.DataFrame,
    *,
    freq: str | None,
    strict_freq: bool,
) -> str | None:
    if freq is not None:
        return freq
    try:
        return infer_series_frequency(group["ds"], strict=bool(strict_freq))
    except ValueError:
        if strict_freq:
            raise
        return None


def align_long_df(
    long_df: Any,
    *,
    freq: str | None = None,
    agg: str = "last",
    columns: tuple[str, ...] | None = None,
    strict_freq: bool = False,
) -> pd.DataFrame:
    """
    Align each series in a long-format DataFrame to a regular frequency.

    The helper aggregates duplicate timestamps within each series, then optionally resamples
    each series to the requested or inferred frequency.
    """
    df = _coerce_long_df(long_df, require_non_empty=True)
    agg_name = _normalize_align_agg(agg)
    freq_normalized = _normalize_freq(freq)
    default_cols = tuple(col for col in df.columns if col not in {"unique_id", "ds"})
    value_cols = _resolve_numeric_columns(df, columns=columns, default_columns=default_cols)

    frames: list[pd.DataFrame] = []
    for unique_id, group in df.groupby("unique_id", sort=False):
        aggregated = _aggregate_group_rows(group, value_cols=value_cols, agg=agg_name)
        group_freq = _group_alignment_frequency(
            aggregated,
            freq=freq_normalized,
            strict_freq=bool(strict_freq),
        )
        if group_freq is None:
            aligned = aggregated.copy()
        else:
            aligned = (
                aggregated.set_index("ds")
                .resample(group_freq)
                .agg(agg_name)
                .reset_index()
            )
        aligned["unique_id"] = unique_id
        frames.append(aligned.loc[:, ["unique_id", "ds", *value_cols]])

    return (
        pd.concat(frames, axis=0, ignore_index=True, sort=False)
        .sort_values(["unique_id", "ds"], kind="mergesort")
        .reset_index(drop=True)
    )


def _normalize_outlier_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in _ALLOWED_OUTLIER_METHODS:
        raise ValueError(f"method must be one of: {sorted(_ALLOWED_OUTLIER_METHODS)}")
    return normalized


def _clip_iqr(series: pd.Series, *, iqr_k: float) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - float(iqr_k) * iqr
    upper = q3 + float(iqr_k) * iqr
    return series.clip(lower=lower, upper=upper)


def _clip_zscore(series: pd.Series, *, zmax: float) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series
    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    if std <= 0.0 or not np.isfinite(std):
        return series
    lower = mean - float(zmax) * std
    upper = mean + float(zmax) * std
    return series.clip(lower=lower, upper=upper)


def clip_long_df_outliers(
    long_df: Any,
    *,
    method: str = "iqr",
    columns: tuple[str, ...] = ("y",),
    iqr_k: float = 1.5,
    zmax: float = 3.0,
) -> pd.DataFrame:
    """
    Clip numeric outliers independently within each series of a long-format DataFrame.
    """
    df = _coerce_long_df(long_df, require_non_empty=True)
    method_name = _normalize_outlier_method(method)
    if float(iqr_k) <= 0.0:
        raise ValueError("iqr_k must be > 0")
    if float(zmax) <= 0.0:
        raise ValueError("zmax must be > 0")
    value_cols = _resolve_numeric_columns(df, columns=columns, default_columns=("y",))

    out = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    for _unique_id, idx in out.groupby("unique_id", sort=False).groups.items():
        for col in value_cols:
            series = out.loc[idx, col]
            if method_name == "iqr":
                clipped = _clip_iqr(series, iqr_k=float(iqr_k))
            else:
                clipped = _clip_zscore(series, zmax=float(zmax))
            out.loc[idx, col] = clipped.to_numpy()
    return out


def enrich_long_df_calendar(
    long_df: Any,
    *,
    prefix: str = "cal_",
    add_time_idx: bool = True,
    add_dow: bool = True,
    add_month: bool = True,
    add_doy: bool = True,
    add_hour: bool = True,
) -> pd.DataFrame:
    """
    Append deterministic calendar features to a long-format DataFrame.

    Calendar features are generated per series so `time_idx` resets for each `unique_id`.
    """
    df = _coerce_long_df(long_df, require_non_empty=True)
    out = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    feature_frames: list[pd.DataFrame] = []
    prefixed_names: list[str] | None = None
    for _unique_id, group in out.groupby("unique_id", sort=False):
        feature_matrix, feature_names = build_time_features(
            group["ds"],
            add_time_idx=bool(add_time_idx),
            add_dow=bool(add_dow),
            add_month=bool(add_month),
            add_doy=bool(add_doy),
            add_hour=bool(add_hour),
        )
        names = [f"{prefix}{name}" for name in feature_names]
        if prefixed_names is None:
            prefixed_names = names
            collisions = [name for name in names if name in out.columns]
            if collisions:
                raise ValueError(f"calendar feature columns already exist: {collisions}")
        feature_frames.append(pd.DataFrame(feature_matrix, columns=names, index=group.index))

    if not feature_frames:
        return out
    features = pd.concat(feature_frames, axis=0).sort_index()
    return pd.concat([out, features], axis=1)


def _normalize_supervised_input_format(input_format: str) -> str:
    normalized = str(input_format).strip().lower()
    if normalized not in {"auto", "long", "wide"}:
        raise ValueError("input_format must be one of: auto, long, wide")
    return normalized


def _coerce_supervised_long_df(
    data: Any,
    *,
    input_format: str,
    ds_col: str,
    target_cols: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data is empty")

    if input_format == "auto":
        is_long = {"unique_id", "ds", "y"}.issubset(data.columns)
        input_format = "long" if is_long else "wide"

    if input_format == "long":
        return _coerce_long_df(data, require_non_empty=True)

    ds_name = str(ds_col).strip()
    if not ds_name:
        raise ValueError("ds_col must be non-empty")
    if ds_name not in data.columns:
        raise KeyError(f"ds_col not found: {ds_name!r}")

    targets = tuple(str(col).strip() for col in target_cols if str(col).strip())
    if not targets:
        targets = tuple(col for col in data.columns if col != ds_name)
    if not targets:
        raise ValueError("target_cols must be non-empty for wide input")
    missing = [col for col in targets if col not in data.columns]
    if missing:
        raise KeyError(f"target cols not found: {missing}")

    long_df = data.loc[:, [ds_name, *targets]].melt(
        id_vars=[ds_name],
        value_vars=list(targets),
        var_name="unique_id",
        value_name="y",
    )
    long_df = long_df.rename(columns={ds_name: "ds"})
    return _coerce_long_df(long_df, require_non_empty=True)


def _infer_supervised_x_cols(df: pd.DataFrame, x_cols: tuple[str, ...]) -> tuple[str, ...]:
    requested = normalize_str_tuple(x_cols)
    if requested:
        missing = [col for col in requested if col not in df.columns]
        if missing:
            raise KeyError(f"x_cols not found: {missing}")
        non_numeric = [col for col in requested if not is_numeric_dtype(df[col])]
        if non_numeric:
            raise ValueError(f"x_cols must be numeric: {non_numeric}")
        return requested

    inferred = tuple(
        col
        for col in df.columns
        if col not in {"unique_id", "ds", "y"} and is_numeric_dtype(df[col])
    )
    return inferred


def _compute_supervised_start_t(
    *,
    lags: Any,
    seasonal_lags: Any,
    seasonal_diff_lags: Any,
) -> tuple[int, tuple[int, ...]]:
    lag_steps = normalize_lag_steps(lags, allow_zero=False, name="lags")
    if not lag_steps:
        raise ValueError("lags must be >= 1")

    start_t = int(max(lag_steps))
    seasonal = normalize_int_tuple(seasonal_lags)
    seasonal_diffs = normalize_int_tuple(seasonal_diff_lags)
    if any(int(period) <= 0 for period in seasonal):
        raise ValueError("seasonal_lags must be >= 1")
    if any(int(period) <= 0 for period in seasonal_diffs):
        raise ValueError("seasonal_diff_lags must be >= 1")

    if seasonal:
        start_t = max(start_t, int(max(seasonal)))
    if seasonal_diffs:
        start_t = max(start_t, 1 + int(max(seasonal_diffs)))
    return start_t, lag_steps


def _group_target_frame(
    y: np.ndarray,
    *,
    lag_steps: tuple[int, ...],
    horizon: int,
    start_t: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(horizon) <= 0:
        raise ValueError("horizon must be >= 1")
    if int(horizon) == 1:
        X_base, y_target = make_lagged_xy(y, lags=lag_steps, start_t=start_t)
        t_index = np.arange(int(start_t), int(y.size), dtype=int)
        return X_base, y_target.reshape(-1, 1), t_index
    return make_lagged_xy_multi(y, lags=lag_steps, horizon=int(horizon), start_t=start_t)


def _prefixed_feature_frame(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    *,
    prefix: str = "feat_",
) -> pd.DataFrame:
    names = [f"{prefix}{name}" for name in feature_names]
    if len(names) != len(set(names)):
        raise ValueError("duplicate feature names generated for supervised frame")
    return pd.DataFrame(feature_matrix, columns=names)


def _base_lag_feature_frame(X_base: np.ndarray, lag_steps: tuple[int, ...]) -> pd.DataFrame:
    names = [f"feat_y_lag{int(lag)}" for lag in lag_steps]
    return pd.DataFrame(X_base, columns=names)


def _group_x_feature_frame(group: pd.DataFrame, *, t_index: np.ndarray, x_cols: tuple[str, ...]) -> pd.DataFrame:
    if not x_cols:
        return pd.DataFrame(index=range(int(t_index.size)))
    values = group.iloc[t_index].loc[:, list(x_cols)].to_numpy(dtype=float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("x_cols must contain finite values at supervised targets")
    names = [f"feat_x_{col}" for col in x_cols]
    return pd.DataFrame(values, columns=names)


def _group_time_feature_frame(group: pd.DataFrame, *, t_index: np.ndarray, add_time_features: bool) -> pd.DataFrame:
    if not add_time_features:
        return pd.DataFrame(index=range(int(t_index.size)))
    matrix, names = build_time_features(group.iloc[t_index]["ds"])
    return _prefixed_feature_frame(matrix, names)


def _group_supervised_frame(
    group: pd.DataFrame,
    *,
    lag_steps: tuple[int, ...],
    horizon: int,
    start_t: int,
    x_cols: tuple[str, ...],
    roll_windows: Any,
    roll_stats: Any,
    diff_lags: Any,
    seasonal_lags: Any,
    seasonal_diff_lags: Any,
    fourier_periods: Any,
    fourier_orders: Any,
    add_time_features: bool,
) -> pd.DataFrame:
    y = group["y"].to_numpy(dtype=float, copy=False)
    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values to build supervised frame")

    X_base, Y, t_index = _group_target_frame(
        y,
        lag_steps=lag_steps,
        horizon=int(horizon),
        start_t=int(start_t),
    )
    base_df = _base_lag_feature_frame(X_base, lag_steps)

    derived_matrix, derived_names = build_lag_derived_features(
        X_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
    )
    seasonal_matrix, seasonal_names = build_seasonal_lag_features(
        y,
        t=t_index,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
    )
    fourier_matrix, fourier_names = build_fourier_features(
        t_index,
        periods=fourier_periods,
        orders=fourier_orders,
    )

    feature_frames = [base_df, _group_x_feature_frame(group, t_index=t_index, x_cols=x_cols)]
    if derived_names:
        feature_frames.append(_prefixed_feature_frame(derived_matrix, derived_names))
    if seasonal_names:
        feature_frames.append(_prefixed_feature_frame(seasonal_matrix, seasonal_names))
    if fourier_names:
        feature_frames.append(_prefixed_feature_frame(fourier_matrix, fourier_names))
    time_df = _group_time_feature_frame(group, t_index=t_index, add_time_features=bool(add_time_features))
    if not time_df.empty:
        feature_frames.append(time_df)

    metadata = pd.DataFrame(
        {
            "unique_id": group["unique_id"].iloc[0],
            "ds": group.iloc[t_index]["ds"].to_numpy(copy=False),
            "target_t": t_index.astype(int, copy=False),
        }
    )
    target_names = ["y_target"] if int(horizon) == 1 else [f"y_t+{i}" for i in range(1, int(horizon) + 1)]
    target_df = pd.DataFrame(Y, columns=target_names)
    if not np.all(np.isfinite(target_df.to_numpy(dtype=float, copy=False))):
        raise ValueError("target window contains non-finite values")

    return pd.concat([metadata, *feature_frames, target_df], axis=1)


def make_supervised_frame(
    data: Any,
    *,
    input_format: str = "auto",
    ds_col: str = "ds",
    target_cols: tuple[str, ...] = (),
    lags: Any = 5,
    horizon: int = 1,
    x_cols: tuple[str, ...] = (),
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    add_time_features: bool = False,
) -> pd.DataFrame:
    """
    Build a supervised training table from long or wide time-series data.
    """
    format_name = _normalize_supervised_input_format(input_format)
    long_df = _coerce_supervised_long_df(
        data,
        input_format=format_name,
        ds_col=ds_col,
        target_cols=target_cols,
    )
    long_df = long_df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    selected_x_cols = _infer_supervised_x_cols(long_df, x_cols)
    start_t, lag_steps = _compute_supervised_start_t(
        lags=lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
    )

    frames: list[pd.DataFrame] = []
    for _unique_id, group in long_df.groupby("unique_id", sort=False):
        frames.append(
            _group_supervised_frame(
                group.reset_index(drop=True),
                lag_steps=lag_steps,
                horizon=int(horizon),
                start_t=int(start_t),
                x_cols=selected_x_cols,
                roll_windows=roll_windows,
                roll_stats=roll_stats,
                diff_lags=diff_lags,
                seasonal_lags=seasonal_lags,
                seasonal_diff_lags=seasonal_diff_lags,
                fourier_periods=fourier_periods,
                fourier_orders=fourier_orders,
                add_time_features=bool(add_time_features),
            )
        )

    return pd.concat(frames, axis=0, ignore_index=True, sort=False)
