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
_ALLOWED_SCALER_METHODS = frozenset({"maxabs", "minmax", "standard"})
_ALLOWED_SCALER_SCOPES = frozenset({"global", "per_series"})


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


def _validate_supervised_frame_output(frame: Any) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame returned by make_supervised_frame")
    required = {"unique_id", "ds", "target_t"}
    missing = required.difference(frame.columns)
    if missing:
        raise TypeError(f"frame missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("frame is empty")
    out = frame.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="raise")
    return out.sort_values(["unique_id", "ds", "target_t"], kind="mergesort").reset_index(drop=True)


def _supervised_target_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    target_names = tuple(
        str(col) for col in frame.columns if str(col) == "y_target" or str(col).startswith("y_t+")
    )
    if not target_names:
        raise TypeError("frame does not contain supervised target columns")
    return target_names


def make_supervised_arrays(
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
    dtype: Any = np.float64,
) -> dict[str, Any]:
    """
    Build dense supervised training arrays from the dataframe returned by make_supervised_frame.
    """
    format_name = _normalize_supervised_input_format(input_format)
    frame = _validate_supervised_frame_output(
        make_supervised_frame(
            data,
            input_format=format_name,
            ds_col=ds_col,
            target_cols=target_cols,
            lags=lags,
            horizon=int(horizon),
            x_cols=x_cols,
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
    target_names = _supervised_target_columns(frame)
    target_name_set = set(target_names)
    feature_names = tuple(
        str(col)
        for col in frame.columns
        if str(col) not in {"unique_id", "ds", "target_t"} and str(col) not in target_name_set
    )

    X = frame.loc[:, list(feature_names)].to_numpy(dtype=dtype, copy=False)
    if len(target_names) == 1:
        y: np.ndarray = frame[target_names[0]].to_numpy(dtype=dtype, copy=False)
    else:
        y = frame.loc[:, list(target_names)].to_numpy(dtype=dtype, copy=False)
    index = frame.loc[:, ["unique_id", "ds", "target_t"]].copy()
    metadata = {
        "input_format": format_name,
        "horizon": int(horizon),
        "n_series": int(index["unique_id"].nunique()),
        "n_rows": int(len(index)),
        "n_features": int(len(feature_names)),
        "n_targets": int(len(target_names)),
        "feature_names": feature_names,
        "target_names": target_names,
    }
    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "target_names": target_names,
        "index": index,
        "metadata": metadata,
    }


def _validate_supervised_arrays_bundle(
    bundle: Any,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], pd.DataFrame, dict[str, Any]]:
    if not isinstance(bundle, dict):
        raise TypeError("bundle must be a dict returned by make_supervised_arrays")
    required = {"X", "y", "feature_names", "target_names", "index", "metadata"}
    missing = required.difference(bundle)
    if missing:
        raise TypeError(f"bundle missing required keys: {sorted(missing)}")

    X = np.asarray(bundle["X"])
    y = np.asarray(bundle["y"])
    feature_names = tuple(str(name) for name in bundle["feature_names"])
    target_names = tuple(str(name) for name in bundle["target_names"])
    index = bundle["index"]
    metadata = bundle["metadata"]

    if not isinstance(index, pd.DataFrame):
        raise TypeError("bundle['index'] must be a pandas DataFrame")
    if not isinstance(metadata, dict):
        raise TypeError("bundle['metadata'] must be a dict")
    required_index = {"unique_id", "ds", "target_t"}
    missing_index = required_index.difference(index.columns)
    if missing_index:
        raise TypeError(f"bundle['index'] missing required columns: {sorted(missing_index)}")
    if X.ndim != 2:
        raise ValueError("bundle['X'] must be a 2D array")
    if y.ndim not in {1, 2}:
        raise ValueError("bundle['y'] must be a 1D or 2D array")
    if int(X.shape[0]) != int(len(index)) or int(y.shape[0]) != int(len(index)):
        raise ValueError("bundle X, y, and index must have the same number of rows")
    if int(X.shape[1]) != int(len(feature_names)):
        raise ValueError("bundle feature_names must match the X column dimension")
    if y.ndim == 1 and int(len(target_names)) != 1:
        raise ValueError("1D supervised y requires exactly one target name")
    if y.ndim == 2 and int(y.shape[1]) != int(len(target_names)):
        raise ValueError("bundle target_names must match the y column dimension")

    index_out = index.copy()
    index_out["ds"] = pd.to_datetime(index_out["ds"], errors="raise")
    order = index_out.sort_values(["unique_id", "ds", "target_t"], kind="mergesort").index.to_numpy()
    return (
        X[order].copy(),
        y[order].copy(),
        feature_names,
        target_names,
        index_out.iloc[order].reset_index(drop=True),
        dict(metadata),
    )


def _split_supervised_row_positions(
    index: pd.DataFrame,
    *,
    valid_size: int | None = None,
    test_size: int | None = None,
    valid_frac: float | None = None,
    test_frac: float | None = None,
    gap: int = 0,
    min_train_size: int = 1,
    error_prefix: str,
) -> dict[str, np.ndarray]:
    gap_int = int(gap)
    min_train_size_int = int(min_train_size)
    if gap_int < 0:
        raise ValueError("gap must be >= 0")
    if min_train_size_int <= 0:
        raise ValueError("min_train_size must be >= 1")

    positions: dict[str, list[np.ndarray]] = {"train": [], "valid": [], "test": []}
    for unique_id, group in index.groupby("unique_id", sort=False):
        group = group.reset_index()
        n_rows = int(len(group))
        valid_n = _resolved_partition_size(
            n_rows=n_rows,
            size=valid_size,
            frac=valid_frac,
            name="valid_size",
        )
        test_n = _resolved_partition_size(
            n_rows=n_rows,
            size=test_size,
            frac=test_frac,
            name="test_size",
        )
        if valid_n == 0 and test_n == 0:
            raise ValueError("at least one of valid/test size or frac must be positive")

        gap_before_valid = gap_int if valid_n > 0 else 0
        gap_before_test = gap_int if test_n > 0 else 0
        train_end = n_rows - valid_n - test_n - gap_before_valid - gap_before_test
        if train_end < min_train_size_int:
            raise ValueError(
                f"{error_prefix} leaves fewer than min_train_size={min_train_size_int} "
                f"rows for unique_id={unique_id!r}"
            )

        valid_start = train_end + gap_before_valid
        valid_end = valid_start + valid_n
        test_start = valid_end + gap_before_test
        if test_start + test_n > n_rows:
            raise ValueError(
                f"{error_prefix} consumes more rows than available for unique_id={unique_id!r}"
            )

        row_positions = group["index"].to_numpy(dtype=int, copy=False)
        positions["train"].append(row_positions[:train_end])
        positions["valid"].append(row_positions[valid_start:valid_end])
        positions["test"].append(row_positions[test_start : test_start + test_n])

    return {
        name: np.concatenate(chunks) if chunks else np.asarray([], dtype=int)
        for name, chunks in positions.items()
    }


def split_supervised_frame(
    frame: Any,
    *,
    valid_size: int | None = None,
    test_size: int | None = None,
    valid_frac: float | None = None,
    test_frac: float | None = None,
    gap: int = 0,
    min_train_size: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Chronologically split supervised training rows into train/valid/test partitions per series.
    """
    frame_df = _validate_supervised_frame_output(frame)
    positions = _split_supervised_row_positions(
        frame_df.loc[:, ["unique_id", "ds", "target_t"]],
        valid_size=valid_size,
        test_size=test_size,
        valid_frac=valid_frac,
        test_frac=test_frac,
        gap=int(gap),
        min_train_size=int(min_train_size),
        error_prefix="supervised frame split",
    )
    return {
        name: frame_df.iloc[take].reset_index(drop=True)
        for name, take in positions.items()
    }


def split_supervised_arrays(
    bundle: Any,
    *,
    valid_size: int | None = None,
    test_size: int | None = None,
    valid_frac: float | None = None,
    test_frac: float | None = None,
    gap: int = 0,
    min_train_size: int = 1,
) -> dict[str, dict[str, Any]]:
    """
    Chronologically split supervised training arrays into train/valid/test partitions per series.
    """
    X, y, feature_names, target_names, index, metadata = _validate_supervised_arrays_bundle(bundle)
    positions = _split_supervised_row_positions(
        index,
        valid_size=valid_size,
        test_size=test_size,
        valid_frac=valid_frac,
        test_frac=test_frac,
        gap=int(gap),
        min_train_size=int(min_train_size),
        error_prefix="supervised array split",
    )

    parts: dict[str, dict[str, Any]] = {}
    for name, take in positions.items():
        part_index = index.iloc[take].reset_index(drop=True)
        part_metadata = dict(metadata)
        part_metadata["n_rows"] = int(len(part_index))
        part_metadata["partition"] = name
        parts[name] = {
            "X": X[take].copy(),
            "y": y[take].copy(),
            "feature_names": feature_names,
            "target_names": target_names,
            "index": part_index,
            "metadata": part_metadata,
        }
    return parts


def _resolve_panel_target_lags(*, lags: Any, target_lags: Any = ()) -> tuple[int, ...]:
    spec = target_lags
    if isinstance(spec, str) and not spec.strip():
        spec = ()
    if spec in (None, (), []):
        spec = lags
    return normalize_lag_steps(spec, allow_zero=False, name="target_lags")


def _resolve_panel_historic_x_lags(historic_x_lags: Any) -> tuple[int, ...]:
    spec = historic_x_lags
    if isinstance(spec, str) and not spec.strip():
        spec = ()
    if spec in (None, (), []):
        return ()
    return normalize_lag_steps(spec, allow_zero=False, name="historic_x_lags")


def _resolve_panel_future_x_lags(*, x_cols: tuple[str, ...], future_x_lags: Any) -> tuple[int, ...]:
    if not x_cols:
        return ()
    spec = future_x_lags
    if isinstance(spec, str) and not spec.strip():
        spec = ()
    if spec in (None, (), []):
        return (0,)
    return normalize_lag_steps(spec, allow_zero=True, name="future_x_lags")


def _resolve_panel_seasonal_lags(seasonal_lags: Any) -> tuple[int, ...]:
    spec = seasonal_lags
    if isinstance(spec, str) and not spec.strip():
        spec = ()
    if spec in (None, (), []):
        return ()
    return normalize_lag_steps(spec, allow_zero=False, name="seasonal_lags")


def _normalize_panel_x_cols(long_df: pd.DataFrame, x_cols: Any) -> tuple[str, ...]:
    cols = normalize_str_tuple(x_cols)
    if not cols:
        return ()
    missing = [col for col in cols if col not in long_df.columns]
    if missing:
        raise KeyError(f"x_cols not found: {missing}")
    reserved = [col for col in cols if col in {"unique_id", "ds", "y"}]
    if reserved:
        raise ValueError(f"x_cols cannot include reserved column names: {reserved}")
    non_numeric = [col for col in cols if not is_numeric_dtype(long_df[col])]
    if non_numeric:
        raise ValueError(f"x_cols must be numeric: {non_numeric}")
    return cols


def _panel_window_required_start(
    *,
    target_lags: tuple[int, ...],
    seasonal_lags: tuple[int, ...],
    historic_x_lags: tuple[int, ...],
    future_x_lags: tuple[int, ...],
) -> int:
    return max(
        [
            int(max(target_lags)),
            *([int(max(seasonal_lags))] if seasonal_lags else []),
            *([int(max(historic_x_lags))] if historic_x_lags else []),
            *([int(max(future_x_lags))] if future_x_lags else []),
        ]
    )


def _validate_panel_window_group(group: pd.DataFrame, *, x_cols: tuple[str, ...]) -> None:
    if group["ds"].duplicated().any():
        raise ValueError("duplicate timestamps found within a series; run align_long_df() first")
    y = group["y"].to_numpy(dtype=float, copy=False)
    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values to build panel windows")
    if not x_cols:
        return
    values = group.loc[:, list(x_cols)].to_numpy(dtype=float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("x_cols must contain only finite values to build panel windows")


def _empty_panel_window_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["unique_id", "cutoff_ds", "target_ds", "step", "y"])


def _group_panel_window_frame(
    group: pd.DataFrame,
    *,
    horizon: int,
    target_lags: tuple[int, ...],
    seasonal_lags: tuple[int, ...],
    historic_x_lags: tuple[int, ...],
    future_x_lags: tuple[int, ...],
    x_cols: tuple[str, ...],
    add_time_features: bool,
) -> pd.DataFrame:
    _validate_panel_window_group(group, x_cols=x_cols)

    y = group["y"].to_numpy(dtype=float, copy=False)
    ds_arr = group["ds"].to_numpy(copy=False)
    exog = None if not x_cols else group.loc[:, list(x_cols)].to_numpy(dtype=float, copy=False)
    time_matrix: np.ndarray | None = None
    time_names: list[str] = []
    if add_time_features:
        time_matrix, base_time_names = build_time_features(group["ds"])
        time_names = [f"time_{name}" for name in base_time_names]

    required_start = _panel_window_required_start(
        target_lags=target_lags,
        seasonal_lags=seasonal_lags,
        historic_x_lags=historic_x_lags,
        future_x_lags=future_x_lags,
    )
    rows: list[dict[str, Any]] = []
    horizon_int = int(horizon)

    for t in range(int(required_start), int(len(group)) - horizon_int + 1):
        cutoff_idx = int(t) - 1
        base_row: dict[str, Any] = {}
        for lag in target_lags:
            base_row[f"y_lag_{int(lag)}"] = float(y[int(t) - int(lag)])
        for lag in seasonal_lags:
            base_row[f"y_seasonal_lag_{int(lag)}"] = float(y[int(t) - int(lag)])
        if exog is not None:
            for lag in historic_x_lags:
                source_idx = int(t) - int(lag)
                for col_idx, col_name in enumerate(x_cols):
                    base_row[f"historic_x__{col_name}_lag_{int(lag)}"] = float(exog[source_idx, col_idx])

        for step_idx in range(horizon_int):
            target_idx = int(t) + int(step_idx)
            row = {
                "unique_id": group["unique_id"].iloc[0],
                "cutoff_ds": ds_arr[cutoff_idx],
                "target_ds": ds_arr[target_idx],
                "step": int(step_idx) + 1,
                "y": float(y[target_idx]),
                **base_row,
            }
            if exog is not None:
                for lag in future_x_lags:
                    source_idx = target_idx - int(lag)
                    for col_idx, col_name in enumerate(x_cols):
                        row[f"future_x__{col_name}_lag_{int(lag)}"] = float(exog[source_idx, col_idx])
            if time_matrix is not None:
                for col_idx, col_name in enumerate(time_names):
                    row[col_name] = float(time_matrix[target_idx, col_idx])
            rows.append(row)

    if not rows:
        return _empty_panel_window_frame()
    return pd.DataFrame(rows)


def make_panel_window_frame(
    long_df: Any,
    *,
    horizon: int = 1,
    lags: Any = 24,
    target_lags: Any = (),
    seasonal_lags: Any = (),
    historic_x_lags: Any = (),
    future_x_lags: Any = (),
    x_cols: tuple[str, ...] = (),
    add_time_features: bool = False,
) -> pd.DataFrame:
    """
    Build a panel training table where each row represents one target step from one sliding window.
    """
    df = _coerce_long_df(long_df, require_non_empty=True)
    horizon_int = int(horizon)
    if horizon_int <= 0:
        raise ValueError("horizon must be >= 1")

    x_cols_tup = _normalize_panel_x_cols(df, x_cols)
    target_lag_steps = _resolve_panel_target_lags(lags=lags, target_lags=target_lags)
    seasonal_lag_steps = _resolve_panel_seasonal_lags(seasonal_lags)
    historic_lag_steps = _resolve_panel_historic_x_lags(historic_x_lags)
    future_lag_steps = _resolve_panel_future_x_lags(x_cols=x_cols_tup, future_x_lags=future_x_lags)

    out = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    frames: list[pd.DataFrame] = []
    for _unique_id, group in out.groupby("unique_id", sort=False):
        frame = _group_panel_window_frame(
            group.reset_index(drop=True),
            horizon=horizon_int,
            target_lags=target_lag_steps,
            seasonal_lags=seasonal_lag_steps,
            historic_x_lags=historic_lag_steps,
            future_x_lags=future_lag_steps,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError("No series had enough history to build panel windows")
    return pd.concat(frames, axis=0, ignore_index=True, sort=False)


def make_panel_window_arrays(
    long_df: Any,
    *,
    horizon: int = 1,
    lags: Any = 24,
    target_lags: Any = (),
    seasonal_lags: Any = (),
    historic_x_lags: Any = (),
    future_x_lags: Any = (),
    x_cols: tuple[str, ...] = (),
    add_time_features: bool = False,
    dtype: Any = np.float64,
) -> dict[str, Any]:
    """
    Build dense training arrays from panel sliding windows plus feature/index metadata.
    """
    frame = make_panel_window_frame(
        long_df,
        horizon=int(horizon),
        lags=lags,
        target_lags=target_lags,
        seasonal_lags=seasonal_lags,
        historic_x_lags=historic_x_lags,
        future_x_lags=future_x_lags,
        x_cols=x_cols,
        add_time_features=bool(add_time_features),
    )

    feature_names = tuple(str(col) for col in frame.columns[5:])
    index_cols = ["unique_id", "cutoff_ds", "target_ds", "step"]
    target_lag_steps = _resolve_panel_target_lags(lags=lags, target_lags=target_lags)
    seasonal_lag_steps = _resolve_panel_seasonal_lags(seasonal_lags)
    x_cols_tup = _normalize_panel_x_cols(_coerce_long_df(long_df, require_non_empty=True), x_cols)
    historic_lag_steps = _resolve_panel_historic_x_lags(historic_x_lags)
    future_lag_steps = _resolve_panel_future_x_lags(x_cols=x_cols_tup, future_x_lags=future_x_lags)

    X = frame.loc[:, list(feature_names)].to_numpy(dtype=dtype, copy=False)
    y = frame["y"].to_numpy(dtype=dtype, copy=False)
    index = frame.loc[:, index_cols].copy()
    window_index = index.loc[:, ["unique_id", "cutoff_ds"]].drop_duplicates()
    metadata = {
        "horizon": int(horizon),
        "target_lags": target_lag_steps,
        "seasonal_lags": seasonal_lag_steps,
        "historic_x_lags": historic_lag_steps,
        "future_x_lags": future_lag_steps,
        "x_cols": x_cols_tup,
        "add_time_features": bool(add_time_features),
        "n_series": int(index["unique_id"].nunique()),
        "n_windows": int(len(window_index)),
        "n_rows": int(len(frame)),
        "n_features": int(len(feature_names)),
    }
    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "index": index,
        "metadata": metadata,
    }


def _validate_panel_window_frame_for_split(frame: Any) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame returned by make_panel_window_frame")
    required = {"unique_id", "cutoff_ds", "target_ds", "step", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise TypeError(f"frame missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("frame is empty")
    out = frame.copy()
    out["cutoff_ds"] = pd.to_datetime(out["cutoff_ds"], errors="raise")
    out["target_ds"] = pd.to_datetime(out["target_ds"], errors="raise")
    return out.sort_values(["unique_id", "cutoff_ds", "target_ds", "step"], kind="mergesort").reset_index(drop=True)


def _panel_window_origin_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.loc[:, ["unique_id", "cutoff_ds"]]
        .drop_duplicates()
        .sort_values(["unique_id", "cutoff_ds"], kind="mergesort")
        .reset_index(drop=True)
    )


def _split_panel_window_origins(
    window_index: pd.DataFrame,
    *,
    valid_size: int | None = None,
    test_size: int | None = None,
    valid_frac: float | None = None,
    test_frac: float | None = None,
    gap: int = 0,
    min_train_size: int = 1,
) -> dict[str, pd.DataFrame]:
    gap_int = int(gap)
    min_train_size_int = int(min_train_size)
    if gap_int < 0:
        raise ValueError("gap must be >= 0")
    if min_train_size_int <= 0:
        raise ValueError("min_train_size must be >= 1")

    origins = window_index.copy()
    origins["cutoff_ds"] = pd.to_datetime(origins["cutoff_ds"], errors="raise")
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}

    for unique_id, group in origins.groupby("unique_id", sort=False):
        group = group.sort_values("cutoff_ds", kind="mergesort").reset_index(drop=True)
        n_rows = int(len(group))
        valid_n = _resolved_partition_size(
            n_rows=n_rows,
            size=valid_size,
            frac=valid_frac,
            name="valid_size",
        )
        test_n = _resolved_partition_size(
            n_rows=n_rows,
            size=test_size,
            frac=test_frac,
            name="test_size",
        )
        if valid_n == 0 and test_n == 0:
            raise ValueError("at least one of valid/test size or frac must be positive")

        gap_before_valid = gap_int if valid_n > 0 else 0
        gap_before_test = gap_int if test_n > 0 else 0
        train_end = n_rows - valid_n - test_n - gap_before_valid - gap_before_test
        if train_end < min_train_size_int:
            raise ValueError(
                f"panel window split leaves fewer than min_train_size={min_train_size_int} "
                f"windows for unique_id={unique_id!r}"
            )

        valid_start = train_end + gap_before_valid
        valid_end = valid_start + valid_n
        test_start = valid_end + gap_before_test
        if test_start + test_n > n_rows:
            raise ValueError(
                f"panel window split consumes more windows than available for unique_id={unique_id!r}"
            )

        parts["train"].append(group.iloc[:train_end].copy())
        parts["valid"].append(group.iloc[valid_start:valid_end].copy())
        parts["test"].append(group.iloc[test_start : test_start + test_n].copy())

    result: dict[str, pd.DataFrame] = {}
    for name, frames in parts.items():
        non_empty = [frame for frame in frames if not frame.empty]
        if non_empty:
            result[name] = pd.concat(non_empty, axis=0, ignore_index=True, sort=False)
        else:
            result[name] = origins.iloc[0:0].copy()
    return result


def _panel_window_origin_mask(rows: pd.DataFrame, window_index: pd.DataFrame) -> np.ndarray:
    row_keys = pd.MultiIndex.from_frame(rows.loc[:, ["unique_id", "cutoff_ds"]])
    origin_keys = pd.MultiIndex.from_frame(window_index.loc[:, ["unique_id", "cutoff_ds"]])
    return np.asarray(row_keys.isin(origin_keys), dtype=bool)


def split_panel_window_frame(
    frame: Any,
    *,
    valid_size: int | None = None,
    test_size: int | None = None,
    valid_frac: float | None = None,
    test_frac: float | None = None,
    gap: int = 0,
    min_train_size: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Chronologically split panel-window training rows into train/valid/test partitions by window origin.
    """
    frame_df = _validate_panel_window_frame_for_split(frame)
    origin_parts = _split_panel_window_origins(
        _panel_window_origin_frame(frame_df),
        valid_size=valid_size,
        test_size=test_size,
        valid_frac=valid_frac,
        test_frac=test_frac,
        gap=int(gap),
        min_train_size=int(min_train_size),
    )
    return {
        name: frame_df.loc[_panel_window_origin_mask(frame_df, origins)].reset_index(drop=True)
        for name, origins in origin_parts.items()
    }


def _validate_panel_window_arrays_bundle(bundle: Any) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], pd.DataFrame, dict[str, Any]]:
    if not isinstance(bundle, dict):
        raise TypeError("bundle must be a dict returned by make_panel_window_arrays")
    required = {"X", "y", "feature_names", "index", "metadata"}
    missing = required.difference(bundle)
    if missing:
        raise TypeError(f"bundle missing required keys: {sorted(missing)}")

    X = np.asarray(bundle["X"])
    y = np.asarray(bundle["y"])
    feature_names = tuple(str(name) for name in bundle["feature_names"])
    index = bundle["index"]
    metadata = bundle["metadata"]

    if not isinstance(index, pd.DataFrame):
        raise TypeError("bundle['index'] must be a pandas DataFrame")
    if not isinstance(metadata, dict):
        raise TypeError("bundle['metadata'] must be a dict")
    required_index = {"unique_id", "cutoff_ds", "target_ds", "step"}
    missing_index = required_index.difference(index.columns)
    if missing_index:
        raise TypeError(f"bundle['index'] missing required columns: {sorted(missing_index)}")
    if X.ndim != 2:
        raise ValueError("bundle['X'] must be a 2D array")
    if y.ndim != 1:
        raise ValueError("bundle['y'] must be a 1D array")
    if int(X.shape[0]) != int(y.shape[0]) or int(X.shape[0]) != int(len(index)):
        raise ValueError("bundle X, y, and index must have the same number of rows")
    if int(X.shape[1]) != int(len(feature_names)):
        raise ValueError("bundle feature_names must match the X column dimension")

    index_out = index.copy()
    index_out["cutoff_ds"] = pd.to_datetime(index_out["cutoff_ds"], errors="raise")
    index_out["target_ds"] = pd.to_datetime(index_out["target_ds"], errors="raise")
    return X, y, feature_names, index_out, dict(metadata)


def split_panel_window_arrays(
    bundle: Any,
    *,
    valid_size: int | None = None,
    test_size: int | None = None,
    valid_frac: float | None = None,
    test_frac: float | None = None,
    gap: int = 0,
    min_train_size: int = 1,
) -> dict[str, dict[str, Any]]:
    """
    Chronologically split panel-window array bundles into train/valid/test partitions by window origin.
    """
    X, y, feature_names, index, metadata = _validate_panel_window_arrays_bundle(bundle)
    origin_parts = _split_panel_window_origins(
        _panel_window_origin_frame(index),
        valid_size=valid_size,
        test_size=test_size,
        valid_frac=valid_frac,
        test_frac=test_frac,
        gap=int(gap),
        min_train_size=int(min_train_size),
    )

    parts: dict[str, dict[str, Any]] = {}
    for name, origins in origin_parts.items():
        mask = _panel_window_origin_mask(index, origins)
        part_index = index.loc[mask].reset_index(drop=True)
        part_metadata = dict(metadata)
        part_metadata["n_windows"] = int(len(origins))
        part_metadata["n_rows"] = int(len(part_index))
        part_metadata["partition"] = name
        parts[name] = {
            "X": X[mask].copy(),
            "y": y[mask].copy(),
            "feature_names": feature_names,
            "index": part_index,
            "metadata": part_metadata,
        }
    return parts


def _normalize_panel_sequence_dtype(dtype: Any) -> np.dtype:
    return np.dtype(dtype)


def _normalize_panel_sequence_max_train_size(max_train_size: int | None) -> int | None:
    if max_train_size is None:
        return None
    value = int(max_train_size)
    if value <= 0:
        raise ValueError("max_train_size must be >= 1 or None")
    return value


def _normalize_panel_sequence_context_length(context_length: int) -> int:
    value = int(context_length)
    if value <= 0:
        raise ValueError("context_length must be >= 1")
    return value


def _normalize_panel_sequence_sample_step(sample_step: int) -> int:
    value = int(sample_step)
    if value <= 0:
        raise ValueError("sample_step must be >= 1")
    return value


def _normalize_panel_sequence_horizon(horizon: int) -> int:
    value = int(horizon)
    if value <= 0:
        raise ValueError("horizon must be >= 1")
    return value


def _panel_sequence_cutoff_index(ds_arr: np.ndarray, cutoff: Any) -> int | None:
    idx = pd.Index(ds_arr).get_indexer([cutoff])[0]
    if int(idx) < 0:
        return None
    return int(idx)


def _panel_sequence_normalize_target(
    y: np.ndarray,
    *,
    normalize: bool,
) -> tuple[np.ndarray, float, float]:
    if not bool(normalize):
        return np.asarray(y, dtype=float), 0.0, 1.0
    mean = float(np.mean(y))
    std = float(np.std(y))
    if std < 1e-8:
        std = 1.0
    return (np.asarray(y, dtype=float) - mean) / std, mean, std


def _panel_sequence_time_block(
    group: pd.DataFrame,
    *,
    add_time_features: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not add_time_features:
        return np.empty((len(group), 0), dtype=float), ()
    time_matrix, feature_names = build_time_features(group["ds"])
    return time_matrix.astype(float, copy=False), tuple(feature_names)


def _panel_sequence_train_window_rows(
    *,
    unique_id: Any,
    ds_arr: np.ndarray,
    y_train_scaled: np.ndarray,
    x_train_seg: np.ndarray,
    time_train_seg: np.ndarray,
    slice_start: int,
    series_code: int,
    context_length: int,
    horizon: int,
    sample_step: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    n_train = int(y_train_scaled.size)
    seq_len = int(context_length) + int(horizon)
    input_dim = 1 + int(x_train_seg.shape[1]) + int(time_train_seg.shape[1])
    n_windows = n_train - int(context_length) - int(horizon) + 1
    if n_windows <= 0:
        return (
            np.empty((0, seq_len, input_dim), dtype=dtype),
            np.empty((0, int(horizon)), dtype=dtype),
            np.empty((0,), dtype=int),
            pd.DataFrame(columns=["unique_id", "cutoff_ds", "target_start_ds", "target_end_ds"]),
        )

    win_indices = list(range(0, n_windows, int(sample_step)))
    n_samples = int(len(win_indices))
    X = np.empty((n_samples, seq_len, input_dim), dtype=dtype)
    Y = np.empty((n_samples, int(horizon)), dtype=dtype)
    series_id = np.full((n_samples,), int(series_code), dtype=int)
    rows: list[dict[str, Any]] = []

    for j, w0 in enumerate(win_indices):
        t = int(w0 + int(context_length))
        past = slice(int(t - int(context_length)), int(t))
        fut = slice(int(t), int(t + int(horizon)))

        y_feat = np.concatenate(
            [
                y_train_scaled[past],
                np.zeros((int(horizon),), dtype=float),
            ],
            axis=0,
        ).reshape(seq_len, 1)
        x_feat = np.concatenate([x_train_seg[past], x_train_seg[fut]], axis=0)
        time_feat = np.concatenate([time_train_seg[past], time_train_seg[fut]], axis=0)

        X[j] = np.concatenate([y_feat, x_feat, time_feat], axis=1).astype(dtype, copy=False)
        Y[j] = np.asarray(y_train_scaled[fut], dtype=dtype)

        cutoff_global = int(slice_start) + int(t) - 1
        target_start_global = int(slice_start) + int(t)
        target_end_global = target_start_global + int(horizon) - 1
        rows.append(
            {
                "unique_id": unique_id,
                "cutoff_ds": ds_arr[cutoff_global],
                "target_start_ds": ds_arr[target_start_global],
                "target_end_ds": ds_arr[target_end_global],
            }
        )

    return X, Y, series_id, pd.DataFrame(rows)


def _panel_sequence_predict_row(
    *,
    unique_id: Any,
    ds_arr: np.ndarray,
    y_arr: np.ndarray,
    x_full: np.ndarray,
    time_full: np.ndarray,
    train_end: int,
    context_length: int,
    horizon: int,
    mean: float,
    std: float,
    normalize: bool,
    series_code: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, int, dict[str, Any], float, float] | None:
    if int(train_end) + int(horizon) > int(y_arr.size):
        return None
    if int(train_end) < int(context_length):
        return None

    y_ctx = y_arr[int(train_end) - int(context_length) : int(train_end)]
    if bool(normalize):
        y_ctx = (y_ctx - float(mean)) / float(std)
    y_feat = np.concatenate(
        [
            np.asarray(y_ctx, dtype=float),
            np.zeros((int(horizon),), dtype=float),
        ],
        axis=0,
    ).reshape(int(context_length) + int(horizon), 1)

    x_ctx = x_full[int(train_end) - int(context_length) : int(train_end)]
    x_fut = x_full[int(train_end) : int(train_end) + int(horizon)]
    time_ctx = time_full[int(train_end) - int(context_length) : int(train_end)]
    time_fut = time_full[int(train_end) : int(train_end) + int(horizon)]
    x_pred = np.concatenate(
        [
            y_feat,
            np.concatenate([x_ctx, x_fut], axis=0),
            np.concatenate([time_ctx, time_fut], axis=0),
        ],
        axis=1,
    ).astype(dtype, copy=False)
    row = {
        "unique_id": unique_id,
        "cutoff_ds": ds_arr[int(train_end) - 1],
        "target_start_ds": ds_arr[int(train_end)],
        "target_end_ds": ds_arr[int(train_end) + int(horizon) - 1],
    }
    return x_pred, int(series_code), row, float(mean), float(std)


def make_panel_sequence_tensors(
    long_df: Any,
    *,
    cutoff: Any,
    horizon: int,
    context_length: int = 96,
    x_cols: tuple[str, ...] = (),
    normalize: bool = True,
    max_train_size: int | None = None,
    sample_step: int = 1,
    add_time_features: bool = True,
    dtype: Any = np.float64,
) -> dict[str, Any]:
    """
    Build packed sequence-model training and prediction bundles from long-format panel data.
    """
    df = _coerce_long_df(long_df, require_non_empty=True)
    horizon_int = _normalize_panel_sequence_horizon(horizon)
    context_length_int = _normalize_panel_sequence_context_length(context_length)
    sample_step_int = _normalize_panel_sequence_sample_step(sample_step)
    max_train_size_int = _normalize_panel_sequence_max_train_size(max_train_size)
    dtype_norm = _normalize_panel_sequence_dtype(dtype)
    x_cols_tup = _normalize_panel_x_cols(df, x_cols)

    out = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    x_dim = int(len(x_cols_tup))

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    series_id_parts: list[np.ndarray] = []
    train_index_parts: list[pd.DataFrame] = []

    pred_x_parts: list[np.ndarray] = []
    pred_series_ids: list[int] = []
    pred_rows: list[dict[str, Any]] = []
    pred_means: list[float] = []
    pred_stds: list[float] = []
    time_feature_names: tuple[str, ...] = ()

    for series_code, (unique_id, group) in enumerate(out.groupby("unique_id", sort=False)):
        group = group.reset_index(drop=True)
        _validate_panel_window_group(group, x_cols=x_cols_tup)

        ds_arr = group["ds"].to_numpy(copy=False)
        y_arr = group["y"].to_numpy(dtype=float, copy=False)
        cut_idx = _panel_sequence_cutoff_index(ds_arr, cutoff)
        if cut_idx is None:
            continue

        train_end = int(cut_idx) + 1
        slice_start = 0
        if max_train_size_int is not None:
            slice_start = max(0, int(train_end) - int(max_train_size_int))
        y_train = y_arr[slice_start:train_end]
        if int(y_train.size) < int(context_length_int) + int(horizon_int):
            continue

        y_train_scaled, mean, std = _panel_sequence_normalize_target(
            y_train,
            normalize=bool(normalize),
        )
        if x_dim > 0:
            x_full = group.loc[:, list(x_cols_tup)].to_numpy(dtype=float, copy=False)
        else:
            x_full = np.empty((len(group), 0), dtype=float)

        time_full, group_time_names = _panel_sequence_time_block(
            group,
            add_time_features=bool(add_time_features),
        )
        if not time_feature_names:
            time_feature_names = group_time_names

        X_series, y_series, series_ids, window_index = _panel_sequence_train_window_rows(
            unique_id=unique_id,
            ds_arr=ds_arr,
            y_train_scaled=y_train_scaled,
            x_train_seg=x_full[slice_start:train_end],
            time_train_seg=time_full[slice_start:train_end],
            slice_start=slice_start,
            series_code=series_code,
            context_length=context_length_int,
            horizon=horizon_int,
            sample_step=sample_step_int,
            dtype=dtype_norm,
        )
        if X_series.size == 0:
            continue

        X_parts.append(X_series)
        y_parts.append(y_series)
        series_id_parts.append(series_ids)
        train_index_parts.append(window_index)

        pred_row = _panel_sequence_predict_row(
            unique_id=unique_id,
            ds_arr=ds_arr,
            y_arr=y_arr,
            x_full=x_full,
            time_full=time_full,
            train_end=train_end,
            context_length=context_length_int,
            horizon=horizon_int,
            mean=mean,
            std=std,
            normalize=bool(normalize),
            series_code=series_code,
            dtype=dtype_norm,
        )
        if pred_row is None:
            continue
        pred_x, pred_series_id, pred_index_row, pred_mean, pred_std = pred_row
        pred_x_parts.append(pred_x)
        pred_series_ids.append(pred_series_id)
        pred_rows.append(pred_index_row)
        pred_means.append(pred_mean)
        pred_stds.append(pred_std)

    if not X_parts:
        raise ValueError("No training windows could be constructed for the given cutoff.")
    if not pred_x_parts:
        raise ValueError("No prediction windows could be constructed for the given cutoff.")

    X_train = np.concatenate(X_parts, axis=0).astype(dtype_norm, copy=False)
    y_train = np.concatenate(y_parts, axis=0).astype(dtype_norm, copy=False)
    series_id = np.concatenate(series_id_parts, axis=0)
    window_index = pd.concat(train_index_parts, axis=0, ignore_index=True, sort=False)

    pred_X = np.stack(pred_x_parts, axis=0).astype(dtype_norm, copy=False)
    pred_series_id = np.asarray(pred_series_ids, dtype=int)
    pred_index = pd.DataFrame(pred_rows)
    pred_mean_arr = np.asarray(pred_means, dtype=dtype_norm)
    pred_std_arr = np.asarray(pred_stds, dtype=dtype_norm)

    channel_names = ("y", *x_cols_tup, *time_feature_names)
    metadata = {
        "cutoff": pd.Timestamp(cutoff),
        "context_length": int(context_length_int),
        "horizon": int(horizon_int),
        "x_cols": x_cols_tup,
        "normalize": bool(normalize),
        "sample_step": int(sample_step_int),
        "max_train_size": max_train_size_int,
        "add_time_features": bool(add_time_features),
        "channel_names": tuple(channel_names),
        "time_feature_names": tuple(time_feature_names),
        "x_dim": int(len(x_cols_tup)),
        "time_dim": int(len(time_feature_names)),
        "n_series": int(out["unique_id"].nunique()),
        "n_train_windows": int(X_train.shape[0]),
        "n_predict_windows": int(pred_X.shape[0]),
        "input_dim": int(X_train.shape[2]),
    }
    return {
        "train": {
            "X": X_train,
            "y": y_train,
            "series_id": series_id,
            "window_index": window_index,
        },
        "predict": {
            "X": pred_X,
            "series_id": pred_series_id,
            "index": pred_index,
            "target_mean": pred_mean_arr,
            "target_std": pred_std_arr,
        },
        "metadata": metadata,
    }


def _validate_panel_sequence_bundle(bundle: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(bundle, dict):
        raise TypeError("bundle must be a dict returned by make_panel_sequence_tensors")
    if "train" not in bundle or "predict" not in bundle or "metadata" not in bundle:
        raise TypeError("bundle must contain train, predict, and metadata keys")
    train = bundle["train"]
    predict = bundle["predict"]
    metadata = bundle["metadata"]
    if not isinstance(train, dict) or not isinstance(predict, dict) or not isinstance(metadata, dict):
        raise TypeError("bundle sections must be dicts")
    required_train = {"X", "y", "series_id", "window_index"}
    missing_train = required_train.difference(train)
    if missing_train:
        raise TypeError(f"bundle['train'] missing required keys: {sorted(missing_train)}")
    if not isinstance(train["window_index"], pd.DataFrame):
        raise TypeError("bundle['train']['window_index'] must be a pandas DataFrame")
    return train, predict, metadata


def _subset_panel_sequence_partition(
    *,
    train: dict[str, Any],
    metadata: dict[str, Any],
    indices: list[int],
) -> dict[str, Any]:
    X = np.asarray(train["X"])
    y = np.asarray(train["y"])
    series_id = np.asarray(train["series_id"])
    window_index = train["window_index"].iloc[indices].reset_index(drop=True).copy()
    part_metadata = dict(metadata)
    part_metadata["n_train_windows"] = int(len(indices))
    return {
        "X": X[indices],
        "y": y[indices],
        "series_id": series_id[indices],
        "window_index": window_index,
        "metadata": part_metadata,
    }


def split_panel_sequence_tensors(
    bundle: Any,
    *,
    valid_size: int | None = None,
    test_size: int | None = None,
    valid_frac: float | None = None,
    test_frac: float | None = None,
    gap: int = 0,
    min_train_size: int = 1,
) -> dict[str, dict[str, Any]]:
    """
    Chronologically split packed panel training windows into train/valid/test partitions.
    """
    train, _predict, metadata = _validate_panel_sequence_bundle(bundle)
    gap_int = int(gap)
    min_train_size_int = int(min_train_size)
    if gap_int < 0:
        raise ValueError("gap must be >= 0")
    if min_train_size_int <= 0:
        raise ValueError("min_train_size must be >= 1")

    X = np.asarray(train["X"])
    y = np.asarray(train["y"])
    series_id = np.asarray(train["series_id"])
    window_index = train["window_index"].copy()
    order = window_index.sort_values(["unique_id", "cutoff_ds", "target_start_ds"], kind="mergesort").index
    X_sorted = X[order]
    y_sorted = y[order]
    series_id_sorted = series_id[order]
    window_index_sorted = window_index.iloc[order].reset_index(drop=True)

    partition_indices: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
    for unique_id, group in window_index_sorted.groupby("unique_id", sort=False):
        positions = group.index.to_list()
        n_rows = int(len(positions))
        valid_n = _resolved_partition_size(
            n_rows=n_rows,
            size=valid_size,
            frac=valid_frac,
            name="valid_size",
        )
        test_n = _resolved_partition_size(
            n_rows=n_rows,
            size=test_size,
            frac=test_frac,
            name="test_size",
        )
        if valid_n == 0 and test_n == 0:
            raise ValueError("at least one of valid/test size or frac must be positive")

        gap_before_valid = gap_int if valid_n > 0 else 0
        gap_before_test = gap_int if test_n > 0 else 0
        train_end = n_rows - valid_n - test_n - gap_before_valid - gap_before_test
        if train_end < min_train_size_int:
            raise ValueError(
                f"split_panel_sequence_tensors leaves fewer than min_train_size={min_train_size_int} "
                f"windows for unique_id={unique_id!r}"
            )

        valid_start = train_end + gap_before_valid
        valid_end = valid_start + valid_n
        test_start = valid_end + gap_before_test
        if test_start + test_n > n_rows:
            raise ValueError(
                f"split_panel_sequence_tensors consumes more windows than available for unique_id={unique_id!r}"
            )

        partition_indices["train"].extend(positions[:train_end])
        partition_indices["valid"].extend(positions[valid_start:valid_end])
        partition_indices["test"].extend(positions[test_start : test_start + test_n])

    sorted_train = {
        "X": X_sorted,
        "y": y_sorted,
        "series_id": series_id_sorted,
        "window_index": window_index_sorted,
    }
    return {
        name: _subset_panel_sequence_partition(
            train=sorted_train,
            metadata=metadata,
            indices=indices,
        )
        for name, indices in partition_indices.items()
    }


def _panel_sequence_block_dims(metadata: dict[str, Any]) -> tuple[int, int, int, int]:
    context_length = int(metadata["context_length"])
    horizon = int(metadata["horizon"])
    x_dim = int(metadata.get("x_dim", len(tuple(metadata.get("x_cols", ())))))
    time_dim = int(metadata.get("time_dim", len(tuple(metadata.get("time_feature_names", ())))))
    return context_length, horizon, x_dim, time_dim


def _packed_partition_to_sequence_blocks(
    section: dict[str, Any],
    *,
    metadata: dict[str, Any],
    include_target_y: bool,
    index_key: str,
) -> dict[str, Any]:
    X = np.asarray(section["X"])
    series_id = np.asarray(section["series_id"])
    index = section[index_key].copy()
    context_length, horizon, x_dim, time_dim = _panel_sequence_block_dims(metadata)

    if X.ndim != 3:
        raise ValueError("packed sequence X must be a 3D array")
    expected_input_dim = 1 + int(x_dim) + int(time_dim)
    if int(X.shape[1]) != int(context_length) + int(horizon):
        raise ValueError("packed sequence X has incompatible sequence length for metadata")
    if int(X.shape[2]) != expected_input_dim:
        raise ValueError("packed sequence X has incompatible channel dimension for metadata")

    x_start = 1
    time_start = 1 + int(x_dim)
    blocks = {
        "past_y": X[:, :context_length, 0:1],
        "future_y_seed": X[:, context_length:, 0:1],
        "past_x": X[:, :context_length, x_start:time_start],
        "future_x": X[:, context_length:, x_start:time_start],
        "past_time": X[:, :context_length, time_start:],
        "future_time": X[:, context_length:, time_start:],
        "series_id": series_id,
        index_key: index,
    }
    if include_target_y:
        blocks["target_y"] = np.asarray(section["y"])
    else:
        blocks["target_mean"] = np.asarray(section["target_mean"])
        blocks["target_std"] = np.asarray(section["target_std"])
    return blocks


def make_panel_sequence_blocks(
    long_df: Any,
    *,
    cutoff: Any,
    horizon: int,
    context_length: int = 96,
    x_cols: tuple[str, ...] = (),
    normalize: bool = True,
    max_train_size: int | None = None,
    sample_step: int = 1,
    add_time_features: bool = True,
    dtype: Any = np.float64,
) -> dict[str, Any]:
    """
    Build structured encoder-decoder style sequence blocks from long-format panel data.
    """
    packed = make_panel_sequence_tensors(
        long_df,
        cutoff=cutoff,
        horizon=int(horizon),
        context_length=int(context_length),
        x_cols=x_cols,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
        dtype=dtype,
    )
    metadata = dict(packed["metadata"])
    context_length_int, horizon_int, x_dim, time_dim = _panel_sequence_block_dims(metadata)
    metadata["block_layout"] = {
        "past_y": (context_length_int, 1),
        "future_y_seed": (horizon_int, 1),
        "past_x": (context_length_int, x_dim),
        "future_x": (horizon_int, x_dim),
        "past_time": (context_length_int, time_dim),
        "future_time": (horizon_int, time_dim),
        "target_y": (horizon_int,),
    }
    return {
        "train": _packed_partition_to_sequence_blocks(
            packed["train"],
            metadata=metadata,
            include_target_y=True,
            index_key="window_index",
        ),
        "predict": _packed_partition_to_sequence_blocks(
            packed["predict"],
            metadata=metadata,
            include_target_y=False,
            index_key="index",
        ),
        "metadata": metadata,
    }


def _validate_panel_sequence_blocks_bundle(
    bundle: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(bundle, dict):
        raise TypeError("bundle must be a dict returned by make_panel_sequence_blocks")
    if "train" not in bundle or "predict" not in bundle or "metadata" not in bundle:
        raise TypeError("bundle must contain train, predict, and metadata keys")
    train = bundle["train"]
    predict = bundle["predict"]
    metadata = bundle["metadata"]
    if not isinstance(train, dict) or not isinstance(predict, dict) or not isinstance(metadata, dict):
        raise TypeError("bundle sections must be dicts")
    required_train = {
        "past_y",
        "future_y_seed",
        "past_x",
        "future_x",
        "past_time",
        "future_time",
        "series_id",
        "target_y",
        "window_index",
    }
    missing_train = required_train.difference(train)
    if missing_train:
        raise TypeError(f"bundle['train'] missing required keys: {sorted(missing_train)}")
    if not isinstance(train["window_index"], pd.DataFrame):
        raise TypeError("bundle['train']['window_index'] must be a pandas DataFrame")
    return train, predict, metadata


def _sequence_blocks_train_to_packed(
    train: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    _context_length, _horizon, _x_dim, _time_dim = _panel_sequence_block_dims(metadata)
    past = np.concatenate(
        [
            np.asarray(train["past_y"]),
            np.asarray(train["past_x"]),
            np.asarray(train["past_time"]),
        ],
        axis=2,
    )
    future = np.concatenate(
        [
            np.asarray(train["future_y_seed"]),
            np.asarray(train["future_x"]),
            np.asarray(train["future_time"]),
        ],
        axis=2,
    )
    return {
        "X": np.concatenate([past, future], axis=1),
        "y": np.asarray(train["target_y"]),
        "series_id": np.asarray(train["series_id"]),
        "window_index": train["window_index"].copy(),
    }


def _sequence_blocks_partition_from_tensor_split(
    part: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    blocks = _packed_partition_to_sequence_blocks(
        {
            "X": part["X"],
            "y": part["y"],
            "series_id": part["series_id"],
            "window_index": part["window_index"],
        },
        metadata=metadata,
        include_target_y=True,
        index_key="window_index",
    )
    blocks["metadata"] = part["metadata"]
    return blocks


def split_panel_sequence_blocks(
    bundle: Any,
    *,
    valid_size: int | None = None,
    test_size: int | None = None,
    valid_frac: float | None = None,
    test_frac: float | None = None,
    gap: int = 0,
    min_train_size: int = 1,
) -> dict[str, dict[str, Any]]:
    """
    Chronologically split structured sequence blocks into train/valid/test partitions.
    """
    train, _predict, metadata = _validate_panel_sequence_blocks_bundle(bundle)
    packed_bundle = {
        "train": _sequence_blocks_train_to_packed(train, metadata=metadata),
        "predict": {},
        "metadata": metadata,
    }
    parts = split_panel_sequence_tensors(
        packed_bundle,
        valid_size=valid_size,
        test_size=test_size,
        valid_frac=valid_frac,
        test_frac=test_frac,
        gap=int(gap),
        min_train_size=int(min_train_size),
    )
    return {
        name: _sequence_blocks_partition_from_tensor_split(part, metadata=metadata)
        for name, part in parts.items()
    }


def _normalize_split_size(value: int | None, *, name: str) -> int:
    if value is None:
        return 0
    value_int = int(value)
    if value_int < 0:
        raise ValueError(f"{name} must be >= 0")
    return value_int


def _normalize_split_frac(value: float | None, *, name: str) -> float:
    if value is None:
        return 0.0
    value_float = float(value)
    if not (0.0 <= value_float < 1.0):
        raise ValueError(f"{name} must be in [0, 1)")
    return value_float


def _resolved_partition_size(
    *,
    n_rows: int,
    size: int | None,
    frac: float | None,
    name: str,
) -> int:
    if size is not None and frac is not None:
        raise ValueError(f"{name} and {name.replace('size', 'frac')} cannot both be set")
    if size is not None:
        return _normalize_split_size(size, name=name)
    return int(np.floor(_normalize_split_frac(frac, name=name.replace("size", "frac")) * float(n_rows)))


def split_long_df(
    long_df: Any,
    *,
    valid_size: int | None = None,
    test_size: int | None = None,
    valid_frac: float | None = None,
    test_frac: float | None = None,
    gap: int = 0,
    min_train_size: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Chronologically split each series in a long-format DataFrame into train/valid/test partitions.
    """
    df = _coerce_long_df(long_df, require_non_empty=True)
    gap_int = int(gap)
    min_train_size_int = int(min_train_size)
    if gap_int < 0:
        raise ValueError("gap must be >= 0")
    if min_train_size_int <= 0:
        raise ValueError("min_train_size must be >= 1")

    out = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    frames: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}

    for unique_id, group in out.groupby("unique_id", sort=False):
        group = group.reset_index(drop=True)
        n_rows = int(len(group))
        valid_n = _resolved_partition_size(
            n_rows=n_rows,
            size=valid_size,
            frac=valid_frac,
            name="valid_size",
        )
        test_n = _resolved_partition_size(
            n_rows=n_rows,
            size=test_size,
            frac=test_frac,
            name="test_size",
        )
        if valid_n == 0 and test_n == 0:
            raise ValueError("at least one of valid/test size or frac must be positive")

        gap_before_valid = gap_int if valid_n > 0 else 0
        gap_before_test = gap_int if test_n > 0 else 0
        train_end = n_rows - valid_n - test_n - gap_before_valid - gap_before_test
        if train_end < min_train_size_int:
            raise ValueError(
                f"split_long_df leaves fewer than min_train_size={min_train_size_int} "
                f"rows for unique_id={unique_id!r}"
            )

        valid_start = train_end + gap_before_valid
        valid_end = valid_start + valid_n
        test_start = valid_end + gap_before_test
        if test_start + test_n > n_rows:
            raise ValueError(f"split_long_df consumes more rows than available for unique_id={unique_id!r}")

        frames["train"].append(group.iloc[:train_end].copy())
        frames["valid"].append(group.iloc[valid_start:valid_end].copy())
        frames["test"].append(group.iloc[test_start : test_start + test_n].copy())

    result: dict[str, pd.DataFrame] = {}
    for name, part_frames in frames.items():
        non_empty = [frame for frame in part_frames if not frame.empty]
        if non_empty:
            result[name] = pd.concat(non_empty, axis=0, ignore_index=True, sort=False)
        else:
            result[name] = out.iloc[0:0].copy()
    return result


def _normalize_scaler_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in _ALLOWED_SCALER_METHODS:
        raise ValueError(f"method must be one of: {sorted(_ALLOWED_SCALER_METHODS)}")
    return normalized


def _normalize_scaler_scope(scope: str) -> str:
    normalized = str(scope).strip().lower()
    if normalized not in _ALLOWED_SCALER_SCOPES:
        raise ValueError(f"scope must be one of: {sorted(_ALLOWED_SCALER_SCOPES)}")
    return normalized


def fit_long_df_scaler(
    long_df: Any,
    *,
    method: str = "standard",
    scope: str = "per_series",
    columns: tuple[str, ...] = ("y",),
) -> pd.DataFrame:
    """
    Fit reversible scaling statistics for selected long-format numeric columns.
    """
    df = _coerce_long_df(long_df, require_non_empty=True)
    method_name = _normalize_scaler_method(method)
    scope_name = _normalize_scaler_scope(scope)
    value_cols = _resolve_numeric_columns(df, columns=columns, default_columns=("y",))
    ordered = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    if scope_name == "global":
        groups = [("__global__", ordered)]
    else:
        groups = list(ordered.groupby("unique_id", sort=False))

    rows: list[dict[str, Any]] = []
    for unique_id, group in groups:
        for col in value_cols:
            values = group[col].to_numpy(dtype=float, copy=False)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(f"cannot fit scaler for column={col!r}, unique_id={unique_id!r} with no finite values")

            data_min = float(np.min(finite))
            data_max = float(np.max(finite))
            if method_name == "standard":
                center = float(np.mean(finite))
                scale = float(np.std(finite, ddof=0))
                scale = 1.0 if not np.isfinite(scale) or scale <= 0.0 else scale
            elif method_name == "minmax":
                center = data_min
                scale = data_max - data_min
                scale = 1.0 if not np.isfinite(scale) or scale <= 0.0 else float(scale)
            else:
                center = 0.0
                scale = float(np.max(np.abs(finite)))
                scale = 1.0 if not np.isfinite(scale) or scale <= 0.0 else scale

            rows.append(
                {
                    "scope": scope_name,
                    "unique_id": str(unique_id),
                    "column": col,
                    "method": method_name,
                    "center": float(center),
                    "scale": float(scale),
                    "data_min": data_min,
                    "data_max": data_max,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["scope", "unique_id", "column", "method", "center", "scale", "data_min", "data_max"],
    )


def _validate_scaler_df(scaler_df: Any) -> pd.DataFrame:
    if not isinstance(scaler_df, pd.DataFrame):
        raise TypeError("scaler_df must be a pandas DataFrame")
    required = {"scope", "unique_id", "column", "method", "center", "scale", "data_min", "data_max"}
    missing = required.difference(scaler_df.columns)
    if missing:
        raise KeyError(f"scaler_df missing required columns: {sorted(missing)}")
    if scaler_df.empty:
        raise ValueError("scaler_df is empty")
    return scaler_df.copy()


def _apply_scale_values(values: np.ndarray, *, center: float, scale: float, inverse: bool) -> np.ndarray:
    out = values.astype(float, copy=True)
    mask = ~np.isnan(out)
    if not np.all(np.isfinite(out[mask])):
        raise ValueError("values to transform must be finite or null")
    if inverse:
        out[mask] = out[mask] * float(scale) + float(center)
    else:
        out[mask] = (out[mask] - float(center)) / float(scale)
    return out


def _scaler_row_lookup(scaler_df: pd.DataFrame, *, scope: str, unique_id: str, column: str) -> pd.Series:
    lookup_uid = "__global__" if scope == "global" else str(unique_id)
    matches = scaler_df.loc[
        (scaler_df["scope"] == scope)
        & (scaler_df["unique_id"].astype(str) == lookup_uid)
        & (scaler_df["column"].astype(str) == str(column))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one scaler row for scope={scope!r}, unique_id={lookup_uid!r}, column={column!r}"
        )
    return matches.iloc[0]


def transform_long_df_with_scaler(
    long_df: Any,
    scaler_df: Any,
    *,
    columns: tuple[str, ...] = ("y",),
) -> pd.DataFrame:
    """
    Apply fitted scaling statistics to selected long-format numeric columns.
    """
    df = _coerce_long_df(long_df, require_non_empty=True)
    scaler = _validate_scaler_df(scaler_df)
    value_cols = _resolve_numeric_columns(df, columns=columns, default_columns=("y",))
    out = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    scopes = {str(scope) for scope in scaler["scope"].astype(str)}
    if len(scopes) != 1:
        raise ValueError("scaler_df must contain a single scope")
    scope_name = _normalize_scaler_scope(next(iter(scopes)))

    for unique_id, idx in out.groupby("unique_id", sort=False).groups.items():
        for col in value_cols:
            row = _scaler_row_lookup(scaler, scope=scope_name, unique_id=str(unique_id), column=col)
            out.loc[idx, col] = _apply_scale_values(
                out.loc[idx, col].to_numpy(dtype=float, copy=False),
                center=float(row["center"]),
                scale=float(row["scale"]),
                inverse=False,
            )
    return out


def inverse_transform_long_df_with_scaler(
    long_df: Any,
    scaler_df: Any,
    *,
    columns: tuple[str, ...] = ("y",),
) -> pd.DataFrame:
    """
    Reverse fitted scaling statistics for selected long-format numeric columns.
    """
    df = _coerce_long_df(long_df, require_non_empty=True)
    scaler = _validate_scaler_df(scaler_df)
    value_cols = _resolve_numeric_columns(df, columns=columns, default_columns=("y",))
    out = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    scopes = {str(scope) for scope in scaler["scope"].astype(str)}
    if len(scopes) != 1:
        raise ValueError("scaler_df must contain a single scope")
    scope_name = _normalize_scaler_scope(next(iter(scopes)))

    for unique_id, idx in out.groupby("unique_id", sort=False).groups.items():
        for col in value_cols:
            row = _scaler_row_lookup(scaler, scope=scope_name, unique_id=str(unique_id), column=col)
            out.loc[idx, col] = _apply_scale_values(
                out.loc[idx, col].to_numpy(dtype=float, copy=False),
                center=float(row["center"]),
                scale=float(row["scale"]),
                inverse=True,
            )
    return out
