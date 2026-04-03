from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..contracts.capabilities import require_x_cols_if_needed as _contracts_require_x_cols_if_needed
from ..contracts.frames import (
    merge_history_and_future_df as _contracts_merge_history_and_future_df,
)
from ..contracts.frames import (
    require_future_df as _contracts_require_future_df,
)
from ..contracts.frames import (
    require_long_df as _contracts_require_long_df,
)
from ..contracts.frames import (
    require_observed_history_only as _contracts_require_observed_history_only,
)
from ..contracts.params import (
    normalize_covariate_roles as _contracts_normalize_covariate_roles,
)
from ..contracts.params import (
    normalize_model_params as _contracts_normalize_model_params,
)
from ..contracts.params import (
    normalize_static_cols as _contracts_normalize_static_cols,
)
from ..contracts.params import (
    normalize_x_cols as _contracts_normalize_x_cols,
)
from ..contracts.params import (
    parse_interval_levels as _contracts_parse_interval_levels,
)
from ..contracts.params import (
    parse_quantiles as _contracts_parse_quantiles,
)
from ..contracts.params import (
    required_quantiles_for_interval_levels as _contracts_required_quantiles_for_interval_levels,
)
from ..intervals import bootstrap_intervals
from ..long_df_cache import cached_series_slices, sorted_long_df
from . import model_execution as _model_execution

_HORIZON_MIN_MSG = "horizon must be >= 1"


def _require_long_df(long_df: Any) -> pd.DataFrame:
    return _contracts_require_long_df(long_df, require_non_empty=True)


def _require_future_df(future_df: Any) -> pd.DataFrame:
    return _contracts_require_future_df(future_df, require_non_empty=True)


def _merge_history_and_future_df(long_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    return _contracts_merge_history_and_future_df(long_df, future_df)


def _require_observed_history_only(df: pd.DataFrame) -> pd.DataFrame:
    return _contracts_require_observed_history_only(df)


def _normalize_model_params(model_params: dict[str, Any] | None) -> dict[str, Any]:
    return _contracts_normalize_model_params(model_params)


def _normalize_x_cols(model_params: dict[str, Any]) -> tuple[str, ...]:
    return _contracts_normalize_x_cols(model_params)


def _normalize_static_cols(model_params: dict[str, Any]) -> tuple[str, ...]:
    return _contracts_normalize_static_cols(model_params)


def _normalize_covariate_roles(
    model_params: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return _contracts_normalize_covariate_roles(model_params)


def _require_x_cols_if_needed(
    *,
    model: str,
    capabilities: dict[str, Any],
    x_cols: tuple[str, ...],
    context: str,
) -> None:
    _contracts_require_x_cols_if_needed(
        model=model,
        capabilities=capabilities,
        x_cols=x_cols,
        context=context,
    )


def _parse_interval_levels(levels: Any) -> tuple[float, ...]:
    return _contracts_parse_interval_levels(levels)


def _interval_level_label(level: float) -> str:
    pct = float(level) * 100.0
    rounded = round(pct)
    if abs(pct - rounded) < 1e-9:
        return str(int(rounded))
    return str(pct).replace(".", "_")


def _resolve_interval_min_train_size(n_obs: int, requested: int | None) -> int:
    n_obs_int = int(n_obs)
    if n_obs_int < 2:
        raise ValueError("Bootstrap forecast intervals require at least 2 observed history points")

    if requested is None:
        return max(1, min(24, n_obs_int - 1))

    min_train_size = int(requested)
    if min_train_size <= 0:
        raise ValueError("interval_min_train_size must be >= 1")
    if min_train_size >= n_obs_int:
        raise ValueError("interval_min_train_size must be smaller than the observed history length")
    return min_train_size


def _interval_column_names(levels: tuple[float, ...]) -> list[str]:
    cols: list[str] = []
    for level in levels:
        label = _interval_level_label(level)
        cols.extend([f"yhat_lo_{label}", f"yhat_hi_{label}"])
    return cols


def _parse_quantiles(quantiles: Any) -> tuple[float, ...]:
    return _contracts_parse_quantiles(quantiles)


def _required_quantiles_for_interval_levels(levels: tuple[float, ...]) -> tuple[float, ...]:
    return _contracts_required_quantiles_for_interval_levels(levels)


def _merge_quantiles_for_interval_levels(
    quantiles: Any,
    *,
    interval_levels: tuple[float, ...],
) -> tuple[float, ...]:
    if not interval_levels:
        return _parse_quantiles(quantiles)
    merged = set(_parse_quantiles(quantiles))
    merged.update(_required_quantiles_for_interval_levels(interval_levels))
    return tuple(sorted(merged))


def _add_interval_columns_from_quantile_predictions(
    pred: pd.DataFrame,
    *,
    interval_levels: tuple[float, ...],
) -> pd.DataFrame:
    if not interval_levels:
        return pred

    out = pred.copy()
    insert_after = list(out.columns)
    for level in interval_levels:
        q_lo = int(round(((1.0 - float(level)) / 2.0) * 100.0))
        q_hi = int(round((1.0 - ((1.0 - float(level)) / 2.0)) * 100.0))
        lo_col = f"yhat_p{q_lo}"
        hi_col = f"yhat_p{q_hi}"
        if lo_col not in out.columns or hi_col not in out.columns:
            raise ValueError(
                f"forecast output is missing quantile columns required for interval level {level!r}: "
                f"{lo_col!r}, {hi_col!r}"
            )
        label = _interval_level_label(level)
        out[f"yhat_lo_{label}"] = out[lo_col].to_numpy(dtype=float, copy=False)
        out[f"yhat_hi_{label}"] = out[hi_col].to_numpy(dtype=float, copy=False)
        insert_after.extend([f"yhat_lo_{label}", f"yhat_hi_{label}"])

    ordered = [c for c in insert_after if c in out.columns and c not in {"model"}]
    if "model" in out.columns:
        ordered.append("model")
    return out.loc[:, ordered]


def _local_interval_columns(
    *,
    train_y: np.ndarray,
    model: str,
    model_params: dict[str, Any],
    horizon: int,
    interval_levels: tuple[float, ...],
    interval_min_train_size: int | None,
    interval_samples: int,
    interval_seed: int | None,
) -> dict[str, np.ndarray]:
    if not interval_levels:
        return {}

    base_forecaster = _model_execution.make_local_forecaster_runner(str(model), model_params)
    min_train_size = _resolve_interval_min_train_size(int(train_y.size), interval_min_train_size)

    out: dict[str, np.ndarray] = {}
    for level in interval_levels:
        q_lo = (1.0 - float(level)) / 2.0
        q_hi = 1.0 - q_lo
        payload = bootstrap_intervals(
            train_y,
            horizon=int(horizon),
            forecaster=base_forecaster,
            min_train_size=min_train_size,
            n_samples=int(interval_samples),
            quantiles=(q_lo, q_hi),
            seed=interval_seed,
        )
        label = _interval_level_label(level)
        out[f"yhat_lo_{label}"] = np.asarray(payload["lower"], dtype=float)
        out[f"yhat_hi_{label}"] = np.asarray(payload["upper"], dtype=float)
    return out


def _local_xreg_interval_payload(
    *,
    model: str,
    train_y: np.ndarray,
    horizon: int,
    train_exog: np.ndarray,
    future_exog: np.ndarray,
    interval_levels: tuple[float, ...],
    model_params: dict[str, Any],
) -> dict[str, Any]:
    if str(model) == "sarimax":
        from ..models.statsmodels_wrap import (
            sarimax_forecast_with_intervals as _local_xreg_forecast_with_intervals,
        )
    elif str(model) == "auto-arima":
        from ..models.statsmodels_wrap import (
            auto_arima_forecast_with_intervals as _local_xreg_forecast_with_intervals,
        )
    else:
        raise ValueError(f"Model {model!r} does not support interval_levels with x_cols")

    payload = _local_xreg_forecast_with_intervals(
        train_y,
        int(horizon),
        interval_levels=interval_levels,
        train_exog=train_exog,
        future_exog=future_exog,
        **model_params,
    )
    mean = np.asarray(payload["mean"], dtype=float)
    if mean.shape != (int(horizon),):
        raise ValueError(f"forecaster must return shape ({int(horizon)},), got {mean.shape}")

    out = {"yhat": mean}
    for level in interval_levels:
        label = _interval_level_label(level)
        lo, hi = payload["intervals"][float(level)]
        out[f"yhat_lo_{label}"] = np.asarray(lo, dtype=float)
        out[f"yhat_hi_{label}"] = np.asarray(hi, dtype=float)
    return out


_call_local_xreg_forecaster = _model_execution.call_local_xreg_forecaster


def _as_datetime_index(ds: Any) -> pd.DatetimeIndex | None:
    if isinstance(ds, pd.DatetimeIndex):
        return ds

    idx = pd.Index(ds)

    if pd.api.types.is_numeric_dtype(idx.dtype):
        return None

    if pd.api.types.is_datetime64_any_dtype(idx.dtype):
        return pd.DatetimeIndex(idx)

    parsed = pd.to_datetime(idx, errors="coerce")
    if getattr(parsed, "isna", lambda: pd.Series([], dtype=bool))().any():
        return None
    return pd.DatetimeIndex(parsed)


def _infer_future_ds(ds: Any, horizon: int) -> pd.Index:
    h = int(horizon)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)

    idx = pd.Index(ds)
    if len(idx) == 0:
        raise ValueError("Cannot infer future timestamps from an empty index")

    dt_idx = _as_datetime_index(idx)
    if dt_idx is not None:
        freq = dt_idx.freqstr or pd.infer_freq(dt_idx)
        if freq:
            return pd.date_range(start=dt_idx[-1], periods=h + 1, freq=freq)[1:]
        if len(dt_idx) >= 2:
            delta = dt_idx[-1] - dt_idx[-2]
            if delta == pd.Timedelta(0):
                raise ValueError("Could not infer future timestamps from repeated ds values")
            return pd.DatetimeIndex([dt_idx[-1] + (i + 1) * delta for i in range(h)])
        raise ValueError("Could not infer future timestamps; provide at least two ds values")

    if len(idx) >= 2:
        last = idx[-1]
        prev = idx[-2]
        try:
            step = last - prev
        except Exception as e:  # noqa: BLE001
            raise ValueError("Could not infer future timestamps from ds") from e
        return pd.Index([last + step * (i + 1) for i in range(h)])

    if np.issubdtype(idx.dtype, np.number):
        last_num = float(idx[-1])
        return pd.Index([last_num + float(i + 1) for i in range(h)])

    raise ValueError("Could not infer future timestamps from ds")


def _future_frame_for_group(g: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    future_ds = _infer_future_ds(g["ds"], int(horizon))
    out = pd.DataFrame(
        {
            "unique_id": [str(g["unique_id"].iloc[0])] * int(horizon),
            "ds": future_ds,
            "y": [np.nan] * int(horizon),
        }
    )
    for col in g.columns:
        if col in {"unique_id", "ds", "y"}:
            continue
        out[col] = np.nan
    return out


def _append_frame_rows(
    columns: dict[str, list[Any]],
    frame: pd.DataFrame,
    *,
    ordered_cols: list[str],
) -> None:
    for col in ordered_cols:
        columns[str(col)].extend(frame[str(col)].tolist())


def _append_future_rows_for_group(
    columns: dict[str, list[Any]],
    g: pd.DataFrame,
    *,
    horizon: int,
    ordered_cols: list[str],
) -> None:
    future_ds = pd.Index(_infer_future_ds(g["ds"], int(horizon)))
    uid = str(g["unique_id"].iloc[0])
    columns["unique_id"].extend([uid] * int(horizon))
    columns["ds"].extend(future_ds.tolist())
    columns["y"].extend([np.nan] * int(horizon))
    for col in ordered_cols:
        if col in {"unique_id", "ds", "y"}:
            continue
        columns[str(col)].extend([np.nan] * int(horizon))


def _prepare_local_xreg_forecast_group(
    g: pd.DataFrame,
    *,
    horizon: int,
    x_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    h = int(horizon)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)

    missing_x_cols = [col for col in x_cols if col not in g.columns]
    if missing_x_cols:
        raise KeyError(f"long_df missing required x_cols: {missing_x_cols}")

    g = g.sort_values(["ds"], kind="mergesort").reset_index(drop=True)
    y_notna = g["y"].notna().to_numpy(dtype=bool, copy=False)
    if not y_notna.any():
        raise ValueError(
            f"Local forecast with x_cols requires observed history for unique_id={g['unique_id'].iloc[0]!r}"
        )

    missing_idx = np.flatnonzero(~y_notna)
    if missing_idx.size > 0:
        first_missing = int(missing_idx[0])
        if y_notna[first_missing:].any():
            raise ValueError(
                "Local forecast with x_cols requires missing y values only after the observed history"
            )

    observed_count = int(y_notna.sum())
    observed = g.iloc[:observed_count].copy()
    future = g.iloc[observed_count:].copy()
    if len(future) < h:
        raise ValueError(
            "Local forecast with x_cols requires at least horizon future rows per series"
        )

    future = future.iloc[:h].copy()
    missing_observed_x = [col for col in x_cols if observed[col].isna().any()]
    if missing_observed_x:
        raise ValueError(
            f"Local forecast observed rows are missing required x_cols: {missing_observed_x}"
        )

    missing_future_x = [col for col in x_cols if future[col].isna().any()]
    if missing_future_x:
        raise ValueError(
            f"Local forecast future rows are missing required x_cols: {missing_future_x}"
        )

    cutoff = observed["ds"].iloc[-1]
    return observed, future, cutoff


def _global_forecast_group_cutoff_and_future(
    uid: Any,
    g: pd.DataFrame,
) -> tuple[Any, pd.DataFrame, int]:
    y_notna = g["y"].notna().to_numpy(dtype=bool, copy=False)
    if not y_notna.any():
        raise ValueError(f"Global forecast requires observed history for unique_id={uid!r}")

    missing_idx = np.flatnonzero(~y_notna)
    if missing_idx.size > 0:
        first_missing = int(missing_idx[0])
        if y_notna[first_missing:].any():
            raise ValueError(
                "Global forecast requires missing y values only after the observed history"
            )

    observed_count = int(y_notna.sum())
    cutoff = g["ds"].iloc[observed_count - 1]
    future = g.iloc[observed_count:]
    return cutoff, future, observed_count


def _validate_global_forecast_group_x_cols(
    g: pd.DataFrame,
    *,
    future: pd.DataFrame,
    observed_count: int,
    horizon: int,
    x_cols: tuple[str, ...],
) -> None:
    if len(future) < int(horizon):
        raise ValueError(
            "Global forecast with x_cols requires at least horizon future rows per series"
        )

    future_slice = future.iloc[: int(horizon)]
    missing_future_x = [col for col in x_cols if future_slice[col].isna().any()]
    if missing_future_x:
        raise ValueError(
            f"Global forecast future rows are missing required x_cols: {missing_future_x}"
        )

    observed_slice = g.iloc[:observed_count]
    missing_observed_x = [col for col in x_cols if observed_slice[col].isna().any()]
    if missing_observed_x:
        raise ValueError(
            f"Global forecast observed rows are missing required x_cols: {missing_observed_x}"
        )


def _global_forecast_group_future_frame(
    uid: Any,
    g: pd.DataFrame,
    *,
    horizon: int,
    x_cols: tuple[str, ...],
) -> tuple[Any, pd.DataFrame | None]:
    cutoff, future, observed_count = _global_forecast_group_cutoff_and_future(uid, g)
    if x_cols:
        _validate_global_forecast_group_x_cols(
            g,
            future=future,
            observed_count=observed_count,
            horizon=int(horizon),
            x_cols=x_cols,
        )
        return cutoff, None

    missing_future = int(horizon) - int(len(future))
    if missing_future <= 0:
        return cutoff, None
    return cutoff, _future_frame_for_group(g, horizon=missing_future)


def _validated_global_forecast_cutoff(cutoffs: list[Any]) -> Any:
    cutoff_by_uid = pd.Series(cutoffs)
    if cutoff_by_uid.nunique() != 1:
        raise ValueError(
            "Global forecast currently requires all series to share the same last observed timestamp"
        )
    return cutoff_by_uid.iloc[0]


def _prepare_global_forecast_input(
    df: pd.DataFrame,
    *,
    horizon: int,
    x_cols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, Any]:
    h = int(horizon)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)

    missing_x_cols = [col for col in x_cols if col not in df.columns]
    if missing_x_cols:
        raise KeyError(f"long_df missing required x_cols: {missing_x_cols}")

    df = sorted_long_df(df, reset_index=True)

    cutoffs: list[Any] = []
    ordered_cols = [str(col) for col in df.columns]
    columns = {col: [] for col in ordered_cols}
    has_future_rows = False
    for uid, start, stop in cached_series_slices(df):
        g = df.iloc[start:stop]
        _append_frame_rows(columns, g, ordered_cols=ordered_cols)
        if x_cols:
            cutoff, future, observed_count = _global_forecast_group_cutoff_and_future(uid, g)
            _validate_global_forecast_group_x_cols(
                g,
                future=future,
                observed_count=observed_count,
                horizon=h,
                x_cols=x_cols,
            )
            future_frame = None
        else:
            cutoff, future, _observed_count = _global_forecast_group_cutoff_and_future(uid, g)
            missing_future = int(h) - int(len(future))
            future_frame = None if missing_future <= 0 else missing_future
        cutoffs.append(cutoff)
        if future_frame is not None:
            has_future_rows = True
            _append_future_rows_for_group(
                columns,
                g,
                horizon=int(future_frame),
                ordered_cols=ordered_cols,
            )

    cutoff = _validated_global_forecast_cutoff(cutoffs)
    if not has_future_rows:
        return df, cutoff

    augmented = pd.DataFrame(columns, columns=ordered_cols)
    return augmented, cutoff


def _is_sorted_forecast_frame(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return True

    uid = df["unique_id"].astype("string")
    ds = pd.to_datetime(df["ds"], errors="coerce")
    prev_uid = uid.shift(1)
    prev_ds = ds.shift(1)
    uid_backwards = (uid < prev_uid).fillna(False)
    ds_backwards = ((uid == prev_uid) & (ds < prev_ds)).fillna(False)
    return not bool((uid_backwards | ds_backwards).any())


def _forecast_step_index(pred: pd.DataFrame) -> np.ndarray:
    uid_arr = pred["unique_id"].astype("string").to_numpy(copy=False)
    if uid_arr.size == 0:
        return np.array([], dtype=int)
    boundaries = (
        np.flatnonzero(uid_arr[1:] != uid_arr[:-1]) + 1
        if uid_arr.size > 1
        else np.array([], dtype=int)
    )
    starts = np.concatenate((np.array([0], dtype=int), boundaries.astype(int, copy=False)))
    stops = np.concatenate(
        (boundaries.astype(int, copy=False), np.array([uid_arr.size], dtype=int))
    )
    steps = np.empty(uid_arr.size, dtype=int)
    for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
        steps[start:stop] = np.arange(1, int(stop - start) + 1, dtype=int)
    return steps


def _finalize_forecast_frame(pred: pd.DataFrame, *, cutoff: Any, model: str) -> pd.DataFrame:
    if not isinstance(pred, pd.DataFrame):
        raise TypeError(f"forecast output must be a pandas DataFrame, got: {type(pred).__name__}")
    required = {"unique_id", "ds", "yhat"}
    missing = required.difference(pred.columns)
    if missing:
        raise KeyError(f"forecast output missing required columns: {sorted(missing)}")

    pred = (
        pred
        if _is_sorted_forecast_frame(pred)
        and isinstance(pred.index, pd.RangeIndex)
        and int(pred.index.start) == 0
        and int(pred.index.step) == 1
        else pred.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    ).copy()
    pred_cols = [c for c in pred.columns if c not in {"unique_id", "ds"}]
    pred["cutoff"] = cutoff
    pred["step"] = _forecast_step_index(pred)
    pred["model"] = str(model)
    ordered = ["unique_id", "ds", "cutoff", "step", *pred_cols, "model"]
    return pred.loc[:, ordered]


def _forecast_result_row(
    *,
    uid: Any,
    ds: Any,
    cutoff: Any,
    step: int,
    yhat: float,
    model: str,
    interval_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "unique_id": str(uid),
        "ds": ds,
        "cutoff": cutoff,
        "step": int(step),
        "yhat": float(yhat),
    }
    if interval_values:
        row.update(interval_values)
    row["model"] = str(model)
    return row


def _forecast_columns(*, interval_cols: list[str]) -> dict[str, list[Any]]:
    columns = {
        "unique_id": [],
        "ds": [],
        "cutoff": [],
        "step": [],
        "yhat": [],
        "model": [],
    }
    for col in interval_cols:
        columns[str(col)] = []
    return columns


def _append_forecast_prediction_rows(
    columns: dict[str, list[Any]],
    *,
    uid: Any,
    ds_values: Any,
    cutoff: Any,
    yhat: np.ndarray,
    model: str,
    interval_cols: list[str],
    interval_data: dict[str, np.ndarray],
) -> None:
    future_ds = pd.Index(ds_values)
    horizon = int(len(future_ds))
    columns["unique_id"].extend([str(uid)] * horizon)
    columns["ds"].extend(future_ds.tolist())
    columns["cutoff"].extend([cutoff] * horizon)
    columns["step"].extend(range(1, horizon + 1))
    columns["yhat"].extend(np.asarray(yhat, dtype=float).tolist())
    for col in interval_cols:
        values = np.asarray(interval_data.get(str(col), np.repeat(np.nan, horizon)), dtype=float)
        columns[str(col)].extend(values.tolist())
    columns["model"].extend([str(model)] * horizon)


def _forecast_frame_from_columns(
    columns: dict[str, list[Any]],
    *,
    interval_cols: list[str],
) -> pd.DataFrame:
    ordered = ["unique_id", "ds", "cutoff", "step", "yhat", *interval_cols, "model"]
    return pd.DataFrame(columns, columns=ordered)


def _forecast_local_xreg_long_df(
    *,
    model: str,
    df: pd.DataFrame,
    horizon: int,
    params: dict[str, Any],
    capabilities: dict[str, Any],
    x_cols: tuple[str, ...],
    levels: tuple[float, ...],
) -> pd.DataFrame:
    if not bool(capabilities.get("supports_x_cols", False)):
        raise ValueError(f"Model {model!r} does not support x_cols in forecast_model_long_df")

    h = int(horizon)
    interval_cols = _interval_column_names(levels)
    columns = _forecast_columns(interval_cols=interval_cols)
    local_xreg_params = dict(params)
    local_xreg_params.pop("x_cols", None)
    for uid, g in df.groupby("unique_id", sort=False):
        observed, future, cutoff = _prepare_local_xreg_forecast_group(
            g,
            horizon=h,
            x_cols=x_cols,
        )
        train_y = observed["y"].to_numpy(dtype=float, copy=False)
        train_exog = observed.loc[:, list(x_cols)].to_numpy(dtype=float, copy=False)
        future_exog = future.loc[:, list(x_cols)].to_numpy(dtype=float, copy=False)
        if levels:
            if not bool(capabilities.get("supports_interval_forecast_with_x_cols", False)):
                raise ValueError(
                    f"Model {model!r} does not support interval_levels with x_cols "
                    "in forecast_model_long_df"
                )
            pred_payload = _local_xreg_interval_payload(
                model=model,
                train_y=train_y,
                horizon=h,
                train_exog=train_exog,
                future_exog=future_exog,
                interval_levels=levels,
                model_params=local_xreg_params,
            )
            yhat = np.asarray(pred_payload["yhat"], dtype=float)
        else:
            pred_payload = {}
            yhat = _call_local_xreg_forecaster(
                model=model,
                train_y=train_y,
                horizon=h,
                train_exog=train_exog,
                future_exog=future_exog,
                model_params=local_xreg_params,
            )

        interval_data = {
            col: np.asarray(pred_payload.get(col, np.repeat(np.nan, h)), dtype=float)
            for col in interval_cols
        }
        _append_forecast_prediction_rows(
            columns,
            uid=uid,
            ds_values=future["ds"],
            cutoff=cutoff,
            yhat=np.asarray(yhat, dtype=float),
            model=model,
            interval_cols=interval_cols,
            interval_data=interval_data,
        )

    return _forecast_frame_from_columns(columns, interval_cols=interval_cols)


def _forecast_local_univariate_long_df(
    *,
    model: str,
    df: pd.DataFrame,
    future_df: Any | None,
    horizon: int,
    params: dict[str, Any],
    levels: tuple[float, ...],
    interval_min_train_size: int | None,
    interval_samples: int,
    interval_seed: int | None,
) -> pd.DataFrame:
    df = _require_observed_history_only(df)
    h = int(horizon)
    interval_cols = _interval_column_names(levels)
    columns = _forecast_columns(interval_cols=interval_cols)
    for uid, g in df.groupby("unique_id", sort=False):
        if future_df is not None:
            observed, future, cutoff = _prepare_local_xreg_forecast_group(
                g,
                horizon=h,
                x_cols=(),
            )
            future_ds = pd.Index(future["ds"])
            train_y = observed["y"].to_numpy(dtype=float, copy=False)
        else:
            cutoff = g["ds"].iloc[-1]
            future_ds = _infer_future_ds(g["ds"], h)
            train_y = g["y"].to_numpy(dtype=float, copy=False)

        forecaster = _model_execution.make_local_forecaster_object_runner(model, params).fit(
            train_y
        )
        yhat = np.asarray(forecaster.predict(h), dtype=float)
        if yhat.shape != (h,):
            raise ValueError(f"forecaster must return shape ({h},), got {yhat.shape}")

        interval_data = _local_interval_columns(
            train_y=np.asarray(train_y, dtype=float),
            model=model,
            model_params=params,
            horizon=h,
            interval_levels=levels,
            interval_min_train_size=interval_min_train_size,
            interval_samples=int(interval_samples),
            interval_seed=interval_seed,
        )

        _append_forecast_prediction_rows(
            columns,
            uid=uid,
            ds_values=future_ds,
            cutoff=cutoff,
            yhat=yhat,
            model=model,
            interval_cols=interval_cols,
            interval_data=interval_data,
        )

    return _forecast_frame_from_columns(columns, interval_cols=interval_cols)


def _forecast_global_long_df(
    *,
    model: str,
    df: pd.DataFrame,
    horizon: int,
    params: dict[str, Any],
    capabilities: dict[str, Any],
    x_cols: tuple[str, ...],
    levels: tuple[float, ...],
) -> pd.DataFrame:
    if x_cols and not bool(capabilities.get("supports_x_cols", False)):
        raise ValueError(f"Model {model!r} does not support x_cols in forecast_model_long_df")
    if levels and not bool(capabilities.get("supports_interval_forecast", False)):
        raise ValueError(
            f"Model {model!r} does not support interval_levels in forecast_model_long_df"
        )

    h = int(horizon)
    params_final = dict(params)
    if levels:
        params_final["quantiles"] = _merge_quantiles_for_interval_levels(
            params_final.get("quantiles"),
            interval_levels=levels,
        )

    augmented, cutoff = _prepare_global_forecast_input(df, horizon=h, x_cols=x_cols)
    forecaster = _model_execution.make_global_forecaster_object_runner(model, params_final).fit(
        augmented
    )
    pred = forecaster.predict(cutoff, h)
    pred = _finalize_forecast_frame(pred, cutoff=cutoff, model=model)
    return _add_interval_columns_from_quantile_predictions(pred, interval_levels=levels)


def forecast_model_long_df(
    *,
    model: str,
    long_df: Any,
    future_df: Any | None = None,
    horizon: int,
    model_params: dict[str, Any] | None = None,
    interval_levels: Any = None,
    interval_min_train_size: int | None = None,
    interval_samples: int = 1000,
    interval_seed: int | None = None,
) -> pd.DataFrame:
    """
    Forecast from the end of each series in a canonical long-format DataFrame.

    Output columns mirror the existing CV predictions table where possible:
      unique_id, ds, cutoff, step, yhat, model
    """
    df = _require_long_df(long_df)
    if future_df is not None:
        df = _merge_history_and_future_df(df, _require_future_df(future_df))

    params = _normalize_model_params(model_params)
    model_name = str(model)
    model_spec = _model_execution.get_model_spec(model_name)
    interface = str(model_spec.interface).lower().strip()
    capabilities = dict(model_spec.capabilities)
    levels = _parse_interval_levels(interval_levels)
    static_cols = _normalize_static_cols(params)

    if interface == "local":
        df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
        historic_x_cols, x_cols = _normalize_covariate_roles(params)
        if historic_x_cols:
            raise ValueError("historic_x_cols are not yet supported in forecast_model_long_df")
        if static_cols:
            if not bool(capabilities.get("supports_static_cols", False)):
                raise ValueError(
                    f"Model {model_name!r} does not support static_cols in forecast_model_long_df"
                )
            raise ValueError(
                "static_cols are not yet supported for local models in forecast_model_long_df"
            )
        _require_x_cols_if_needed(
            model=model_name,
            capabilities=capabilities,
            x_cols=x_cols,
            context="forecast_model_long_df",
        )
        if x_cols:
            return _forecast_local_xreg_long_df(
                model=model_name,
                df=df,
                horizon=int(horizon),
                params=params,
                capabilities=capabilities,
                x_cols=x_cols,
                levels=levels,
            )

        return _forecast_local_univariate_long_df(
            model=model_name,
            df=df,
            future_df=future_df,
            horizon=int(horizon),
            params=params,
            levels=levels,
            interval_min_train_size=interval_min_train_size,
            interval_samples=int(interval_samples),
            interval_seed=interval_seed,
        )

    if interface == "global":
        historic_x_cols, x_cols = _normalize_covariate_roles(params)
        if historic_x_cols:
            raise ValueError("historic_x_cols are not yet supported in forecast_model_long_df")
        if static_cols and not bool(capabilities.get("supports_static_cols", False)):
            raise ValueError(
                f"Model {model_name!r} does not support static_cols in forecast_model_long_df"
            )
        _require_x_cols_if_needed(
            model=model_name,
            capabilities=capabilities,
            x_cols=x_cols,
            context="forecast_model_long_df",
        )
        return _forecast_global_long_df(
            model=model_name,
            df=df,
            horizon=int(horizon),
            params=params,
            capabilities=capabilities,
            x_cols=x_cols,
            levels=levels,
        )

    raise ValueError(f"Unknown model interface: {model_spec.interface!r}")


def forecast_model(
    *,
    model: str,
    y: Any,
    horizon: int,
    ds: Any | None = None,
    unique_id: str = "series=0",
    model_params: dict[str, Any] | None = None,
    interval_levels: Any = None,
    interval_min_train_size: int | None = None,
    interval_samples: int = 1000,
    interval_seed: int | None = None,
) -> pd.DataFrame:
    """
    Forecast a single series into the future.

    If `ds` is omitted, a simple integer index is assumed.
    """
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 1:
        raise ValueError(f"Expected a 1D series, got shape {y_arr.shape}")

    if ds is None:
        if isinstance(y, pd.Series) and not isinstance(y.index, pd.RangeIndex):
            ds_values = y.index
        else:
            ds_values = np.arange(y_arr.size, dtype=int)
    else:
        ds_values = ds

    if len(pd.Index(ds_values)) != int(y_arr.size):
        raise ValueError("y and ds must have the same length")

    long_df = pd.DataFrame(
        {
            "unique_id": [str(unique_id)] * int(y_arr.size),
            "ds": pd.Index(ds_values),
            "y": y_arr.astype(float, copy=False),
        }
    )

    return forecast_model_long_df(
        model=str(model),
        long_df=long_df,
        horizon=int(horizon),
        model_params=model_params,
        interval_levels=interval_levels,
        interval_min_train_size=interval_min_train_size,
        interval_samples=int(interval_samples),
        interval_seed=interval_seed,
    )


forecast_long_df = forecast_model_long_df
forecast_series = forecast_model


__all__ = [
    "_finalize_forecast_frame",
    "_infer_future_ds",
    "_prepare_global_forecast_input",
    "forecast_long_df",
    "forecast_model",
    "forecast_model_long_df",
    "forecast_series",
]
