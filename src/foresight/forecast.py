from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .intervals import bootstrap_intervals
from .models.registry import (
    get_model_spec,
    make_forecaster,
    make_forecaster_object,
    make_global_forecaster_object,
)


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


def _require_future_df(future_df: Any) -> pd.DataFrame:
    if not isinstance(future_df, pd.DataFrame):
        raise TypeError("future_df must be a pandas DataFrame")
    required = {"unique_id", "ds"}
    missing = required.difference(future_df.columns)
    if missing:
        raise KeyError(f"future_df missing required columns: {sorted(missing)}")
    if future_df.empty:
        raise ValueError("future_df is empty")

    out = future_df.copy()
    if "y" not in out.columns:
        out["y"] = np.nan
    elif out["y"].notna().any():
        raise ValueError("future_df must not contain observed y values")
    return out


def _merge_history_and_future_df(long_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    overlap = (
        long_df.loc[:, ["unique_id", "ds"]]
        .merge(future_df.loc[:, ["unique_id", "ds"]], on=["unique_id", "ds"], how="inner")
        .drop_duplicates()
    )
    if not overlap.empty:
        raise ValueError("future_df overlaps with long_df on unique_id/ds")

    cols = list(long_df.columns)
    for col in future_df.columns:
        if col not in cols:
            cols.append(col)

    left = long_df.copy()
    right = future_df.copy()
    for col in cols:
        if col not in left.columns:
            left[col] = np.nan
        if col not in right.columns:
            right[col] = np.nan

    merged = pd.concat(
        [left.loc[:, cols], right.loc[:, cols]], axis=0, ignore_index=True, sort=False
    )
    return merged.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)


def _require_observed_history_only(df: pd.DataFrame) -> pd.DataFrame:
    if df["y"].isna().any():
        raise ValueError("long_df contains missing y values; provide observed history only")
    return df


def _normalize_model_params(model_params: dict[str, Any] | None) -> dict[str, Any]:
    return dict(model_params or {})


def _normalize_x_cols(model_params: dict[str, Any]) -> tuple[str, ...]:
    raw = model_params.get("x_cols")
    if raw is None:
        return ()
    if isinstance(raw, str):
        s = raw.strip()
        return tuple([p.strip() for p in s.split(",") if p.strip()]) if s else ()
    if isinstance(raw, list | tuple):
        return tuple([str(v).strip() for v in raw if str(v).strip()])
    s = str(raw).strip()
    return (s,) if s else ()


def _normalize_covariate_roles(
    model_params: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def _normalize(raw: Any) -> tuple[str, ...]:
        if raw is None:
            return ()
        if isinstance(raw, str):
            s = raw.strip()
            return tuple([p.strip() for p in s.split(",") if p.strip()]) if s else ()
        if isinstance(raw, list | tuple):
            return tuple([str(v).strip() for v in raw if str(v).strip()])
        s = str(raw).strip()
        return (s,) if s else ()

    future = _normalize(model_params.get("future_x_cols"))
    legacy = _normalize_x_cols(model_params)
    if legacy:
        future = tuple([*future, *[c for c in legacy if c not in future]])
    historic = _normalize(model_params.get("historic_x_cols"))
    return historic, future


def _require_x_cols_if_needed(
    *,
    model: str,
    capabilities: dict[str, Any],
    x_cols: tuple[str, ...],
    context: str,
) -> None:
    if bool(capabilities.get("requires_future_covariates", False)) and not x_cols:
        raise ValueError(f"Model {model!r} requires future covariates via x_cols in {context}")


def _parse_interval_levels(levels: Any) -> tuple[float, ...]:
    if levels is None:
        return ()

    if isinstance(levels, list | tuple):
        items = list(levels)
    elif isinstance(levels, str):
        s = levels.strip()
        items = [] if not s else [p.strip() for p in s.split(",") if p.strip()]
    else:
        items = [levels]

    out: list[float] = []
    for it in items:
        level = float(it)
        if level >= 1.0:
            level = level / 100.0
        if not (0.0 < level < 1.0):
            raise ValueError("interval_levels must be in (0,1) or percentages like 80,90")
        out.append(level)
    return tuple(sorted(set(out)))


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
    if quantiles is None:
        return ()

    if isinstance(quantiles, list | tuple):
        items = list(quantiles)
    elif isinstance(quantiles, str):
        s = quantiles.strip()
        items = [] if not s else [p.strip() for p in s.split(",") if p.strip()]
    else:
        items = [quantiles]

    out: list[float] = []
    for item in items:
        q = float(item)
        if q >= 1.0:
            q = q / 100.0
        if not (0.0 < q < 1.0):
            raise ValueError("quantiles must be in (0,1) or percentages like 10,50,90")
        pct = q * 100.0
        if abs(pct - round(pct)) > 1e-9:
            raise ValueError("quantiles must align to integer percentiles (e.g. 0.1,0.5,0.9)")
        out.append(q)
    return tuple(sorted(set(out)))


def _required_quantiles_for_interval_levels(levels: tuple[float, ...]) -> tuple[float, ...]:
    out: set[float] = {0.5}
    for level in levels:
        q_lo = (1.0 - float(level)) / 2.0
        q_hi = 1.0 - q_lo
        for q in (q_lo, q_hi):
            pct = q * 100.0
            if abs(pct - round(pct)) > 1e-9:
                raise ValueError(
                    "interval_levels for quantile global models must align to integer percentiles"
                )
            out.add(float(q))
    return tuple(sorted(out))


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

    base_forecaster = make_forecaster(str(model), **dict(model_params))
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
        from .models.statsmodels_wrap import (
            sarimax_forecast_with_intervals as _local_xreg_forecast_with_intervals,
        )
    elif str(model) == "auto-arima":
        from .models.statsmodels_wrap import (
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


def _call_local_xreg_forecaster(
    *,
    model: str,
    train_y: np.ndarray,
    horizon: int,
    train_exog: np.ndarray,
    future_exog: np.ndarray,
    model_params: dict[str, Any],
) -> np.ndarray:
    forecaster = make_forecaster(str(model), **dict(model_params))
    try:
        out = forecaster(
            train_y,
            int(horizon),
            train_exog=train_exog,
            future_exog=future_exog,
        )
    except TypeError as e:
        raise ValueError(
            f"Model {model!r} advertises x_cols support but its local callable does not accept "
            "`train_exog` / `future_exog`."
        ) from e

    yhat = np.asarray(out, dtype=float)
    if yhat.shape != (int(horizon),):
        raise ValueError(f"forecaster must return shape ({int(horizon)},), got {yhat.shape}")
    return yhat


def _as_datetime_index(ds: Any) -> pd.DatetimeIndex | None:
    idx = pd.Index(ds)
    if isinstance(idx, pd.DatetimeIndex):
        return idx

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
        raise ValueError("horizon must be >= 1")

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


def _prepare_local_xreg_forecast_group(
    g: pd.DataFrame,
    *,
    horizon: int,
    x_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")

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


def _prepare_global_forecast_input(
    df: pd.DataFrame,
    *,
    horizon: int,
    x_cols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, Any]:
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")

    missing_x_cols = [col for col in x_cols if col not in df.columns]
    if missing_x_cols:
        raise KeyError(f"long_df missing required x_cols: {missing_x_cols}")

    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    cutoffs: list[Any] = []
    future_frames: list[pd.DataFrame] = []
    for uid, g in df.groupby("unique_id", sort=False):
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
        cutoffs.append(g["ds"].iloc[observed_count - 1])
        future = g.iloc[observed_count:]

        if x_cols:
            if len(future) < h:
                raise ValueError(
                    "Global forecast with x_cols requires at least horizon future rows per series"
                )
            missing_future_x = [col for col in x_cols if future.iloc[:h][col].isna().any()]
            if missing_future_x:
                raise ValueError(
                    f"Global forecast future rows are missing required x_cols: {missing_future_x}"
                )
            missing_observed_x = [
                col for col in x_cols if g.iloc[:observed_count][col].isna().any()
            ]
            if missing_observed_x:
                raise ValueError(
                    f"Global forecast observed rows are missing required x_cols: {missing_observed_x}"
                )
            continue

        missing_future = h - int(len(future))
        if missing_future > 0:
            future_frames.append(_future_frame_for_group(g, horizon=missing_future))

    cutoff_by_uid = pd.Series(cutoffs)
    if cutoff_by_uid.nunique() != 1:
        raise ValueError(
            "Global forecast currently requires all series to share the same last observed timestamp"
        )

    if not future_frames:
        return df, cutoff_by_uid.iloc[0]

    augmented = pd.concat([df, *future_frames], axis=0, ignore_index=True, sort=False)
    augmented = augmented.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    return augmented, cutoff_by_uid.iloc[0]


def _finalize_forecast_frame(pred: pd.DataFrame, *, cutoff: Any, model: str) -> pd.DataFrame:
    if not isinstance(pred, pd.DataFrame):
        raise TypeError(f"forecast output must be a pandas DataFrame, got: {type(pred).__name__}")
    required = {"unique_id", "ds", "yhat"}
    missing = required.difference(pred.columns)
    if missing:
        raise KeyError(f"forecast output missing required columns: {sorted(missing)}")

    pred = pred.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True).copy()
    pred_cols = [c for c in pred.columns if c not in {"unique_id", "ds"}]
    pred["cutoff"] = cutoff
    pred["step"] = pred.groupby("unique_id", sort=False).cumcount() + 1
    pred["model"] = str(model)
    ordered = ["unique_id", "ds", "cutoff", "step", *pred_cols, "model"]
    return pred.loc[:, ordered]


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
    model_spec = get_model_spec(str(model))
    interface = str(model_spec.interface).lower().strip()
    capabilities = dict(model_spec.capabilities)
    levels = _parse_interval_levels(interval_levels)

    if interface == "local":
        df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
        historic_x_cols, x_cols = _normalize_covariate_roles(params)
        if historic_x_cols:
            raise ValueError("historic_x_cols are not yet supported in forecast_model_long_df")
        _require_x_cols_if_needed(
            model=str(model),
            capabilities=capabilities,
            x_cols=x_cols,
            context="forecast_model_long_df",
        )
        if x_cols:
            if not bool(capabilities.get("supports_x_cols", False)):
                raise ValueError(
                    f"Model {model!r} does not support x_cols in forecast_model_long_df"
                )
            interval_cols = _interval_column_names(levels)
            rows: list[dict[str, Any]] = []
            local_xreg_params = dict(params)
            local_xreg_params.pop("x_cols", None)
            for uid, g in df.groupby("unique_id", sort=False):
                observed, future, cutoff = _prepare_local_xreg_forecast_group(
                    g,
                    horizon=int(horizon),
                    x_cols=x_cols,
                )
                train_exog = observed.loc[:, list(x_cols)].to_numpy(dtype=float, copy=False)
                future_exog = future.loc[:, list(x_cols)].to_numpy(dtype=float, copy=False)
                if levels:
                    if not bool(capabilities.get("supports_interval_forecast_with_x_cols", False)):
                        raise ValueError(
                            f"Model {model!r} does not support interval_levels with x_cols "
                            "in forecast_model_long_df"
                        )
                    pred_payload = _local_xreg_interval_payload(
                        model=str(model),
                        train_y=observed["y"].to_numpy(dtype=float, copy=False),
                        horizon=int(horizon),
                        train_exog=train_exog,
                        future_exog=future_exog,
                        interval_levels=levels,
                        model_params=local_xreg_params,
                    )
                    yhat = np.asarray(pred_payload["yhat"], dtype=float)
                else:
                    yhat = _call_local_xreg_forecaster(
                        model=str(model),
                        train_y=observed["y"].to_numpy(dtype=float, copy=False),
                        horizon=int(horizon),
                        train_exog=train_exog,
                        future_exog=future_exog,
                        model_params=local_xreg_params,
                    )
                for i in range(int(horizon)):
                    row = {
                        "unique_id": str(uid),
                        "ds": future["ds"].iloc[i],
                        "cutoff": cutoff,
                        "step": int(i + 1),
                        "yhat": float(yhat[i]),
                    }
                    for col in interval_cols:
                        if levels:
                            row[col] = float(pred_payload[col][i])
                    row["model"] = str(model)
                    rows.append(row)
            return pd.DataFrame(
                rows,
                columns=["unique_id", "ds", "cutoff", "step", "yhat", *interval_cols, "model"],
            )

        df = _require_observed_history_only(df)
        interval_cols = _interval_column_names(levels)
        rows: list[dict[str, Any]] = []
        for uid, g in df.groupby("unique_id", sort=False):
            if future_df is not None:
                observed, future, cutoff = _prepare_local_xreg_forecast_group(
                    g,
                    horizon=int(horizon),
                    x_cols=(),
                )
                future_ds = pd.Index(future["ds"])
                train_y = observed["y"].to_numpy(dtype=float, copy=False)
            else:
                cutoff = g["ds"].iloc[-1]
                future_ds = _infer_future_ds(g["ds"], int(horizon))
                train_y = g["y"].to_numpy(dtype=float, copy=False)
            forecaster = make_forecaster_object(str(model), **params).fit(train_y)
            yhat = np.asarray(forecaster.predict(int(horizon)), dtype=float)
            if yhat.shape != (int(horizon),):
                raise ValueError(
                    f"forecaster must return shape ({int(horizon)},), got {yhat.shape}"
                )
            interval_data = _local_interval_columns(
                train_y=np.asarray(train_y, dtype=float),
                model=str(model),
                model_params=params,
                horizon=int(horizon),
                interval_levels=levels,
                interval_min_train_size=interval_min_train_size,
                interval_samples=int(interval_samples),
                interval_seed=interval_seed,
            )

            for i in range(int(horizon)):
                row = {
                    "unique_id": str(uid),
                    "ds": future_ds[i],
                    "cutoff": cutoff,
                    "step": int(i + 1),
                    "yhat": float(yhat[i]),
                }
                for col in interval_cols:
                    row[col] = float(interval_data[col][i])
                row["model"] = str(model)
                rows.append(row)

        return pd.DataFrame(
            rows,
            columns=["unique_id", "ds", "cutoff", "step", "yhat", *interval_cols, "model"],
        )

    if interface == "global":
        historic_x_cols, x_cols = _normalize_covariate_roles(params)
        if historic_x_cols:
            raise ValueError("historic_x_cols are not yet supported in forecast_model_long_df")
        _require_x_cols_if_needed(
            model=str(model),
            capabilities=capabilities,
            x_cols=x_cols,
            context="forecast_model_long_df",
        )
        if x_cols and not bool(capabilities.get("supports_x_cols", False)):
            raise ValueError(f"Model {model!r} does not support x_cols in forecast_model_long_df")
        if levels:
            if not bool(capabilities.get("supports_interval_forecast", False)):
                raise ValueError(
                    f"Model {model!r} does not support interval_levels in forecast_model_long_df"
                )
            params = dict(params)
            params["quantiles"] = _merge_quantiles_for_interval_levels(
                params.get("quantiles"),
                interval_levels=levels,
            )
        augmented, cutoff = _prepare_global_forecast_input(df, horizon=int(horizon), x_cols=x_cols)

        forecaster = make_global_forecaster_object(str(model), **params).fit(augmented)
        pred = forecaster.predict(cutoff, int(horizon))
        pred = _finalize_forecast_frame(pred, cutoff=cutoff, model=str(model))
        return _add_interval_columns_from_quantile_predictions(pred, interval_levels=levels)

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
