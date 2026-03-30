from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cli_runtime import compact_log_payload, emit_cli_event
from ..conformal import summarize_conformal_predictions
from ..contracts.capabilities import require_x_cols_if_needed as _contracts_require_x_cols_if_needed
from ..contracts.frames import require_long_df as _contracts_require_long_df
from ..contracts.params import (
    normalize_covariate_roles as _contracts_normalize_covariate_roles,
)
from ..contracts.params import (
    normalize_static_cols as _contracts_normalize_static_cols,
)
from ..dataset_long_df_cache import get_or_build_dataset_long_df
from ..hierarchical import check_hierarchical_consistency, reconcile_hierarchical_forecasts
from ..long_df_cache import (
    cached_series_slices,
    cached_split_sequence,
    cached_x_matrix,
    cached_y_array,
    sorted_long_df,
)
from ..metrics import mae, mape, rmse, smape
from ..splits import rolling_origin_split_sequence
from . import model_execution as _model_execution

_MAX_WINDOWS_MIN_ERROR = "max_windows must be >= 1"


def _require_long_df(long_df: Any) -> pd.DataFrame:
    return _contracts_require_long_df(long_df, require_non_empty=False)


def _parse_levels(levels: Any) -> tuple[float, ...]:
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
        f = float(it)
        if f >= 1.0:
            f = f / 100.0  # allow 80 -> 0.8
        if not (0.0 < f < 1.0):
            raise ValueError("conformal_levels must be in (0,1) or percentages like 80,90")
        out.append(f)
    return tuple(sorted(set(out)))


def _normalize_static_cols(model_params: dict[str, Any] | None) -> tuple[str, ...]:
    return _contracts_normalize_static_cols(model_params or {})


def _normalize_covariate_roles(
    model_params: dict[str, Any] | None,
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


_call_local_xreg_forecaster = _model_execution.call_local_xreg_forecaster


def _require_multivariate_df(
    df: Any,
    *,
    target_cols: Any,
    ds_col: str | None = "ds",
) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if isinstance(target_cols, str):
        targets = [p.strip() for p in target_cols.split(",") if p.strip()]
    elif isinstance(target_cols, list | tuple):
        targets = [str(col).strip() for col in target_cols if str(col).strip()]
    else:
        raise TypeError("target_cols must be a string or sequence of column names")

    if len(targets) < 2:
        raise ValueError("target_cols must contain at least 2 target columns")

    missing = [col for col in targets if col not in df.columns]
    if missing:
        raise KeyError(f"df missing required target columns: {missing}")

    ds_col_final = None if ds_col is None else str(ds_col)
    if ds_col_final:
        if ds_col_final not in df.columns:
            raise KeyError(f"df missing time column: {ds_col_final!r}")
        out = df.sort_values(ds_col_final, kind="mergesort").reset_index(drop=True)
    else:
        out = df.reset_index(drop=True)
    return out, targets


def _walk_forward_multivariate(
    y: np.ndarray,
    *,
    horizon: int,
    step: int,
    min_train_size: int,
    max_train_size: int | None,
    max_windows: int | None,
    forecaster: Any,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(y, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D multivariate series, got shape {matrix.shape}")

    splits = rolling_origin_split_sequence(
        matrix.shape[0],
        horizon=int(horizon),
        step_size=int(step),
        min_train_size=int(min_train_size),
        max_train_size=max_train_size,
        limit=max_windows,
        keep="first",
        limit_error=_MAX_WINDOWS_MIN_ERROR,
    )
    if not splits:
        raise ValueError("No windows available for the requested multivariate backtest parameters.")

    n_windows = len(splits)
    n_targets = int(matrix.shape[1])
    y_true_arr = np.empty((n_windows, int(horizon), n_targets), dtype=float)
    y_pred_arr = np.empty((n_windows, int(horizon), n_targets), dtype=float)

    for idx, split in enumerate(splits):
        train = matrix[split.train_start : split.train_end, :]
        true = matrix[split.test_start : split.test_end, :]
        pred = np.asarray(forecaster(train, int(horizon)), dtype=float)
        expected_shape = (int(horizon), matrix.shape[1])
        if pred.shape != expected_shape:
            raise ValueError(
                f"multivariate forecaster must return shape {expected_shape}, got {pred.shape}"
            )

        y_true_arr[idx, :, :] = true
        y_pred_arr[idx, :, :] = pred

    return y_true_arr, y_pred_arr


def eval_multivariate_model_df(
    *,
    model: str,
    df: Any,
    target_cols: Any,
    horizon: int,
    step: int,
    min_train_size: int,
    ds_col: str | None = "ds",
    model_params: dict[str, Any] | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate a registered multivariate forecaster on a wide DataFrame.

    The input must contain one row per timestamp and two or more target columns.
    """
    wide_df, targets = _require_multivariate_df(df, target_cols=target_cols, ds_col=ds_col)
    if wide_df.empty:
        raise ValueError("df is empty")

    model_spec = _model_execution.get_model_spec(str(model))
    interface = str(model_spec.interface).lower().strip()
    if interface != "multivariate":
        raise ValueError(
            f"Model {model!r} uses interface={model_spec.interface!r} (not 'multivariate')."
        )

    matrix = wide_df.loc[:, targets].to_numpy(dtype=float, copy=False)
    y_true, y_pred = _walk_forward_multivariate(
        matrix,
        horizon=int(horizon),
        step=int(step),
        min_train_size=int(min_train_size),
        max_train_size=max_train_size,
        max_windows=max_windows,
        forecaster=_model_execution.make_multivariate_forecaster_runner(
            str(model),
            model_params,
        ),
    )

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    target_metrics: dict[str, Any] = {}
    for idx, target in enumerate(targets):
        target_true = y_true[:, :, idx]
        target_pred = y_pred[:, :, idx]
        target_metrics[str(target)] = {
            "n_points": int(target_true.size),
            "mae": mae(target_true, target_pred),
            "rmse": rmse(target_true, target_pred),
            "mape": mape(target_true, target_pred),
            "smape": smape(target_true, target_pred),
            "mae_by_step": [
                mae(target_true[:, step_idx], target_pred[:, step_idx])
                for step_idx in range(int(horizon))
            ],
            "rmse_by_step": [
                rmse(target_true[:, step_idx], target_pred[:, step_idx])
                for step_idx in range(int(horizon))
            ],
            "mape_by_step": [
                mape(target_true[:, step_idx], target_pred[:, step_idx])
                for step_idx in range(int(horizon))
            ],
            "smape_by_step": [
                smape(target_true[:, step_idx], target_pred[:, step_idx])
                for step_idx in range(int(horizon))
            ],
        }

    return {
        "model": str(model),
        "horizon": int(horizon),
        "step": int(step),
        "min_train_size": int(min_train_size),
        "max_windows": None if max_windows is None else int(max_windows),
        "max_train_size": None if max_train_size is None else int(max_train_size),
        "n_targets": int(len(targets)),
        "target_cols": list(targets),
        "n_windows": int(y_true.shape[0]),
        "n_points": int(yt.size),
        "mae": mae(yt, yp),
        "rmse": rmse(yt, yp),
        "mape": mape(yt, yp),
        "smape": smape(yt, yp),
        "mae_by_step": [
            mae(y_true[:, step_idx, :], y_pred[:, step_idx, :]) for step_idx in range(int(horizon))
        ],
        "rmse_by_step": [
            rmse(y_true[:, step_idx, :], y_pred[:, step_idx, :]) for step_idx in range(int(horizon))
        ],
        "mape_by_step": [
            mape(y_true[:, step_idx, :], y_pred[:, step_idx, :]) for step_idx in range(int(horizon))
        ],
        "smape_by_step": [
            smape(y_true[:, step_idx, :], y_pred[:, step_idx, :])
            for step_idx in range(int(horizon))
        ],
        "target_metrics": target_metrics,
    }


def eval_hierarchical_forecast_df(
    *,
    forecast_df: Any,
    hierarchy: Any,
    method: str,
    history_df: Any = None,
    yhat_col: str = "yhat",
    exog_agg: Any = None,
) -> dict[str, Any]:
    """
    Reconcile a forecast table to a hierarchy and report consistency summary.

    If the reconciled table also includes a `y` column, standard point metrics are
    included as well.
    """
    from ..eval_predictions import evaluate_predictions

    reconciled = reconcile_hierarchical_forecasts(
        forecast_df=forecast_df,
        hierarchy=hierarchy,
        method=str(method),
        history_df=history_df,
        yhat_col=str(yhat_col),
        exog_agg=exog_agg,
    )
    consistency = check_hierarchical_consistency(
        reconciled,
        hierarchy=hierarchy,
        yhat_col=str(yhat_col),
    )

    out: dict[str, Any] = {
        "method": str(method).strip().lower(),
        "n_rows": int(len(reconciled)),
        "n_series": int(reconciled["unique_id"].nunique()),
        "n_timestamps": int(reconciled["ds"].nunique()),
        "is_consistent": bool(consistency["is_consistent"]),
        "n_inconsistencies": int(consistency["n_inconsistencies"]),
        "forecast_df": reconciled,
    }

    if "y" in reconciled.columns:
        out.update(evaluate_predictions(reconciled, y_col="y", yhat_col=str(yhat_col)))

    return out


def _eval_global_model_long_df(
    *,
    model: str,
    df: pd.DataFrame,
    horizon: int,
    step: int,
    min_train_size: int,
    model_params: dict[str, Any] | None,
    max_windows: int | None,
    max_train_size: int | None,
    conformal_levels: Any,
    conformal_per_step: bool,
) -> dict[str, Any]:
    from ..cv import cross_validation_predictions_long_df
    from ..eval_predictions import evaluate_predictions

    pred_df = cross_validation_predictions_long_df(
        model=str(model),
        long_df=df,
        horizon=int(horizon),
        step_size=int(step),
        min_train_size=int(min_train_size),
        model_params=model_params,
        max_train_size=max_train_size,
        n_windows=max_windows,
    )

    metrics_payload = evaluate_predictions(pred_df)
    out = _global_eval_metrics_payload(
        pred_df,
        metrics_payload=metrics_payload,
        model=model,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        max_windows=max_windows,
        max_train_size=max_train_size,
    )
    _update_global_eval_quantile_payload(out, pred_df)
    _update_global_eval_conformal_payload(
        out,
        pred_df,
        conformal_levels=conformal_levels,
        conformal_per_step=conformal_per_step,
    )
    emit_cli_event(
        "EVAL global",
        event="eval_global_completed",
        payload=compact_log_payload(
            model=str(model),
            n_points=out["n_points"],
            n_series=out["n_series"],
            n_series_skipped=out["n_series_skipped"],
        ),
    )

    return out


def _global_eval_metrics_payload(
    pred_df: pd.DataFrame,
    *,
    metrics_payload: dict[str, Any],
    model: str,
    horizon: int,
    step: int,
    min_train_size: int,
    max_windows: int | None,
    max_train_size: int | None,
) -> dict[str, Any]:
    return {
        "model": str(model),
        "horizon": int(horizon),
        "step": int(step),
        "min_train_size": int(min_train_size),
        "max_windows": None if max_windows is None else int(max_windows),
        "max_train_size": None if max_train_size is None else int(max_train_size),
        "n_series": int(pred_df.attrs.get("n_series", pred_df["unique_id"].nunique())),
        "n_series_skipped": int(pred_df.attrs.get("n_series_skipped", 0)),
        "n_windows": int(pred_df.attrs.get("n_windows", pred_df["cutoff"].nunique())),
        "n_points": int(metrics_payload["n_points"]),
        "mae": float(metrics_payload["mae"]),
        "rmse": float(metrics_payload["rmse"]),
        "mape": float(metrics_payload["mape"]),
        "smape": float(metrics_payload["smape"]),
        "mae_by_step": list(metrics_payload.get("mae_by_step", [])),
        "rmse_by_step": list(metrics_payload.get("rmse_by_step", [])),
        "mape_by_step": list(metrics_payload.get("mape_by_step", [])),
        "smape_by_step": list(metrics_payload.get("smape_by_step", [])),
    }


def _update_global_eval_quantile_payload(
    out: dict[str, Any],
    pred_df: pd.DataFrame,
) -> None:
    from ..eval_predictions import evaluate_quantile_predictions

    q_payload = evaluate_quantile_predictions(pred_df)
    if q_payload.get("quantiles"):
        for k, v in q_payload.items():
            out[f"q_{k}"] = v


def _update_global_eval_conformal_payload(
    out: dict[str, Any],
    pred_df: pd.DataFrame,
    *,
    conformal_levels: Any,
    conformal_per_step: bool,
) -> None:
    levels = _parse_levels(conformal_levels)
    if levels:
        out.update(
            summarize_conformal_predictions(
                pred_df,
                y_col="y",
                yhat_col="yhat",
                step_col="step",
                levels=levels,
                per_step=bool(conformal_per_step),
            )
        )


def _append_eval_window_results(
    results: dict[str, Any],
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    _update_eval_metric_state(results["metric_state"], y_true=y_true, y_pred=y_pred)

    if not bool(results.get("collect_raw_arrays", False)):
        return

    if y_true.ndim == 1:
        for i in range(y_true.size):
            results["y_true_by_step"][i].append(y_true[i : i + 1])
            results["y_pred_by_step"][i].append(y_pred[i : i + 1])
        return

    for i in range(y_true.shape[1]):
        results["y_true_by_step"][i].append(y_true[:, i])
        results["y_pred_by_step"][i].append(y_pred[:, i])


def _new_eval_metric_state(*, horizon: int) -> dict[str, Any]:
    horizon_i = int(horizon)
    return {
        "n_points": 0,
        "abs_error_sum": 0.0,
        "sq_error_sum": 0.0,
        "abs_pct_error_sum": 0.0,
        "smape_sum": 0.0,
        "n_points_by_step": [0] * horizon_i,
        "abs_error_sum_by_step": [0.0] * horizon_i,
        "sq_error_sum_by_step": [0.0] * horizon_i,
        "abs_pct_error_sum_by_step": [0.0] * horizon_i,
        "smape_sum_by_step": [0.0] * horizon_i,
    }


def _update_eval_metric_state(
    metric_state: dict[str, Any],
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> None:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true{yt.shape} vs y_pred{yp.shape}")

    abs_err = np.abs(yt - yp)
    sq_err = (yt - yp) ** 2
    pct_denom = np.where(np.abs(yt) < eps, eps, np.abs(yt))
    abs_pct_err = abs_err / pct_denom
    smape_denom = np.abs(yt) + np.abs(yp)
    smape_denom = np.where(smape_denom < eps, eps, smape_denom)
    smape_terms = 2.0 * abs_err / smape_denom

    metric_state["n_points"] += int(abs_err.size)
    metric_state["abs_error_sum"] += float(abs_err.sum())
    metric_state["sq_error_sum"] += float(sq_err.sum())
    metric_state["abs_pct_error_sum"] += float(abs_pct_err.sum())
    metric_state["smape_sum"] += float(smape_terms.sum())

    if yt.ndim == 1:
        for idx in range(int(yt.size)):
            metric_state["n_points_by_step"][idx] += 1
            metric_state["abs_error_sum_by_step"][idx] += float(abs_err[idx])
            metric_state["sq_error_sum_by_step"][idx] += float(sq_err[idx])
            metric_state["abs_pct_error_sum_by_step"][idx] += float(abs_pct_err[idx])
            metric_state["smape_sum_by_step"][idx] += float(smape_terms[idx])
        return

    for idx in range(int(yt.shape[1])):
        metric_state["n_points_by_step"][idx] += int(yt[:, idx].size)
        metric_state["abs_error_sum_by_step"][idx] += float(abs_err[:, idx].sum())
        metric_state["sq_error_sum_by_step"][idx] += float(sq_err[:, idx].sum())
        metric_state["abs_pct_error_sum_by_step"][idx] += float(abs_pct_err[:, idx].sum())
        metric_state["smape_sum_by_step"][idx] += float(smape_terms[:, idx].sum())


def _metric_state_mean(total: float, count: int) -> float:
    if int(count) <= 0:
        raise ValueError("count must be >= 1")
    return float(total / float(count))


def _metric_state_payload(metric_state: dict[str, Any]) -> dict[str, Any]:
    n_points = int(metric_state["n_points"])
    if n_points <= 0:
        raise ValueError("No evaluated points are available.")

    n_points_by_step = [int(item) for item in metric_state["n_points_by_step"]]
    return {
        "n_points": n_points,
        "mae": _metric_state_mean(float(metric_state["abs_error_sum"]), n_points),
        "rmse": float(np.sqrt(_metric_state_mean(float(metric_state["sq_error_sum"]), n_points))),
        "mape": _metric_state_mean(float(metric_state["abs_pct_error_sum"]), n_points),
        "smape": _metric_state_mean(float(metric_state["smape_sum"]), n_points),
        "mae_by_step": [
            _metric_state_mean(float(total), count)
            for total, count in zip(
                metric_state["abs_error_sum_by_step"],
                n_points_by_step,
                strict=True,
            )
        ],
        "rmse_by_step": [
            float(np.sqrt(_metric_state_mean(float(total), count)))
            for total, count in zip(
                metric_state["sq_error_sum_by_step"],
                n_points_by_step,
                strict=True,
            )
        ],
        "mape_by_step": [
            _metric_state_mean(float(total), count)
            for total, count in zip(
                metric_state["abs_pct_error_sum_by_step"],
                n_points_by_step,
                strict=True,
            )
        ],
        "smape_by_step": [
            _metric_state_mean(float(total), count)
            for total, count in zip(
                metric_state["smape_sum_by_step"],
                n_points_by_step,
                strict=True,
            )
        ],
    }


def _validated_eval_long_df_request(
    *,
    model: str,
    long_df: Any,
    model_params: dict[str, Any] | None,
) -> tuple[pd.DataFrame, str, dict[str, Any], dict[str, Any], tuple[str, ...]]:
    df = _require_long_df(long_df)
    if df.empty:
        raise ValueError("long_df is empty")

    model_spec = _model_execution.get_model_spec(str(model))
    interface = str(model_spec.interface).lower().strip()
    params = dict(model_params or {})
    capabilities = dict(model_spec.capabilities)
    historic_x_cols, x_cols = _normalize_covariate_roles(params)
    static_cols = _normalize_static_cols(params)

    _require_x_cols_if_needed(
        model=str(model),
        capabilities=capabilities,
        x_cols=x_cols,
        context="eval_model_long_df",
    )
    if historic_x_cols:
        raise ValueError("historic_x_cols are not yet supported in eval_model_long_df")
    if static_cols:
        if not bool(capabilities.get("supports_static_cols", False)):
            raise ValueError(f"Model {model!r} does not support static_cols in eval_model_long_df")
        if interface != "global":
            raise ValueError("static_cols are not yet supported for local models in eval_model_long_df")
    if interface == "multivariate":
        raise ValueError(
            f"Model {model!r} is multivariate and cannot be evaluated with `eval_model_long_df()`. "
            "Use `eval_multivariate_model_df()` with a wide DataFrame and explicit target columns instead."
        )
    if interface != "global":
        df = sorted_long_df(df, reset_index=True)
    return df, interface, params, capabilities, x_cols


def _local_eval_model_long_df_results(
    *,
    model: str,
    df: pd.DataFrame,
    horizon: int,
    step: int,
    min_train_size: int,
    params: dict[str, Any],
    capabilities: dict[str, Any],
    x_cols: tuple[str, ...],
    max_windows: int | None,
    max_train_size: int | None,
    collect_raw_arrays: bool,
) -> dict[str, Any]:
    if x_cols:
        if not bool(capabilities.get("supports_x_cols", False)):
            raise ValueError(f"Model {model!r} does not support x_cols in eval_model_long_df")
        return _eval_local_xreg_model_long_df(
            model=str(model),
            df=df,
            horizon=int(horizon),
            step=int(step),
            min_train_size=int(min_train_size),
            params=params,
            x_cols=x_cols,
            max_windows=max_windows,
            max_train_size=max_train_size,
            collect_raw_arrays=collect_raw_arrays,
        )

    return _eval_local_univariate_model_long_df(
        model=str(model),
        df=df,
        horizon=int(horizon),
        step=int(step),
        min_train_size=int(min_train_size),
        params=params,
        max_windows=max_windows,
        max_train_size=max_train_size,
        collect_raw_arrays=collect_raw_arrays,
    )


def _local_xreg_eval_arrays(
    df: pd.DataFrame,
    *,
    x_cols: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    y_all = cached_y_array(df)
    x_all = cached_x_matrix(df, x_cols=x_cols)
    if np.isnan(y_all).any():
        raise ValueError("eval_model_long_df does not support missing y values")
    if np.isnan(x_all).any():
        raise ValueError("eval_model_long_df does not support missing x_cols values")
    return y_all, x_all


def _eval_local_xreg_model_long_df(
    *,
    model: str,
    df: pd.DataFrame,
    horizon: int,
    step: int,
    min_train_size: int,
    params: dict[str, Any],
    x_cols: tuple[str, ...],
    max_windows: int | None,
    max_train_size: int | None,
    collect_raw_arrays: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "metric_state": _new_eval_metric_state(horizon=int(horizon)),
        "collect_raw_arrays": bool(collect_raw_arrays),
        "y_true_by_step": [[] for _ in range(int(horizon))] if collect_raw_arrays else None,
        "y_pred_by_step": [[] for _ in range(int(horizon))] if collect_raw_arrays else None,
        "n_series": 0,
        "n_series_skipped": 0,
        "n_windows": 0,
    }
    local_xreg_params = dict(params)
    local_xreg_params.pop("x_cols", None)
    min_required = int(min_train_size) + int(horizon)
    series_slices = cached_series_slices(df)
    y_all, x_all = _local_xreg_eval_arrays(df, x_cols=x_cols)

    for uid, start, stop in series_slices:
        results["n_series"] += 1
        y = y_all[start:stop]
        x = x_all[start:stop, :]
        if y.size < min_required:
            results["n_series_skipped"] += 1
            emit_cli_event(
                "EVAL skip",
                event="eval_series_skipped",
                payload=compact_log_payload(unique_id=str(uid)),
                progress=True,
            )
            continue

        splits = cached_split_sequence(
            df,
            namespace="eval_local",
            n_obs=int(y.size),
            horizon=int(horizon),
            step_size=int(step),
            min_train_size=int(min_train_size),
            max_train_size=max_train_size,
            limit=max_windows,
            keep="first",
            limit_error=_MAX_WINDOWS_MIN_ERROR,
        )
        n_windows = len(splits)
        y_true_arr = np.empty((n_windows, int(horizon)), dtype=float)
        y_pred_arr = np.empty((n_windows, int(horizon)), dtype=float)
        windows_run = 0
        for idx, split in enumerate(splits):
            pred = _call_local_xreg_forecaster(
                model=str(model),
                train_y=y[split.train_start : split.train_end],
                horizon=int(horizon),
                train_exog=x[split.train_start : split.train_end, :],
                future_exog=x[split.test_start : split.test_end, :],
                model_params=local_xreg_params,
            )
            y_true_arr[idx, :] = y[split.test_start : split.test_end]
            y_pred_arr[idx, :] = pred

            windows_run += 1
            results["n_windows"] += 1
        _append_eval_window_results(results, y_true=y_true_arr, y_pred=y_pred_arr)
        emit_cli_event(
            "EVAL series",
            event="eval_series_completed",
            payload=compact_log_payload(unique_id=str(uid), windows=int(windows_run)),
            progress=True,
        )

    return results


def _eval_local_univariate_model_long_df(
    *,
    model: str,
    df: pd.DataFrame,
    horizon: int,
    step: int,
    min_train_size: int,
    params: dict[str, Any],
    max_windows: int | None,
    max_train_size: int | None,
    collect_raw_arrays: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "metric_state": _new_eval_metric_state(horizon=int(horizon)),
        "collect_raw_arrays": bool(collect_raw_arrays),
        "y_true_by_step": [[] for _ in range(int(horizon))] if collect_raw_arrays else None,
        "y_pred_by_step": [[] for _ in range(int(horizon))] if collect_raw_arrays else None,
        "n_series": 0,
        "n_series_skipped": 0,
        "n_windows": 0,
    }
    forecaster = _model_execution.make_local_forecaster_runner(str(model), params)
    min_required = int(min_train_size) + int(horizon)
    series_slices = cached_series_slices(df)
    y_all = cached_y_array(df)

    for uid, start, stop in series_slices:
        results["n_series"] += 1
        y = y_all[start:stop]
        if y.size < min_required:
            results["n_series_skipped"] += 1
            emit_cli_event(
                "EVAL skip",
                event="eval_series_skipped",
                payload=compact_log_payload(unique_id=str(uid)),
                progress=True,
            )
            continue

        splits = cached_split_sequence(
            df,
            namespace="eval_local",
            n_obs=int(y.size),
            horizon=int(horizon),
            step_size=int(step),
            min_train_size=int(min_train_size),
            max_train_size=max_train_size,
            limit=max_windows,
            keep="first",
            limit_error=_MAX_WINDOWS_MIN_ERROR,
        )
        n_windows = len(splits)
        y_true_arr = np.empty((n_windows, int(horizon)), dtype=float)
        y_pred_arr = np.empty((n_windows, int(horizon)), dtype=float)
        for idx, split in enumerate(splits):
            train = y[split.train_start : split.train_end]
            pred = np.asarray(forecaster(train, int(horizon)), dtype=float)
            if pred.shape != (int(horizon),):
                raise ValueError(f"forecaster must return shape ({int(horizon)},), got {pred.shape}")
            y_true_arr[idx, :] = y[split.test_start : split.test_end]
            y_pred_arr[idx, :] = pred
            results["n_windows"] += 1

        _append_eval_window_results(
            results,
            y_true=y_true_arr,
            y_pred=y_pred_arr,
        )
        emit_cli_event(
            "EVAL series",
            event="eval_series_completed",
            payload=compact_log_payload(
                unique_id=str(uid),
                windows=int(len(splits)),
            ),
            progress=True,
        )

    return results


def _summarize_eval_model_long_df_results(
    *,
    model: str,
    horizon: int,
    step: int,
    min_train_size: int,
    max_windows: int | None,
    max_train_size: int | None,
    conformal_levels: Any,
    conformal_per_step: bool,
    results: dict[str, Any],
) -> dict[str, Any]:
    metric_payload = _metric_state_payload(results["metric_state"])

    out: dict[str, Any] = {
        "model": str(model),
        "horizon": int(horizon),
        "step": int(step),
        "min_train_size": int(min_train_size),
        "max_windows": None if max_windows is None else int(max_windows),
        "max_train_size": None if max_train_size is None else int(max_train_size),
        "n_series": int(results["n_series"]),
        "n_series_skipped": int(results["n_series_skipped"]),
        "n_windows": int(results["n_windows"]),
        "n_points": int(metric_payload["n_points"]),
        "mae": float(metric_payload["mae"]),
        "rmse": float(metric_payload["rmse"]),
        "mape": float(metric_payload["mape"]),
        "smape": float(metric_payload["smape"]),
        "mae_by_step": list(metric_payload["mae_by_step"]),
        "rmse_by_step": list(metric_payload["rmse_by_step"]),
        "mape_by_step": list(metric_payload["mape_by_step"]),
        "smape_by_step": list(metric_payload["smape_by_step"]),
    }

    levels = _parse_levels(conformal_levels)
    if levels:
        if not bool(results.get("collect_raw_arrays", False)):
            raise ValueError("Conformal summaries require raw per-step evaluation arrays.")
        conf_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "y": np.concatenate(results["y_true_by_step"][i]),
                        "yhat": np.concatenate(results["y_pred_by_step"][i]),
                        "step": int(i + 1),
                    }
                )
                for i in range(int(horizon))
            ],
            axis=0,
            ignore_index=True,
        )
        out.update(
            summarize_conformal_predictions(
                conf_df,
                y_col="y",
                yhat_col="yhat",
                step_col="step",
                levels=levels,
                per_step=bool(conformal_per_step),
            )
        )

    emit_cli_event(
        "EVAL done",
        event="eval_local_completed",
        payload=compact_log_payload(
            model=str(model),
            n_windows=int(out["n_windows"]),
            n_points=int(out["n_points"]),
            n_series=int(out["n_series"]),
            n_series_skipped=int(out["n_series_skipped"]),
        ),
    )
    return out


def eval_model_long_df(
    *,
    model: str,
    long_df: Any,
    horizon: int,
    step: int,
    min_train_size: int,
    model_params: dict[str, Any] | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
    conformal_levels: Any = None,
    conformal_per_step: bool = True,
) -> dict[str, Any]:
    """
    Generic evaluation for any registered model on a canonical long-format DataFrame.

    The input must have columns: unique_id, ds, y.
    """
    df, interface, params, capabilities, x_cols = _validated_eval_long_df_request(
        model=str(model),
        long_df=long_df,
        model_params=model_params,
    )

    if interface == "global":
        return _eval_global_model_long_df(
            model=str(model),
            df=df,
            horizon=int(horizon),
            step=int(step),
            min_train_size=int(min_train_size),
            model_params=model_params,
            max_windows=max_windows,
            max_train_size=max_train_size,
            conformal_levels=conformal_levels,
            conformal_per_step=bool(conformal_per_step),
        )

    results = _local_eval_model_long_df_results(
        model=str(model),
        df=df,
        horizon=int(horizon),
        step=int(step),
        min_train_size=int(min_train_size),
        params=params,
        capabilities=capabilities,
        x_cols=x_cols,
        max_windows=max_windows,
        max_train_size=max_train_size,
        collect_raw_arrays=bool(_parse_levels(conformal_levels)),
    )

    return _summarize_eval_model_long_df_results(
        model=str(model),
        horizon=int(horizon),
        step=int(step),
        min_train_size=int(min_train_size),
        max_windows=max_windows,
        max_train_size=max_train_size,
        conformal_levels=conformal_levels,
        conformal_per_step=bool(conformal_per_step),
        results=results,
    )


def eval_model(
    *,
    model: str,
    dataset: str,
    horizon: int,
    step: int,
    min_train_size: int,
    y_col: str | None = None,
    model_params: dict[str, Any] | None = None,
    data_dir: str | Path | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
    conformal_levels: Any = None,
    conformal_per_step: bool = True,
) -> dict[str, Any]:
    """
    Generic evaluation for any registered model on a dataset spec (supports panel datasets).

    Data is converted to a canonical long format (unique_id, ds, y), then evaluated
    per-series using walk-forward backtesting and aggregated across series.
    """
    model_spec = _model_execution.get_model_spec(str(model))
    interface = str(model_spec.interface).lower().strip()
    if interface == "multivariate":
        raise ValueError(
            f"Model {model!r} is multivariate and cannot be evaluated with `eval_model()`. "
            "Use `eval_multivariate_model_df()` with a loaded wide DataFrame and explicit target columns instead."
        )

    frame_bundle = get_or_build_dataset_long_df(
        dataset=str(dataset),
        y_col=y_col,
        data_dir=data_dir,
        model_params=model_params,
    )
    long_df = frame_bundle["long_df"]
    y_col_final = str(frame_bundle["y_col_final"])

    payload = eval_model_long_df(
        model=str(model),
        long_df=long_df,
        horizon=int(horizon),
        step=int(step),
        min_train_size=int(min_train_size),
        model_params=model_params,
        max_windows=max_windows,
        max_train_size=max_train_size,
        conformal_levels=conformal_levels,
        conformal_per_step=conformal_per_step,
    )
    payload["dataset"] = str(dataset)
    payload["y_col"] = y_col_final
    return payload


evaluate_dataset = eval_model
evaluate_long_df = eval_model_long_df


__all__ = [
    "eval_hierarchical_forecast_df",
    "eval_model",
    "eval_model_long_df",
    "eval_multivariate_model_df",
    "evaluate_dataset",
    "evaluate_long_df",
]
