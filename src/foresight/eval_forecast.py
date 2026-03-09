from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtesting import walk_forward
from .conformal import summarize_conformal_predictions
from .data.format import to_long
from .datasets.loaders import load_dataset
from .datasets.registry import get_dataset_spec
from .hierarchical import check_hierarchical_consistency, reconcile_hierarchical_forecasts
from .metrics import mae, mape, rmse, smape
from .models.registry import get_model_spec, make_forecaster, make_multivariate_forecaster
from .splits import rolling_origin_splits


def _require_long_df(long_df: Any) -> pd.DataFrame:
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df must be a pandas DataFrame")
    required = {"unique_id", "ds", "y"}
    missing = required.difference(long_df.columns)
    if missing:
        raise KeyError(f"long_df missing required columns: {sorted(missing)}")
    return long_df


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


def _normalize_x_cols(model_params: dict[str, Any] | None) -> tuple[str, ...]:
    raw = (model_params or {}).get("x_cols")
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
    model_params: dict[str, Any] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    params = dict(model_params or {})

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

    future = _normalize(params.get("future_x_cols"))
    legacy = _normalize_x_cols(params)
    if legacy:
        future = tuple([*future, *[c for c in legacy if c not in future]])
    historic = _normalize(params.get("historic_x_cols"))
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

    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []

    for split in rolling_origin_splits(
        matrix.shape[0],
        horizon=int(horizon),
        step_size=int(step),
        min_train_size=int(min_train_size),
        max_train_size=max_train_size,
    ):
        train = matrix[split.train_start : split.train_end, :]
        true = matrix[split.test_start : split.test_end, :]
        pred = np.asarray(forecaster(train, int(horizon)), dtype=float)
        expected_shape = (int(horizon), matrix.shape[1])
        if pred.shape != expected_shape:
            raise ValueError(
                f"multivariate forecaster must return shape {expected_shape}, got {pred.shape}"
            )

        y_true_list.append(true)
        y_pred_list.append(pred)
        if max_windows is not None and len(y_true_list) >= max_windows:
            break

    if not y_true_list:
        raise ValueError("No windows available for the requested multivariate backtest parameters.")

    return np.stack(y_true_list, axis=0), np.stack(y_pred_list, axis=0)


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

    model_spec = get_model_spec(str(model))
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
        forecaster=make_multivariate_forecaster(str(model), **(model_params or {})),
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
    from .eval_predictions import evaluate_predictions

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
    df = _require_long_df(long_df)
    if df.empty:
        raise ValueError("long_df is empty")

    df = df.sort_values(["unique_id", "ds"], kind="mergesort")
    model_spec = get_model_spec(str(model))
    interface = str(model_spec.interface).lower().strip()
    params = dict(model_params or {})
    capabilities = dict(model_spec.capabilities)
    historic_x_cols, x_cols = _normalize_covariate_roles(params)

    _require_x_cols_if_needed(
        model=str(model),
        capabilities=capabilities,
        x_cols=x_cols,
        context="eval_model_long_df",
    )
    if historic_x_cols:
        raise ValueError("historic_x_cols are not yet supported in eval_model_long_df")

    if interface == "multivariate":
        raise ValueError(
            f"Model {model!r} is multivariate and cannot be evaluated with `eval_model_long_df()`. "
            "Use `eval_multivariate_model_df()` with a wide DataFrame and explicit target columns instead."
        )

    if interface == "global":
        from .cv import cross_validation_predictions_long_df
        from .eval_predictions import evaluate_predictions, evaluate_quantile_predictions

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
        out: dict[str, Any] = {
            "model": str(model),
            "horizon": int(horizon),
            "step": int(step),
            "min_train_size": int(min_train_size),
            "max_windows": None if max_windows is None else int(max_windows),
            "max_train_size": None if max_train_size is None else int(max_train_size),
            "n_series": int(pred_df.attrs.get("n_series", pred_df["unique_id"].nunique())),
            "n_series_skipped": int(pred_df.attrs.get("n_series_skipped", 0)),
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

        q_payload = evaluate_quantile_predictions(pred_df)
        if q_payload.get("quantiles"):
            for k, v in q_payload.items():
                out[f"q_{k}"] = v

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

        return out

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_true_by_step: list[list[np.ndarray]] = [[] for _ in range(int(horizon))]
    y_pred_by_step: list[list[np.ndarray]] = [[] for _ in range(int(horizon))]

    n_series = 0
    n_series_skipped = 0
    if x_cols:
        if not bool(capabilities.get("supports_x_cols", False)):
            raise ValueError(f"Model {model!r} does not support x_cols in eval_model_long_df")
        local_xreg_params = dict(params)
        local_xreg_params.pop("x_cols", None)
        for _uid, g in df.groupby("unique_id", sort=False):
            n_series += 1
            missing_x_cols = [col for col in x_cols if col not in g.columns]
            if missing_x_cols:
                raise KeyError(f"long_df missing required x_cols: {missing_x_cols}")

            y = g["y"].to_numpy(dtype=float, copy=False)
            x = g.loc[:, list(x_cols)].to_numpy(dtype=float, copy=False)
            if np.isnan(y).any():
                raise ValueError("eval_model_long_df does not support missing y values")
            if np.isnan(x).any():
                raise ValueError("eval_model_long_df does not support missing x_cols values")

            min_required = int(min_train_size) + int(horizon)
            if y.size < min_required:
                n_series_skipped += 1
                continue

            windows_run = 0
            for split in rolling_origin_splits(
                y.size,
                horizon=int(horizon),
                step_size=int(step),
                min_train_size=int(min_train_size),
                max_train_size=max_train_size,
            ):
                pred = _call_local_xreg_forecaster(
                    model=str(model),
                    train_y=y[split.train_start : split.train_end],
                    horizon=int(horizon),
                    train_exog=x[split.train_start : split.train_end, :],
                    future_exog=x[split.test_start : split.test_end, :],
                    model_params=local_xreg_params,
                )
                y_true = y[split.test_start : split.test_end]
                y_true_all.append(y_true.reshape(-1))
                y_pred_all.append(pred.reshape(-1))
                for i in range(int(horizon)):
                    y_true_by_step[i].append(y_true[i : i + 1])
                    y_pred_by_step[i].append(pred[i : i + 1])

                windows_run += 1
                if max_windows is not None and windows_run >= int(max_windows):
                    break
    else:
        forecaster = make_forecaster(str(model), **params)

        for _uid, g in df.groupby("unique_id", sort=False):
            n_series += 1
            y = g["y"].to_numpy(dtype=float, copy=False)
            min_required = int(min_train_size) + int(horizon)
            if y.size < min_required:
                n_series_skipped += 1
                continue
            res = walk_forward(
                y,
                horizon=int(horizon),
                step=int(step),
                min_train_size=int(min_train_size),
                max_train_size=max_train_size,
                max_windows=max_windows,
                forecaster=forecaster,
            )

            y_true_all.append(res.y_true.reshape(-1))
            y_pred_all.append(res.y_pred.reshape(-1))

            for i in range(res.horizon):
                y_true_by_step[i].append(res.y_true[:, i])
                y_pred_by_step[i].append(res.y_pred[:, i])

    if not y_true_all:
        raise ValueError("No series had enough data for the requested backtest parameters.")

    yt = np.concatenate(y_true_all, axis=0)
    yp = np.concatenate(y_pred_all, axis=0)

    mae_by_step = [
        mae(np.concatenate(t), np.concatenate(p))
        for t, p in zip(y_true_by_step, y_pred_by_step, strict=True)
    ]
    rmse_by_step = [
        rmse(np.concatenate(t), np.concatenate(p))
        for t, p in zip(y_true_by_step, y_pred_by_step, strict=True)
    ]
    mape_by_step = [
        mape(np.concatenate(t), np.concatenate(p))
        for t, p in zip(y_true_by_step, y_pred_by_step, strict=True)
    ]
    smape_by_step = [
        smape(np.concatenate(t), np.concatenate(p))
        for t, p in zip(y_true_by_step, y_pred_by_step, strict=True)
    ]

    out: dict[str, Any] = {
        "model": str(model),
        "horizon": int(horizon),
        "step": int(step),
        "min_train_size": int(min_train_size),
        "max_windows": None if max_windows is None else int(max_windows),
        "max_train_size": None if max_train_size is None else int(max_train_size),
        "n_series": int(n_series),
        "n_series_skipped": int(n_series_skipped),
        "n_points": int(yt.size),
        "mae": mae(yt, yp),
        "rmse": rmse(yt, yp),
        "mape": mape(yt, yp),
        "smape": smape(yt, yp),
        "mae_by_step": mae_by_step,
        "rmse_by_step": rmse_by_step,
        "mape_by_step": mape_by_step,
        "smape_by_step": smape_by_step,
    }

    levels = _parse_levels(conformal_levels)
    if levels:
        conf_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "y": np.concatenate(y_true_by_step[i]),
                        "yhat": np.concatenate(y_pred_by_step[i]),
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

    return out


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
    spec = get_dataset_spec(str(dataset))
    model_spec = get_model_spec(str(model))
    interface = str(model_spec.interface).lower().strip()
    if interface == "multivariate":
        raise ValueError(
            f"Model {model!r} is multivariate and cannot be evaluated with `eval_model()`. "
            "Use `eval_multivariate_model_df()` with a loaded wide DataFrame and explicit target columns instead."
        )

    y_col_final = str(y_col) if (y_col is not None and str(y_col).strip()) else spec.default_y

    df = load_dataset(str(dataset), data_dir=data_dir)
    x_cols = _normalize_x_cols(model_params)

    long_df = to_long(
        df,
        time_col=spec.time_col,
        y_col=y_col_final,
        id_cols=tuple(spec.group_cols),
        x_cols=x_cols,
        dropna=True,
    )
    if long_df.empty:
        raise ValueError("Loaded 0 rows after to_long(dropna=True). Check dataset and y_col.")

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
