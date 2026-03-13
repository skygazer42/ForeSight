from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..base import BaseForecaster, BaseGlobalForecaster
from ..contracts.params import normalize_x_cols as _contracts_normalize_x_cols
from ..data.format import resolve_covariate_roles, to_long
from ..io import ensure_datetime, load_csv
from ..serialization import load_forecaster_artifact, save_forecaster
from . import evaluation as _evaluation
from . import forecasting as _forecasting
from . import model_execution as _model_execution


def _load_csv_frame(
    path: str,
    *,
    parse_dates: bool,
    time_col: str,
) -> pd.DataFrame:
    df = load_csv(str(path))
    if parse_dates:
        ensure_datetime(df, str(time_col))
    return df


def forecast_csv_workflow(
    *,
    model: str,
    path: str,
    time_col: str,
    y_col: str,
    id_cols: tuple[str, ...] = (),
    parse_dates: bool = False,
    horizon: int,
    model_params: dict[str, Any] | None = None,
    future_path: str | None = None,
    interval_levels: Any = None,
    interval_min_train_size: int | None = None,
    interval_samples: int = 1000,
    interval_seed: int | None = None,
    save_artifact_path: str | None = None,
) -> pd.DataFrame:
    params = dict(model_params or {})
    model_key = str(model)
    time_col_s = str(time_col)
    y_col_s = str(y_col)

    df = _load_csv_frame(path, parse_dates=bool(parse_dates), time_col=time_col_s)
    model_spec = _model_execution.get_model_spec(model_key)
    historic_x_cols, future_x_cols, all_x_cols = resolve_covariate_roles(
        x_cols=params.get("x_cols", ()),
        historic_x_cols=params.get("historic_x_cols", ()),
        future_x_cols=params.get("future_x_cols", ()),
    )

    long_df = to_long(
        df,
        time_col=time_col_s,
        y_col=y_col_s,
        id_cols=id_cols,
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
        x_cols=all_x_cols,
        dropna=not (
            model_spec.interface == "global"
            or (model_spec.interface == "local" and future_x_cols)
        ),
    )

    future_df = None
    future_path_s = str(future_path or "").strip()
    if future_path_s:
        future_raw = _load_csv_frame(
            future_path_s,
            parse_dates=bool(parse_dates),
            time_col=time_col_s,
        )
        if y_col_s not in future_raw.columns:
            future_raw[y_col_s] = np.nan
        future_df = to_long(
            future_raw,
            time_col=time_col_s,
            y_col=y_col_s,
            id_cols=id_cols,
            historic_x_cols=historic_x_cols,
            future_x_cols=future_x_cols,
            x_cols=all_x_cols,
            dropna=False,
        )

    pred = _forecasting.forecast_model_long_df(
        model=model_key,
        long_df=long_df,
        future_df=future_df,
        horizon=int(horizon),
        model_params=params,
        interval_levels=interval_levels,
        interval_min_train_size=interval_min_train_size,
        interval_samples=int(interval_samples),
        interval_seed=interval_seed,
    )

    artifact_path = str(save_artifact_path or "").strip()
    if artifact_path:
        if model_spec.interface == "local":
            if future_x_cols:
                raise ValueError(
                    "Saving local forecast artifacts is not yet supported when x_cols are used"
                )
            if int(long_df["unique_id"].nunique()) != 1:
                raise ValueError(
                    "Saving local forecast artifacts currently requires a single series"
                )
            group = next(iter(long_df.groupby("unique_id", sort=False)))[1]
            forecaster = _model_execution.make_local_forecaster_object_runner(
                model_key,
                params,
            ).fit(group["y"].to_numpy(dtype=float, copy=False))
            save_forecaster(
                forecaster,
                artifact_path,
                extra={
                    "artifact_type": "forecast-local",
                    "ds": pd.Index(group["ds"]),
                    "unique_id": str(group["unique_id"].iloc[0]),
                },
            )
        else:
            augmented, cutoff = _forecasting._prepare_global_forecast_input(
                long_df,
                horizon=int(horizon),
                x_cols=future_x_cols,
            )
            forecaster = _model_execution.make_global_forecaster_object_runner(
                model_key,
                params,
            ).fit(augmented)
            save_forecaster(
                forecaster,
                artifact_path,
                extra={
                    "artifact_type": "forecast-global",
                    "cutoff": cutoff,
                    "max_horizon": int(horizon),
                    "model_key": model_key,
                },
            )

    return pred


def forecast_artifact_workflow(
    *,
    artifact: str,
    horizon: int,
    interval_levels: Any = None,
    interval_min_train_size: int | None = None,
    interval_samples: int = 1000,
    interval_seed: int | None = None,
    cutoff: Any = None,
) -> pd.DataFrame:
    payload = load_forecaster_artifact(str(artifact))
    forecaster = payload["forecaster"]
    extra = dict(payload.get("extra", {}))
    horizon_int = int(horizon)

    if isinstance(forecaster, BaseForecaster):
        ds = pd.Index(extra.get("ds", []))
        artifact_type = str(extra.get("artifact_type", "")).strip()
        train_y = getattr(forecaster, "_train_y", None)
        if len(ds) == 0:
            if artifact_type == "forecast-local":
                raise ValueError("Local forecast artifact is missing required ds context")
            if train_y is not None:
                ds = pd.RangeIndex(start=0, stop=int(len(train_y)), step=1)
        if len(ds) == 0:
            raise ValueError("Artifact is missing local forecast ds context")

        future_ds = _forecasting._infer_future_ds(ds, horizon_int)
        yhat = forecaster.predict(horizon_int).astype(float)
        levels = _forecasting._parse_interval_levels(interval_levels)
        interval_cols = _forecasting._interval_column_names(levels)
        pred = pd.DataFrame(
            {
                "unique_id": [str(extra.get("unique_id", "series=0"))] * horizon_int,
                "ds": future_ds,
                "cutoff": [pd.Index(ds)[-1]] * horizon_int,
                "step": list(range(1, horizon_int + 1)),
                "yhat": yhat,
            }
        )
        if levels:
            if train_y is None:
                raise ValueError(
                    "Local artifact is missing training history required for intervals"
                )
            interval_data = _forecasting._local_interval_columns(
                train_y=np.asarray(train_y, dtype=float),
                model=str(forecaster.model_key),
                model_params=dict(forecaster.model_params),
                horizon=horizon_int,
                interval_levels=levels,
                interval_min_train_size=interval_min_train_size,
                interval_samples=int(interval_samples),
                interval_seed=interval_seed,
            )
            for col in interval_cols:
                pred[col] = interval_data[col].astype(float)
        pred["model"] = [str(forecaster.model_key)] * horizon_int
        return pred.loc[
            :,
            ["unique_id", "ds", "cutoff", "step", "yhat", *interval_cols, "model"],
        ]

    if isinstance(forecaster, BaseGlobalForecaster):
        levels = _forecasting._parse_interval_levels(interval_levels)
        if levels:
            spec = _model_execution.get_model_spec(str(forecaster.model_key))
            raise ValueError(
                f"Forecast intervals are not yet supported for artifact model {spec.key!r} "
                "with interface='global'"
            )

        cutoff_value = extra.get("cutoff")
        if cutoff is not None and str(cutoff).strip():
            cutoff_value = pd.to_datetime(str(cutoff).strip(), errors="raise")
        if cutoff_value is None:
            raise ValueError("Global artifact prediction requires a cutoff")

        max_horizon = extra.get("max_horizon")
        if max_horizon is not None and horizon_int > int(max_horizon):
            raise ValueError(
                f"Requested horizon={horizon_int} exceeds artifact max_horizon={int(max_horizon)}"
            )

        pred = forecaster.predict(cutoff_value, horizon_int)
        return _forecasting._finalize_forecast_frame(
            pred,
            cutoff=cutoff_value,
            model=str(forecaster.model_key),
        )

    raise TypeError(f"Unsupported artifact forecaster type: {type(forecaster).__name__}")


def eval_csv_workflow(
    *,
    model: str,
    path: str,
    time_col: str,
    y_col: str,
    id_cols: tuple[str, ...] = (),
    parse_dates: bool = False,
    horizon: int,
    step: int,
    min_train_size: int,
    model_params: dict[str, Any] | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
    conformal_levels: Any = None,
    conformal_per_step: bool = True,
) -> dict[str, Any]:
    params = dict(model_params or {})
    time_col_s = str(time_col)
    y_col_s = str(y_col)
    x_cols = _contracts_normalize_x_cols(params)

    df = _load_csv_frame(path, parse_dates=bool(parse_dates), time_col=time_col_s)
    long_df = to_long(
        df,
        time_col=time_col_s,
        y_col=y_col_s,
        id_cols=id_cols,
        x_cols=x_cols,
        dropna=True,
    )

    payload = _evaluation.eval_model_long_df(
        model=str(model),
        long_df=long_df,
        horizon=int(horizon),
        step=int(step),
        min_train_size=int(min_train_size),
        max_windows=max_windows,
        max_train_size=max_train_size,
        conformal_levels=conformal_levels,
        conformal_per_step=bool(conformal_per_step),
        model_params=params,
    )
    payload.update(
        {
            "dataset": str(path),
            "time_col": time_col_s,
            "y_col": y_col_s,
            "id_cols": list(id_cols),
        }
    )
    return payload


__all__ = [
    "eval_csv_workflow",
    "forecast_artifact_workflow",
    "forecast_csv_workflow",
]
