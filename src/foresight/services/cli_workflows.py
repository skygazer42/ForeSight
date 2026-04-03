from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..base import BaseForecaster, BaseGlobalForecaster
from ..cli_runtime import compact_log_payload, emit_cli_event
from ..contracts.params import normalize_covariate_roles as _contracts_normalize_covariate_roles
from ..contracts.params import normalize_static_cols as _contracts_normalize_static_cols
from ..cv import cross_validation_predictions_long_df
from ..data.format import to_long
from ..io import ensure_datetime, load_csv
from ..serialization import load_forecaster_artifact, save_forecaster
from . import detection as _detection
from . import evaluation as _evaluation
from . import forecasting as _forecasting
from . import model_execution as _model_execution

_ARTIFACT_DIFF_MISSING = "<missing>"


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


def _normalize_artifact_info_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_artifact_info_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_normalize_artifact_info_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_normalize_artifact_info_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Index):
        return [_normalize_artifact_info_value(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _artifact_forecaster_type(forecaster: Any) -> str:
    if isinstance(forecaster, BaseForecaster):
        return "local"
    if isinstance(forecaster, BaseGlobalForecaster):
        return "global"
    return type(forecaster).__name__


def _extract_artifact_tracking_summary(
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = _normalize_artifact_info_value(dict(metadata))
    tracking: dict[str, Any] = {}
    train_schema = normalized.get("train_schema")
    if not isinstance(train_schema, dict):
        return normalized, tracking
    runtime = train_schema.get("runtime")
    if not isinstance(runtime, dict):
        return normalized, tracking
    tracking_value = runtime.pop("tracking", None)
    if isinstance(tracking_value, dict):
        tracking = dict(tracking_value)
    return normalized, tracking


def _artifact_tracking_backends(tracking: dict[str, Any]) -> list[str]:
    return sorted(
        str(name) for name, payload in tracking.items() if isinstance(payload, dict) and payload
    )


def _artifact_tracking_summary(tracking: dict[str, Any]) -> dict[str, str]:
    summary: dict[str, str] = {}
    for backend in _artifact_tracking_backends(tracking):
        payload = tracking.get(backend, {})
        if not isinstance(payload, dict):
            continue
        if backend == "tensorboard":
            run_name = str(payload.get("run_name", "")).strip()
            log_dir = str(payload.get("log_dir", "")).strip()
            parts = []
            if run_name:
                parts.append(run_name)
            if log_dir:
                parts.append(f"@ {log_dir}" if run_name else log_dir)
            if parts:
                summary[backend] = " ".join(parts)
            continue
        if backend == "mlflow":
            experiment_name = str(payload.get("experiment_name", "")).strip()
            run_name = str(payload.get("run_name", "")).strip()
            items = [item for item in (experiment_name, run_name) if item]
            if items:
                summary[backend] = " / ".join(items)
            continue
        if backend == "wandb":
            project = str(payload.get("project", "")).strip()
            run_name = str(payload.get("run_name", "")).strip()
            mode = str(payload.get("mode", "")).strip()
            items = [item for item in (project, run_name) if item]
            if items:
                text = " / ".join(items)
                if mode:
                    text = f"{text} [{mode}]"
                summary[backend] = text
            continue
        text = ", ".join(
            f"{key}={value}"
            for key, value in payload.items()
            if value not in ("", None, [], {}, ())
        )
        if text:
            summary[backend] = text
    return summary


def _artifact_composition_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    train_schema = metadata.get("train_schema")
    if not isinstance(train_schema, dict):
        return {}
    composition = train_schema.get("composition")
    if not isinstance(composition, dict):
        return {}
    normalized = _normalize_artifact_info_value(dict(composition))
    kind = str(normalized.get("kind", "")).strip()
    if not kind:
        return {}
    return {
        str(key): value for key, value in normalized.items() if value not in ("", None, [], {}, ())
    }


def _artifact_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    forecaster = payload["forecaster"]
    extra = dict(payload.get("extra", {}))
    metadata, tracking = _extract_artifact_tracking_summary(dict(payload["metadata"]))
    summary = {
        "artifact_schema_version": int(payload["artifact_schema_version"]),
        "forecaster_type": _artifact_forecaster_type(forecaster),
        "metadata": metadata,
        "extra": _normalize_artifact_info_value(extra),
    }
    composition_summary = _artifact_composition_summary(metadata)
    if composition_summary:
        summary["composition_summary"] = composition_summary
    future_override_schema = _artifact_future_override_schema(
        forecaster=forecaster,
        extra=extra,
    )
    if future_override_schema is not None:
        summary["future_override_schema"] = _normalize_artifact_info_value(future_override_schema)
    if tracking:
        summary["tracking"] = tracking
        summary["tracking_backends"] = _artifact_tracking_backends(tracking)
        summary["tracking_summary"] = _artifact_tracking_summary(tracking)
    return summary


def _collect_artifact_differences(
    *,
    left: Any,
    right: Any,
    path: str,
    out: dict[str, dict[str, Any]],
) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left).union(right)):
            next_path = str(key) if not path else f"{path}.{key}"
            if key not in left:
                out[next_path] = {"left": _ARTIFACT_DIFF_MISSING, "right": right[key]}
                continue
            if key not in right:
                out[next_path] = {"left": left[key], "right": _ARTIFACT_DIFF_MISSING}
                continue
            _collect_artifact_differences(
                left=left[key],
                right=right[key],
                path=next_path,
                out=out,
            )
        return

    if left != right:
        out[path] = {"left": left, "right": right}


def _flatten_artifact_summary_rows(
    payload: dict[str, Any],
    *,
    path: str = "",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    stack: list[tuple[str, dict[str, Any]]] = [(str(path), payload)]
    while stack:
        current_path, current_payload = stack.pop()
        for key in sorted(current_payload, reverse=True):
            value = current_payload[key]
            next_path = str(key) if not current_path else f"{current_path}.{key}"
            if isinstance(value, dict):
                stack.append((next_path, value))
                continue
            rows.append(
                {
                    "field": str(next_path),
                    "value": _stringify_artifact_diff_value(value),
                }
            )
    return rows


def _stringify_artifact_diff_value(value: Any) -> str:
    if value == _ARTIFACT_DIFF_MISSING:
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _artifact_info_summary_rows(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows = [
        {
            "field": "artifact_schema_version",
            "value": _stringify_artifact_diff_value(payload["artifact_schema_version"]),
        },
        {
            "field": "forecaster_type",
            "value": _stringify_artifact_diff_value(payload["forecaster_type"]),
        },
        {
            "field": "is_fitted",
            "value": _stringify_artifact_diff_value(payload["is_fitted"]),
        },
    ]
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        model_key = str(metadata.get("model_key", "")).strip()
        if model_key:
            rows.append({"field": "model_key", "value": model_key})
        train_schema = metadata.get("train_schema", {})
        if isinstance(train_schema, dict):
            train_kind = str(train_schema.get("kind", "")).strip()
            if train_kind:
                rows.append({"field": "train_kind", "value": train_kind})
    return rows


def _artifact_info_tracking_rows(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    backends = payload.get("tracking_backends", [])
    if isinstance(backends, list) and backends:
        rows.append({"backend": "backends", "summary": ", ".join(str(item) for item in backends)})
    tracking_summary = payload.get("tracking_summary", {})
    if isinstance(tracking_summary, dict):
        for backend in sorted(tracking_summary):
            rows.append(
                {
                    "backend": str(backend),
                    "summary": str(tracking_summary[backend]),
                }
            )
    return rows


def _artifact_future_override_schema(
    *,
    forecaster: Any,
    extra: dict[str, Any],
) -> dict[str, Any] | None:
    artifact_type = str(extra.get("artifact_type", "")).strip()

    if artifact_type == "forecast-local" and isinstance(forecaster, BaseForecaster):
        x_cols = tuple(str(item) for item in extra.get("x_cols", []))
        if not x_cols:
            return {
                "supported": False,
                "artifact_interface": "local",
                "default_context_source": "inferred_from_history",
                "reason": (
                    "future_path override is only supported for local forecast artifacts "
                    "saved with x_cols"
                ),
            }

        schema: dict[str, Any] = {
            "supported": True,
            "artifact_interface": "local",
            "default_context_source": "saved_future_context",
            "requires_time_col": True,
            "required_covariate_columns": list(x_cols),
            "saved_id_cols": [],
            "allow_unique_id_column": False,
            "allow_saved_raw_id_columns": False,
            "allow_omitting_id_columns": True,
        }
        max_horizon = extra.get("max_horizon")
        if max_horizon is not None:
            schema["max_horizon"] = int(max_horizon)
        return schema

    if artifact_type == "forecast-global" and isinstance(forecaster, BaseGlobalForecaster):
        _historic_x_cols, x_cols = _contracts_normalize_covariate_roles(
            dict(forecaster.model_params)
        )
        id_cols = tuple(str(item) for item in extra.get("id_cols", []))
        allow_omitting_ids = False
        train_df = getattr(forecaster, "_train_df", None)
        if train_df is not None:
            try:
                train_df_checked = _forecasting._require_long_df(pd.DataFrame(train_df).copy())
                allow_omitting_ids = int(train_df_checked["unique_id"].nunique()) == 1
            except Exception:  # noqa: BLE001
                allow_omitting_ids = False

        schema = {
            "supported": True,
            "artifact_interface": "global",
            "default_context_source": "saved_future_context",
            "requires_time_col": True,
            "required_covariate_columns": list(x_cols),
            "saved_id_cols": list(id_cols),
            "allow_unique_id_column": True,
            "allow_saved_raw_id_columns": bool(id_cols),
            "allow_omitting_id_columns": bool(allow_omitting_ids),
        }
        max_horizon = extra.get("max_horizon")
        if max_horizon is not None:
            schema["max_horizon"] = int(max_horizon)
        return schema

    return None


def _render_markdown_section(
    *,
    title: str,
    rows: list[dict[str, str]],
    columns: list[str],
) -> str:
    from .. import cli_shared as _cli_shared

    if not rows:
        return ""
    return f"## {title}\n\n{_cli_shared._format_table(rows, columns=columns, fmt='md')}"


def _trim_diff_path(path: str, *, prefix: str) -> str:
    text = str(path)
    if text == prefix:
        return text
    return text[len(prefix) + 1 :] if text.startswith(f"{prefix}.") else text


def _artifact_diff_summary_rows(
    payload: dict[str, Any],
    *,
    path_prefix: str | None,
) -> list[dict[str, str]]:
    rows = [
        {
            "field": "equal",
            "value": _stringify_artifact_diff_value(payload["equal"]),
        },
        {
            "field": "difference_count",
            "value": _stringify_artifact_diff_value(payload["difference_count"]),
        },
    ]
    prefix = str(path_prefix or "").strip()
    if prefix:
        rows.append({"field": "path_prefix", "value": prefix})
    return rows


def _artifact_diff_tracking_summary_rows(
    differences: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if "tracking_backends" in differences:
        item = differences["tracking_backends"]
        rows.append(
            {
                "backend": "backends",
                "left": _stringify_artifact_diff_value(item["left"]),
                "right": _stringify_artifact_diff_value(item["right"]),
            }
        )
    for path in sorted(differences):
        if not path.startswith("tracking_summary."):
            continue
        item = differences[path]
        rows.append(
            {
                "backend": _trim_diff_path(path, prefix="tracking_summary"),
                "left": _stringify_artifact_diff_value(item["left"]),
                "right": _stringify_artifact_diff_value(item["right"]),
            }
        )
    return rows


def _artifact_diff_prefixed_rows(
    differences: dict[str, dict[str, Any]],
    *,
    prefix: str,
    field_name: str = "field",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(differences):
        if path != prefix and not path.startswith(f"{prefix}."):
            continue
        item = differences[path]
        rows.append(
            {
                field_name: _trim_diff_path(path, prefix=prefix),
                "left": _stringify_artifact_diff_value(item["left"]),
                "right": _stringify_artifact_diff_value(item["right"]),
            }
        )
    return rows


def _artifact_diff_other_rows(
    differences: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    excluded_prefixes = (
        "tracking.",
        "tracking_summary.",
        "metadata.",
        "extra.",
        "future_override_schema.",
    )
    excluded_exact = {"tracking_backends", "future_override_schema"}
    for path in sorted(differences):
        if path in excluded_exact or any(path.startswith(prefix) for prefix in excluded_prefixes):
            continue
        item = differences[path]
        rows.append(
            {
                "field": str(path),
                "left": _stringify_artifact_diff_value(item["left"]),
                "right": _stringify_artifact_diff_value(item["right"]),
            }
        )
    return rows


def _filter_artifact_differences(
    differences: dict[str, dict[str, Any]],
    *,
    path_prefix: str | None,
) -> dict[str, dict[str, Any]]:
    prefix = str(path_prefix or "").strip()
    if not prefix:
        return dict(differences)
    return {
        path: value
        for path, value in differences.items()
        if path == prefix or path.startswith(f"{prefix}.")
    }


def _resolve_forecast_covariates(
    params: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    historic_x_cols, future_x_cols = _contracts_normalize_covariate_roles(params)
    static_cols = _contracts_normalize_static_cols(params)
    return historic_x_cols, future_x_cols, static_cols


def _resolve_model_param_covariates(
    params: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return _resolve_forecast_covariates(params)


def _build_long_frame(
    df: pd.DataFrame,
    *,
    time_col: str,
    y_col: str,
    id_cols: tuple[str, ...],
    historic_x_cols: tuple[str, ...],
    future_x_cols: tuple[str, ...],
    static_cols: tuple[str, ...],
    dropna: bool,
) -> pd.DataFrame:
    return to_long(
        df,
        time_col=time_col,
        y_col=y_col,
        id_cols=id_cols,
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
        static_cols=static_cols,
        dropna=dropna,
    )


def _build_future_long_df(
    *,
    future_path: str | None,
    parse_dates: bool,
    time_col: str,
    y_col: str,
    id_cols: tuple[str, ...],
    future_x_cols: tuple[str, ...],
) -> pd.DataFrame | None:
    future_path_s = str(future_path or "").strip()
    if not future_path_s:
        return None
    future_raw = _load_csv_frame(
        future_path_s,
        parse_dates=parse_dates,
        time_col=time_col,
    )
    if y_col not in future_raw.columns:
        future_raw[y_col] = np.nan
    return _build_long_frame(
        future_raw,
        time_col=time_col,
        y_col=y_col,
        id_cols=id_cols,
        historic_x_cols=(),
        future_x_cols=future_x_cols,
        static_cols=(),
        dropna=False,
    )


def _build_forecast_long_frames(
    *,
    model_key: str,
    params: dict[str, Any],
    path: str,
    future_path: str | None,
    parse_dates: bool,
    time_col: str,
    y_col: str,
    id_cols: tuple[str, ...],
) -> tuple[
    Any, pd.DataFrame, pd.DataFrame | None, tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    df = _load_csv_frame(path, parse_dates=parse_dates, time_col=time_col)
    model_spec = _model_execution.get_model_spec(model_key)
    historic_x_cols, future_x_cols, static_cols = _resolve_forecast_covariates(params)
    long_df = _build_long_frame(
        df,
        time_col=time_col,
        y_col=y_col,
        id_cols=id_cols,
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
        static_cols=static_cols,
        dropna=not (
            model_spec.interface == "global" or (model_spec.interface == "local" and future_x_cols)
        ),
    )
    future_df = _build_future_long_df(
        future_path=future_path,
        parse_dates=parse_dates,
        time_col=time_col,
        y_col=y_col,
        id_cols=id_cols,
        future_x_cols=future_x_cols,
    )
    return model_spec, long_df, future_df, historic_x_cols, future_x_cols, static_cols


def _save_local_forecast_artifact(
    *,
    artifact_path: str,
    model_key: str,
    params: dict[str, Any],
    future_x_cols: tuple[str, ...],
    long_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    horizon: int,
) -> None:
    artifact_df = (
        _forecasting._merge_history_and_future_df(long_df, future_df)
        if future_x_cols and future_df is not None
        else long_df
    )
    if int(artifact_df["unique_id"].nunique()) != 1:
        raise ValueError("Saving local forecast artifacts currently requires a single series")
    group = next(iter(artifact_df.groupby("unique_id", sort=False)))[1]
    group = group.sort_values("ds", kind="mergesort").reset_index(drop=True)
    unique_id = str(group["unique_id"].iloc[0])

    extra: dict[str, Any] = {
        "artifact_type": "forecast-local",
        "unique_id": unique_id,
    }
    if future_x_cols:
        observed, future, _ = _forecasting._prepare_local_xreg_forecast_group(
            group,
            horizon=int(horizon),
            x_cols=future_x_cols,
        )
        train_y = observed["y"].to_numpy(dtype=float, copy=False)
        forecaster = _model_execution.make_local_forecaster_object_runner(
            model_key,
            params,
        ).fit(train_y)
        extra.update(
            {
                "ds": pd.Index(observed["ds"]),
                "x_cols": list(future_x_cols),
                "train_exog": observed.loc[:, list(future_x_cols)].to_numpy(dtype=float, copy=True),
                "future_exog": future.loc[:, list(future_x_cols)].to_numpy(dtype=float, copy=True),
                "future_ds": pd.Index(future["ds"]),
                "max_horizon": int(horizon),
            }
        )
    else:
        train_y = group["y"].to_numpy(dtype=float, copy=False)
        forecaster = _model_execution.make_local_forecaster_object_runner(
            model_key,
            params,
        ).fit(train_y)
        extra["ds"] = pd.Index(group["ds"])

    save_forecaster(
        forecaster,
        artifact_path,
        extra=extra,
    )


def _save_global_forecast_artifact(
    *,
    artifact_path: str,
    model_key: str,
    params: dict[str, Any],
    long_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_cols: tuple[str, ...],
    future_x_cols: tuple[str, ...],
    horizon: int,
) -> None:
    artifact_df = (
        _forecasting._merge_history_and_future_df(long_df, future_df)
        if future_df is not None
        else long_df
    )
    augmented, cutoff = _forecasting._prepare_global_forecast_input(
        artifact_df,
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
            "id_cols": list(id_cols),
        },
    )


def _resolve_local_artifact_ds(
    *,
    extra: dict[str, Any],
    train_y: Any,
) -> pd.Index:
    ds = pd.Index(extra.get("ds", []))
    artifact_type = str(extra.get("artifact_type", "")).strip()
    if len(ds) == 0:
        if artifact_type == "forecast-local":
            raise ValueError("Local forecast artifact is missing required ds context")
        if train_y is not None:
            ds = pd.RangeIndex(start=0, stop=int(len(train_y)), step=1)
    if len(ds) == 0:
        raise ValueError("Artifact is missing local forecast ds context")
    return ds


def _resolve_local_artifact_train_exog(
    *,
    extra: dict[str, Any],
    train_y: Any,
) -> np.ndarray:
    train_exog_raw = extra.get("train_exog")
    if train_exog_raw is None:
        raise ValueError("Local forecast artifact is missing required train_exog context")

    train_exog = np.asarray(train_exog_raw, dtype=float)
    if train_exog.ndim != 2:
        raise ValueError("Local forecast artifact train_exog must be a 2D array")

    train_y_arr = np.asarray(train_y, dtype=float)
    if train_exog.shape[0] != train_y_arr.shape[0]:
        raise ValueError("Local forecast artifact train_exog rows do not match training history")
    return train_exog


def _resolve_local_artifact_saved_future_context(
    *,
    extra: dict[str, Any],
    horizon: int,
) -> tuple[np.ndarray, pd.Index, int]:
    future_exog_raw = extra.get("future_exog")
    if future_exog_raw is None:
        raise ValueError("Local forecast artifact is missing required future_exog context")

    future_exog = np.asarray(future_exog_raw, dtype=float)
    if future_exog.ndim != 2:
        raise ValueError("Local forecast artifact future_exog must be a 2D array")
    future_ds = pd.Index(extra.get("future_ds", []))
    if len(future_ds) == 0:
        raise ValueError("Local forecast artifact is missing required future ds context")

    max_horizon = int(extra.get("max_horizon", min(len(future_ds), future_exog.shape[0])))
    if int(horizon) > max_horizon:
        raise ValueError(
            f"Requested horizon={int(horizon)} exceeds artifact max_horizon={max_horizon}"
        )
    if future_exog.shape[0] < int(horizon):
        raise ValueError("Local forecast artifact is missing future_exog rows for the request")
    if len(future_ds) < int(horizon):
        raise ValueError("Local forecast artifact is missing future ds rows for the request")

    return (
        future_exog[: int(horizon), :],
        pd.Index(future_ds[: int(horizon)]),
        max_horizon,
    )


def _load_local_artifact_future_override(
    *,
    path: str,
    time_col: str,
    parse_dates: bool,
    x_cols: tuple[str, ...],
    horizon: int,
) -> tuple[np.ndarray, pd.Index]:
    future_raw = _load_csv_frame(
        str(path),
        parse_dates=bool(parse_dates),
        time_col=str(time_col),
    )
    required = [str(time_col), *list(x_cols)]
    missing = [col for col in required if col not in future_raw.columns]
    if missing:
        raise ValueError(f"Artifact future-path is missing required columns: {missing}")
    if len(future_raw) < int(horizon):
        raise ValueError("Artifact future-path requires at least horizon rows")
    future = future_raw.iloc[: int(horizon)].copy()
    missing_future_x = [col for col in x_cols if future[col].isna().any()]
    if missing_future_x:
        raise ValueError(
            f"Artifact future-path is missing required x_cols values: {missing_future_x}"
        )
    return (
        future.loc[:, list(x_cols)].to_numpy(dtype=float, copy=False),
        pd.Index(future[str(time_col)]),
    )


def _append_local_artifact_intervals(
    pred: pd.DataFrame,
    *,
    forecaster: BaseForecaster,
    train_y: Any,
    levels: tuple[float, ...],
    horizon: int,
    interval_min_train_size: int | None,
    interval_samples: int,
    interval_seed: int | None,
    interval_cols: list[str],
) -> pd.DataFrame:
    if not levels:
        return pred
    if train_y is None:
        raise ValueError("Local artifact is missing training history required for intervals")
    interval_data = _forecasting._local_interval_columns(
        train_y=np.asarray(train_y, dtype=float),
        model=str(forecaster.model_key),
        model_params=dict(forecaster.model_params),
        horizon=horizon,
        interval_levels=levels,
        interval_min_train_size=interval_min_train_size,
        interval_samples=interval_samples,
        interval_seed=interval_seed,
    )
    for col in interval_cols:
        pred[col] = interval_data[col].astype(float)
    return pred


def _forecast_local_artifact(
    forecaster: BaseForecaster,
    *,
    extra: dict[str, Any],
    horizon: int,
    interval_levels: Any,
    interval_min_train_size: int | None,
    interval_samples: int,
    interval_seed: int | None,
    future_path: str | None = None,
    time_col: str | None = None,
    parse_dates: bool = False,
) -> pd.DataFrame:
    train_y = getattr(forecaster, "_train_y", None)
    ds = _resolve_local_artifact_ds(extra=extra, train_y=train_y)
    x_cols = tuple(str(item) for item in extra.get("x_cols", []))
    future_path_s = str(future_path or "").strip()
    if x_cols:
        if train_y is None:
            raise ValueError("Local artifact is missing training history required for x_cols")
        train_exog = _resolve_local_artifact_train_exog(extra=extra, train_y=train_y)
        if future_path_s:
            time_col_s = str(time_col or "").strip()
            if not time_col_s:
                raise ValueError("forecast artifact requires time_col when future_path is provided")
            future_exog, future_ds = _load_local_artifact_future_override(
                path=future_path_s,
                time_col=time_col_s,
                parse_dates=bool(parse_dates),
                x_cols=x_cols,
                horizon=horizon,
            )
        else:
            future_exog, future_ds, _ = _resolve_local_artifact_saved_future_context(
                extra=extra,
                horizon=horizon,
            )
        levels = _forecasting._parse_interval_levels(interval_levels)
        interval_cols = _forecasting._interval_column_names(levels)
        local_xreg_params = dict(forecaster.model_params)
        local_xreg_params.pop("x_cols", None)
        if levels:
            model_spec = _model_execution.get_model_spec(str(forecaster.model_key))
            capabilities = dict(model_spec.capabilities)
            if not bool(capabilities.get("supports_interval_forecast_with_x_cols", False)):
                raise ValueError(
                    f"Model {forecaster.model_key!r} does not support interval_levels with x_cols "
                    "for forecast artifact"
                )
            pred_payload = _forecasting._local_xreg_interval_payload(
                model=str(forecaster.model_key),
                train_y=np.asarray(train_y, dtype=float),
                horizon=int(horizon),
                train_exog=train_exog,
                future_exog=future_exog,
                interval_levels=levels,
                model_params=local_xreg_params,
            )
            yhat = np.asarray(pred_payload["yhat"], dtype=float)
        else:
            pred_payload = {}
            yhat = _forecasting._call_local_xreg_forecaster(
                model=str(forecaster.model_key),
                train_y=np.asarray(train_y, dtype=float),
                horizon=int(horizon),
                train_exog=train_exog,
                future_exog=future_exog,
                model_params=local_xreg_params,
            )

        rows: list[dict[str, Any]] = []
        cutoff = pd.Index(ds)[-1]
        for i in range(int(horizon)):
            interval_values = {col: float(pred_payload[col][i]) for col in interval_cols}
            rows.append(
                {
                    "unique_id": str(extra.get("unique_id", "series=0")),
                    "ds": future_ds[i],
                    "cutoff": cutoff,
                    "step": i + 1,
                    "yhat": float(yhat[i]),
                    **interval_values,
                    "model": str(forecaster.model_key),
                }
            )
        return pd.DataFrame(
            rows,
            columns=["unique_id", "ds", "cutoff", "step", "yhat", *interval_cols, "model"],
        )

    if future_path_s:
        raise ValueError(
            "forecast artifact future_path is only supported for local artifacts saved with x_cols"
        )

    future_ds = _forecasting._infer_future_ds(ds, horizon)
    yhat = forecaster.predict(horizon).astype(float)
    levels = _forecasting._parse_interval_levels(interval_levels)
    interval_cols = _forecasting._interval_column_names(levels)
    pred = pd.DataFrame(
        {
            "unique_id": [str(extra.get("unique_id", "series=0"))] * horizon,
            "ds": future_ds,
            "cutoff": [pd.Index(ds)[-1]] * horizon,
            "step": list(range(1, horizon + 1)),
            "yhat": yhat,
        }
    )
    pred = _append_local_artifact_intervals(
        pred,
        forecaster=forecaster,
        train_y=train_y,
        levels=levels,
        horizon=horizon,
        interval_min_train_size=interval_min_train_size,
        interval_samples=interval_samples,
        interval_seed=interval_seed,
        interval_cols=interval_cols,
    )
    pred["model"] = [str(forecaster.model_key)] * horizon
    return pred.loc[
        :,
        ["unique_id", "ds", "cutoff", "step", "yhat", *interval_cols, "model"],
    ]


def _resolve_global_artifact_cutoff(
    *,
    extra: dict[str, Any],
    cutoff: Any,
) -> Any:
    cutoff_value = extra.get("cutoff")
    if cutoff is not None and str(cutoff).strip():
        cutoff_value = pd.to_datetime(str(cutoff).strip(), errors="raise")
    if cutoff_value is None:
        raise ValueError("Global artifact prediction requires a cutoff")
    return cutoff_value


def _resolve_global_artifact_train_df(
    *,
    forecaster: BaseGlobalForecaster,
) -> pd.DataFrame:
    train_df = getattr(forecaster, "_train_df", None)
    if train_df is None:
        raise ValueError("Global artifact is missing training history required for future_path")
    return _forecasting._require_long_df(pd.DataFrame(train_df).copy())


def _resolve_global_artifact_observed_history(
    *,
    train_df: pd.DataFrame,
    cutoff: Any,
) -> pd.DataFrame:
    observed_frames: list[pd.DataFrame] = []
    for uid, group in train_df.groupby("unique_id", sort=False):
        g = group.sort_values("ds", kind="mergesort").reset_index(drop=True)
        cutoff_idx = pd.Index(g["ds"]).get_indexer([cutoff])[0]
        if int(cutoff_idx) < 0:
            raise ValueError(
                f"Global artifact cutoff {cutoff!r} was not found for unique_id={uid!r}"
            )
        observed = g.iloc[: int(cutoff_idx) + 1].copy()
        observed_frames.append(_forecasting._require_observed_history_only(observed))

    if not observed_frames:
        raise ValueError("Global artifact is missing observed history required for future_path")
    return pd.concat(observed_frames, axis=0, ignore_index=True, sort=False)


def _load_global_artifact_future_override(
    *,
    path: str,
    time_col: str,
    parse_dates: bool,
    id_cols: tuple[str, ...],
    x_cols: tuple[str, ...],
    static_cols: tuple[str, ...],
    observed_history: pd.DataFrame,
) -> pd.DataFrame:
    future_raw = _load_csv_frame(
        str(path),
        parse_dates=bool(parse_dates),
        time_col=str(time_col),
    )
    future: pd.DataFrame
    if "unique_id" in future_raw.columns:
        required = [str(time_col), *list(x_cols)]
        missing = [col for col in required if col not in future_raw.columns]
        if missing:
            raise ValueError(f"Artifact future-path is missing required columns: {missing}")
        future = pd.DataFrame(
            {
                "unique_id": future_raw["unique_id"],
                "ds": future_raw[str(time_col)],
            }
        )
        if "y" in future_raw.columns:
            future["y"] = future_raw["y"]
        for col in x_cols:
            future[col] = future_raw[col]
        for col in static_cols:
            if col in future_raw.columns:
                future[col] = future_raw[col]
    else:
        observed_unique_ids = pd.Index(
            observed_history["unique_id"].dropna().astype("string").unique()
        )
        raw_id_present = [col for col in id_cols if col in future_raw.columns]
        if len(observed_unique_ids) == 1 and not raw_id_present:
            required = [str(time_col), *list(x_cols)]
            missing = [col for col in required if col not in future_raw.columns]
            if missing:
                raise ValueError(
                    "Artifact future-path is missing required columns: "
                    f"{list(dict.fromkeys(['unique_id', *missing]))}"
                )
            future = pd.DataFrame(
                {
                    "unique_id": [str(observed_unique_ids[0])] * len(future_raw),
                    "ds": future_raw[str(time_col)],
                }
            )
            if "y" in future_raw.columns:
                future["y"] = future_raw["y"]
            for col in x_cols:
                future[col] = future_raw[col]
            for col in static_cols:
                if col in future_raw.columns:
                    future[col] = future_raw[col]
        else:
            required = [*list(id_cols), str(time_col), *list(x_cols)]
            missing = [col for col in required if col not in future_raw.columns]
            if missing:
                missing_id_msg = (
                    list(dict.fromkeys(["unique_id", *missing]))
                    if not id_cols
                    else ["unique_id", *list(id_cols)]
                )
                raise ValueError(
                    "Artifact future-path is missing required columns: "
                    f"{missing if id_cols else missing_id_msg}"
                )

            temp_y_col = "__artifact_future_y__"
            future_for_long = future_raw.copy()
            while temp_y_col in future_for_long.columns:
                temp_y_col = f"_{temp_y_col}"
            future_for_long[temp_y_col] = future_raw["y"] if "y" in future_raw.columns else np.nan
            future = _build_long_frame(
                future_for_long,
                time_col=str(time_col),
                y_col=temp_y_col,
                id_cols=id_cols,
                historic_x_cols=(),
                future_x_cols=x_cols,
                static_cols=tuple(col for col in static_cols if col in future_raw.columns),
                dropna=False,
            )

    if static_cols:
        for col in static_cols:
            if col in future_raw.columns:
                future[col] = future_raw[col]
                continue
            if col not in observed_history.columns:
                raise ValueError(f"Global artifact is missing static_cols context for {col!r}")
            lookup = (
                observed_history.loc[:, ["unique_id", col]]
                .dropna(subset=[col])
                .drop_duplicates(subset=["unique_id"], keep="last")
                .set_index("unique_id")[col]
            )
            future[col] = future["unique_id"].map(lookup)
            if future[col].isna().any():
                raise ValueError(
                    f"Artifact future-path is missing required static_cols values: {[col]}"
                )

    return _forecasting._require_future_df(future)


def _forecast_global_artifact(
    forecaster: BaseGlobalForecaster,
    *,
    extra: dict[str, Any],
    horizon: int,
    interval_levels: Any,
    cutoff: Any,
    future_path: str | None = None,
    time_col: str | None = None,
    parse_dates: bool = False,
) -> pd.DataFrame:
    levels = _forecasting._parse_interval_levels(interval_levels)
    spec = _model_execution.get_model_spec(str(forecaster.model_key))
    if levels:
        if not bool(spec.capabilities.get("supports_interval_forecast", False)):
            raise ValueError(
                f"Forecast intervals are not yet supported for artifact model {spec.key!r} "
                "with interface='global'"
            )

    cutoff_value = _resolve_global_artifact_cutoff(extra=extra, cutoff=cutoff)
    max_horizon = extra.get("max_horizon")
    if max_horizon is not None and horizon > int(max_horizon):
        raise ValueError(
            f"Requested horizon={horizon} exceeds artifact max_horizon={int(max_horizon)}"
        )

    future_path_s = str(future_path or "").strip()
    if future_path_s:
        time_col_s = str(time_col or "").strip()
        if not time_col_s:
            raise ValueError("forecast artifact requires time_col when future_path is provided")

        _historic_x_cols, x_cols = _contracts_normalize_covariate_roles(
            dict(forecaster.model_params)
        )
        static_cols = _contracts_normalize_static_cols(dict(forecaster.model_params))
        id_cols = tuple(str(item) for item in extra.get("id_cols", []))
        original_train_df = _resolve_global_artifact_train_df(forecaster=forecaster)
        observed_history = _resolve_global_artifact_observed_history(
            train_df=original_train_df,
            cutoff=cutoff_value,
        )
        future_override = _load_global_artifact_future_override(
            path=future_path_s,
            time_col=time_col_s,
            parse_dates=bool(parse_dates),
            id_cols=id_cols,
            x_cols=x_cols,
            static_cols=static_cols,
            observed_history=observed_history,
        )
        augmented, _ = _forecasting._prepare_global_forecast_input(
            _forecasting._merge_history_and_future_df(observed_history, future_override),
            horizon=int(horizon),
            x_cols=x_cols,
        )
        try:
            setattr(forecaster, "_train_df", augmented)
            pred = forecaster.predict(cutoff_value, horizon)
        finally:
            setattr(forecaster, "_train_df", original_train_df)
    else:
        pred = forecaster.predict(cutoff_value, horizon)
    pred = _forecasting._finalize_forecast_frame(
        pred,
        cutoff=cutoff_value,
        model=str(forecaster.model_key),
    )
    return _forecasting._add_interval_columns_from_quantile_predictions(
        pred,
        interval_levels=levels,
    )


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

    (
        model_spec,
        long_df,
        future_df,
        historic_x_cols,
        future_x_cols,
        static_cols,
    ) = _build_forecast_long_frames(
        model_key=model_key,
        params=params,
        path=path,
        future_path=future_path,
        parse_dates=bool(parse_dates),
        time_col=time_col_s,
        y_col=y_col_s,
        id_cols=id_cols,
    )
    emit_cli_event(
        "LOAD csv",
        event="forecast_csv_loaded",
        payload=compact_log_payload(
            model=model_key,
            rows=int(len(long_df)),
            n_series=int(long_df["unique_id"].nunique()),
            future_rows=(None if future_df is None else int(len(future_df))),
            historic_x_cols=list(historic_x_cols),
            future_x_cols=list(future_x_cols),
            static_cols=list(static_cols),
        ),
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
    emit_cli_event(
        "FORECAST ready",
        event="forecast_completed",
        payload=compact_log_payload(
            model=model_key,
            rows=int(len(pred)),
            n_series=int(pred["unique_id"].nunique()) if not pred.empty else 0,
        ),
    )

    artifact_path = str(save_artifact_path or "").strip()
    if artifact_path:
        if model_spec.interface == "local":
            _save_local_forecast_artifact(
                artifact_path=artifact_path,
                model_key=model_key,
                params=params,
                future_x_cols=future_x_cols,
                long_df=long_df,
                future_df=future_df,
                horizon=int(horizon),
            )
        else:
            _save_global_forecast_artifact(
                artifact_path=artifact_path,
                model_key=model_key,
                params=params,
                long_df=long_df,
                future_df=future_df,
                id_cols=id_cols,
                future_x_cols=future_x_cols,
                horizon=int(horizon),
            )
        emit_cli_event(
            "ARTIFACT saved",
            event="artifact_saved",
            payload=compact_log_payload(
                model=model_key,
                artifact=artifact_path,
                interface=str(model_spec.interface),
            ),
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
    future_path: str | None = None,
    time_col: str | None = None,
    parse_dates: bool = False,
) -> pd.DataFrame:
    emit_cli_event(
        "ARTIFACT load",
        event="artifact_loaded",
        payload=compact_log_payload(artifact=str(artifact), horizon=int(horizon)),
    )
    payload = load_forecaster_artifact(str(artifact))
    forecaster = payload["forecaster"]
    extra = dict(payload.get("extra", {}))
    horizon_int = int(horizon)

    if isinstance(forecaster, BaseForecaster):
        return _forecast_local_artifact(
            forecaster,
            extra=extra,
            horizon=horizon_int,
            interval_levels=interval_levels,
            interval_min_train_size=interval_min_train_size,
            interval_samples=int(interval_samples),
            interval_seed=interval_seed,
            future_path=future_path,
            time_col=time_col,
            parse_dates=bool(parse_dates),
        )

    if isinstance(forecaster, BaseGlobalForecaster):
        return _forecast_global_artifact(
            forecaster,
            extra=extra,
            horizon=horizon_int,
            interval_levels=interval_levels,
            cutoff=cutoff,
            future_path=future_path,
            time_col=time_col,
            parse_dates=bool(parse_dates),
        )

    raise TypeError(f"Unsupported artifact forecaster type: {type(forecaster).__name__}")


def artifact_info_workflow(
    *,
    artifact: str,
) -> dict[str, Any]:
    payload = load_forecaster_artifact(str(artifact))
    forecaster = payload["forecaster"]
    extra = dict(payload.get("extra", {}))
    result = {
        **_artifact_summary_payload(payload),
        "is_fitted": bool(getattr(forecaster, "is_fitted", False)),
    }
    future_override_schema = _artifact_future_override_schema(
        forecaster=forecaster,
        extra=extra,
    )
    if future_override_schema is not None:
        result["future_override_schema"] = _normalize_artifact_info_value(future_override_schema)
    return result


def _validate_artifact_runtime_contract(
    *,
    forecaster: Any,
    extra: dict[str, Any],
) -> str:
    artifact_type = str(extra.get("artifact_type", "")).strip()
    if artifact_type == "forecast-local":
        if not isinstance(forecaster, BaseForecaster):
            raise TypeError("forecast-local artifacts must wrap a local forecaster")
        train_y = getattr(forecaster, "_train_y", None)
        _resolve_local_artifact_ds(extra=extra, train_y=train_y)
        x_cols = tuple(str(item) for item in extra.get("x_cols", []))
        if x_cols:
            if train_y is None:
                raise ValueError("Local artifact is missing training history required for x_cols")
            _resolve_local_artifact_train_exog(extra=extra, train_y=train_y)
            max_horizon = extra.get("max_horizon")
            if max_horizon is None:
                raise ValueError("Local forecast artifact is missing required max_horizon context")
            max_horizon_int = int(max_horizon)
            if max_horizon_int < 1:
                raise ValueError("Local forecast artifact max_horizon must be >= 1")
            _resolve_local_artifact_saved_future_context(
                extra=extra,
                horizon=max_horizon_int,
            )
        return artifact_type

    if artifact_type == "forecast-global":
        if not isinstance(forecaster, BaseGlobalForecaster):
            raise TypeError("forecast-global artifacts must wrap a global forecaster")
        cutoff = _resolve_global_artifact_cutoff(extra=extra, cutoff=None)
        train_df = _resolve_global_artifact_train_df(forecaster=forecaster)
        _resolve_global_artifact_observed_history(
            train_df=train_df,
            cutoff=cutoff,
        )
        max_horizon = extra.get("max_horizon")
        if max_horizon is None:
            raise ValueError("Global forecast artifact is missing required max_horizon context")
        if int(max_horizon) < 1:
            raise ValueError("Global forecast artifact max_horizon must be >= 1")
        return artifact_type

    return artifact_type


def artifact_validate_workflow(
    *,
    artifact: str,
) -> dict[str, Any]:
    payload = load_forecaster_artifact(str(artifact))
    forecaster = payload["forecaster"]
    metadata = dict(payload["metadata"])
    extra = dict(payload.get("extra", {}))
    artifact_type = _validate_artifact_runtime_contract(
        forecaster=forecaster,
        extra=extra,
    )
    result = {
        "valid": True,
        "artifact_schema_version": int(payload["artifact_schema_version"]),
        "forecaster_type": _artifact_forecaster_type(forecaster),
        "model_key": str(metadata["model_key"]),
        "train_kind": str(dict(metadata["train_schema"]).get("kind", "")),
    }
    if artifact_type:
        result["artifact_type"] = artifact_type
    return result


def artifact_diff_workflow(
    *,
    left_artifact: str,
    right_artifact: str,
    path_prefix: str | None = None,
) -> dict[str, Any]:
    left_payload = _artifact_summary_payload(load_forecaster_artifact(str(left_artifact)))
    right_payload = _artifact_summary_payload(load_forecaster_artifact(str(right_artifact)))
    differences: dict[str, dict[str, Any]] = {}
    _collect_artifact_differences(
        left=left_payload,
        right=right_payload,
        path="",
        out=differences,
    )
    filtered = _filter_artifact_differences(differences, path_prefix=path_prefix)
    return {
        "equal": not filtered,
        "difference_count": int(len(filtered)),
        "differences": filtered,
    }


def artifact_diff_rows_workflow(
    *,
    left_artifact: str,
    right_artifact: str,
    path_prefix: str | None = None,
) -> list[dict[str, str]]:
    payload = artifact_diff_workflow(
        left_artifact=left_artifact,
        right_artifact=right_artifact,
        path_prefix=path_prefix,
    )
    rows: list[dict[str, str]] = []
    for path in sorted(payload["differences"]):
        item = payload["differences"][path]
        rows.append(
            {
                "path": str(path),
                "left": _stringify_artifact_diff_value(item["left"]),
                "right": _stringify_artifact_diff_value(item["right"]),
            }
        )
    return rows


def artifact_diff_markdown_workflow(
    *,
    left_artifact: str,
    right_artifact: str,
    path_prefix: str | None = None,
) -> str:
    payload = artifact_diff_workflow(
        left_artifact=left_artifact,
        right_artifact=right_artifact,
        path_prefix=path_prefix,
    )
    if bool(payload["equal"]):
        prefix = str(path_prefix or "").strip()
        suffix = f" under `{prefix}`" if prefix else ""
        return f"## Summary\n\nNo differences found{suffix}."
    differences = dict(payload["differences"])
    sections = [
        _render_markdown_section(
            title="Summary",
            rows=_artifact_diff_summary_rows(payload, path_prefix=path_prefix),
            columns=["field", "value"],
        )
    ]

    sections.append(
        _render_markdown_section(
            title="Tracking Summary",
            rows=_artifact_diff_tracking_summary_rows(differences),
            columns=["backend", "left", "right"],
        )
    )
    sections.append(
        _render_markdown_section(
            title="Tracking Details",
            rows=_artifact_diff_prefixed_rows(differences, prefix="tracking"),
            columns=["field", "left", "right"],
        )
    )
    sections.append(
        _render_markdown_section(
            title="Metadata",
            rows=_artifact_diff_prefixed_rows(differences, prefix="metadata"),
            columns=["field", "left", "right"],
        )
    )
    sections.append(
        _render_markdown_section(
            title="Extra",
            rows=_artifact_diff_prefixed_rows(differences, prefix="extra"),
            columns=["field", "left", "right"],
        )
    )
    sections.append(
        _render_markdown_section(
            title="Future Override",
            rows=_artifact_diff_prefixed_rows(differences, prefix="future_override_schema"),
            columns=["field", "left", "right"],
        )
    )
    sections.append(
        _render_markdown_section(
            title="Other",
            rows=_artifact_diff_other_rows(differences),
            columns=["field", "left", "right"],
        )
    )

    return "\n\n".join(section for section in sections if section)


def artifact_info_rows_workflow(
    *,
    artifact: str,
) -> list[dict[str, str]]:
    payload = artifact_info_workflow(artifact=artifact)
    return _flatten_artifact_summary_rows(payload)


def artifact_info_markdown_workflow(
    *,
    artifact: str,
) -> str:
    payload = artifact_info_workflow(artifact=artifact)
    sections = [
        _render_markdown_section(
            title="Summary",
            rows=_artifact_info_summary_rows(payload),
            columns=["field", "value"],
        )
    ]

    composition_summary = payload.get("composition_summary", {})
    if isinstance(composition_summary, dict) and composition_summary:
        sections.append(
            _render_markdown_section(
                title="Composition",
                rows=_flatten_artifact_summary_rows(composition_summary),
                columns=["field", "value"],
            )
        )

    tracking = payload.get("tracking", {})
    if isinstance(tracking, dict) and tracking:
        sections.append(
            _render_markdown_section(
                title="Tracking",
                rows=_artifact_info_tracking_rows(payload),
                columns=["backend", "summary"],
            )
        )
        sections.append(
            _render_markdown_section(
                title="Tracking Details",
                rows=_flatten_artifact_summary_rows(tracking),
                columns=["field", "value"],
            )
        )

    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and metadata:
        sections.append(
            _render_markdown_section(
                title="Metadata",
                rows=_flatten_artifact_summary_rows(metadata),
                columns=["field", "value"],
            )
        )

    extra = payload.get("extra", {})
    if isinstance(extra, dict) and extra:
        sections.append(
            _render_markdown_section(
                title="Extra",
                rows=_flatten_artifact_summary_rows(extra),
                columns=["field", "value"],
            )
        )

    future_override = payload.get("future_override_schema", {})
    if isinstance(future_override, dict) and future_override:
        sections.append(
            _render_markdown_section(
                title="Future Override",
                rows=_flatten_artifact_summary_rows(future_override),
                columns=["field", "value"],
            )
        )

    return "\n\n".join(section for section in sections if section)


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
    historic_x_cols, future_x_cols, static_cols = _resolve_forecast_covariates(params)

    df = _load_csv_frame(path, parse_dates=bool(parse_dates), time_col=time_col_s)
    long_df = _build_long_frame(
        df,
        time_col=time_col_s,
        y_col=y_col_s,
        id_cols=id_cols,
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
        static_cols=static_cols,
        dropna=True,
    )
    emit_cli_event(
        "LOAD csv",
        event="eval_csv_loaded",
        payload=compact_log_payload(
            model=str(model),
            rows=int(len(long_df)),
            n_series=int(long_df["unique_id"].nunique()),
            historic_x_cols=list(historic_x_cols),
            future_x_cols=list(future_x_cols),
            static_cols=list(static_cols),
        ),
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
    emit_cli_event(
        "EVAL done",
        event="eval_completed",
        payload=compact_log_payload(
            model=str(model),
            n_points=payload.get("n_points"),
            n_series=payload.get("n_series"),
        ),
    )
    return payload


def detect_csv_workflow(
    *,
    path: str,
    time_col: str,
    y_col: str,
    model: str | None = None,
    id_cols: tuple[str, ...] = (),
    parse_dates: bool = False,
    model_params: dict[str, Any] | None = None,
    score_method: str | None = None,
    threshold_method: str | None = None,
    threshold_k: float = 3.0,
    threshold_quantile: float = 0.99,
    window: int = 12,
    min_history: int = 3,
    min_train_size: int | None = None,
    step_size: int = 1,
    max_train_size: int | None = None,
    n_windows: int | None = None,
) -> pd.DataFrame:
    params = dict(model_params or {})
    time_col_s = str(time_col)
    y_col_s = str(y_col)
    historic_x_cols, future_x_cols, static_cols = _resolve_model_param_covariates(params)

    df = _load_csv_frame(path, parse_dates=bool(parse_dates), time_col=time_col_s)
    long_df = _build_long_frame(
        df,
        time_col=time_col_s,
        y_col=y_col_s,
        id_cols=id_cols,
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
        static_cols=static_cols,
        dropna=True,
    )
    emit_cli_event(
        "LOAD csv",
        event="detect_csv_loaded",
        payload=compact_log_payload(
            model=str(model or "").strip() or None,
            rows=int(len(long_df)),
            n_series=int(long_df["unique_id"].nunique()),
            historic_x_cols=list(historic_x_cols),
            future_x_cols=list(future_x_cols),
            static_cols=list(static_cols),
        ),
    )

    out = _detection.detect_anomalies_long_df(
        long_df=long_df,
        model=(str(model).strip() or None),
        model_params=params,
        score_method=score_method,
        threshold_method=threshold_method,
        threshold_k=float(threshold_k),
        threshold_quantile=float(threshold_quantile),
        window=int(window),
        min_history=int(min_history),
        min_train_size=min_train_size,
        step_size=int(step_size),
        max_train_size=max_train_size,
        n_windows=n_windows,
    )
    out.attrs.update(
        {
            "dataset": str(path),
            "time_col": time_col_s,
            "y_col": y_col_s,
            "id_cols": list(id_cols),
        }
    )
    return out


def cv_csv_workflow(
    *,
    model: str,
    path: str,
    time_col: str,
    y_col: str,
    horizon: int,
    step_size: int,
    min_train_size: int,
    id_cols: tuple[str, ...] = (),
    parse_dates: bool = False,
    model_params: dict[str, Any] | None = None,
    max_train_size: int | None = None,
    n_windows: int | None = None,
) -> pd.DataFrame:
    params = dict(model_params or {})
    time_col_s = str(time_col)
    y_col_s = str(y_col)
    historic_x_cols, future_x_cols, static_cols = _resolve_model_param_covariates(params)

    df = _load_csv_frame(path, parse_dates=bool(parse_dates), time_col=time_col_s)
    long_df = _build_long_frame(
        df,
        time_col=time_col_s,
        y_col=y_col_s,
        id_cols=id_cols,
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
        static_cols=static_cols,
        dropna=True,
    )
    emit_cli_event(
        "LOAD csv",
        event="cv_csv_loaded",
        payload=compact_log_payload(
            model=str(model),
            rows=int(len(long_df)),
            n_series=int(long_df["unique_id"].nunique()),
            historic_x_cols=list(historic_x_cols),
            future_x_cols=list(future_x_cols),
            static_cols=list(static_cols),
        ),
    )

    out = cross_validation_predictions_long_df(
        model=str(model),
        long_df=long_df,
        horizon=int(horizon),
        step_size=int(step_size),
        min_train_size=int(min_train_size),
        model_params=params,
        max_train_size=max_train_size,
        n_windows=n_windows,
    )
    out.attrs.update(
        {
            "dataset": str(path),
            "time_col": time_col_s,
            "y_col": y_col_s,
            "id_cols": list(id_cols),
        }
    )
    return out


__all__ = [
    "artifact_diff_markdown_workflow",
    "artifact_info_markdown_workflow",
    "artifact_info_rows_workflow",
    "artifact_diff_workflow",
    "artifact_diff_rows_workflow",
    "artifact_info_workflow",
    "artifact_validate_workflow",
    "cv_csv_workflow",
    "detect_csv_workflow",
    "eval_csv_workflow",
    "forecast_artifact_workflow",
    "forecast_csv_workflow",
]
