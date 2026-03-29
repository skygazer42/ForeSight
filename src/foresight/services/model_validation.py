from __future__ import annotations

import json
import pickle
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.format import long_to_wide, to_long
from ..data.prep import prepare_long_df
from ..datasets import load_dataset
from ..datasets.registry import get_dataset_spec
from ..models.registry import get_model_spec, list_models
from ..optional_deps import is_dependency_available, require_dependency
from .evaluation import eval_model_long_df, eval_multivariate_model_df

_VALIDATION_DATASET_KEY = "promotion_data"
_VALIDATION_FREQ = "W-MON"
_VALIDATION_HORIZON = 3
_VALIDATION_STEP = 3
_VALIDATION_MIN_TRAIN_SIZE = 12
_VALIDATION_MAX_WINDOWS = 8
_MULTIVARIATE_N_SERIES = 4
_VALIDATION_MAX_SERIES = 32
_CPU_EPOCHS = 1
_GPU_EPOCHS = 1
_CPU_BATCH_SIZE = 16
_GPU_BATCH_SIZE = 32
_LIGHTWEIGHT_N_ESTIMATORS = 20
_LIGHTWEIGHT_ITERATIONS = 20
_LIGHTWEIGHT_LEARNING_RATE = 0.1
_LIGHTWEIGHT_MAX_ITER = 200
_LIGHTWEIGHT_RANDOM_STATE = 0
_LIGHTWEIGHT_SEED = 0
_LIGHTWEIGHT_PATIENCE = 2
_LIGHTWEIGHT_CONTEXT_LENGTH = 8
_LIGHTWEIGHT_NUM_SAMPLES = 20
_DEFAULT_OUTPUT_DIR = Path("artifacts") / "validate_all_models"
_DEFAULT_PROGRESS_EVERY = 50


def _normalize_data_dir(data_dir: str | Path | None) -> str | None:
    if data_dir is None:
        return None
    data_dir_s = str(data_dir).strip()
    return data_dir_s or None


def _limit_series(long_df: pd.DataFrame, *, max_series: int | None) -> pd.DataFrame:
    if max_series is None:
        return long_df.reset_index(drop=True)

    max_series_int = int(max_series)
    if max_series_int < 2:
        raise ValueError("max_series must be >= 2 or None")

    ranked = (
        long_df.groupby("unique_id", sort=False)["y"]
        .agg(
            n_obs="size",
            n_nonzero=lambda s: int((s != 0).sum()),
            n_unique="nunique",
            y_std="std",
        )
        .reset_index()
        .assign(
            unique_id=lambda df: df["unique_id"].astype("string"),
            y_std=lambda df: df["y_std"].fillna(0.0),
        )
        .sort_values(
            ["n_nonzero", "n_unique", "y_std", "unique_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
    )
    selected = ranked["unique_id"].head(max_series_int).tolist()
    return (
        long_df.loc[long_df["unique_id"].astype("string").isin(selected)]
        .sort_values(["unique_id", "ds"], kind="mergesort")
        .reset_index(drop=True)
    )


def _requires_label(spec: Any) -> str:
    requires = tuple(str(item).strip() for item in getattr(spec, "requires", ()) if str(item).strip())
    if not requires:
        return "core"
    return str(requires[0])


def _supported_model_params(spec: Any) -> set[str]:
    return {str(name) for name in set(getattr(spec, "default_params", {})).union(getattr(spec, "param_help", {}))}


def prepare_promotion_long_df(
    data_dir: str | Path | None = None,
    *,
    max_series: int | None = _VALIDATION_MAX_SERIES,
) -> pd.DataFrame:
    spec = get_dataset_spec(_VALIDATION_DATASET_KEY)
    raw_df = load_dataset(_VALIDATION_DATASET_KEY, data_dir=_normalize_data_dir(data_dir))
    long_df = to_long(
        raw_df,
        time_col=str(spec.time_col),
        y_col=str(spec.default_y),
        id_cols=tuple(spec.group_cols),
        dropna=True,
    )
    prepared = prepare_long_df(
        long_df,
        freq=_VALIDATION_FREQ,
        strict_freq=False,
        y_missing="zero",
    )
    ds = pd.to_datetime(prepared["ds"])
    month_angle = 2.0 * np.pi * (ds.dt.month.astype(float) - 1.0) / 12.0
    week_angle = 2.0 * np.pi * (ds.dt.isocalendar().week.astype(int).astype(float) - 1.0) / 53.0
    prepared["time_month_sin"] = np.sin(month_angle)
    prepared["time_month_cos"] = np.cos(month_angle)
    prepared["time_week_sin"] = np.sin(week_angle)
    prepared["time_week_cos"] = np.cos(week_angle)
    return _limit_series(prepared, max_series=max_series)


def build_promotion_multivariate_wide_df(
    long_df: pd.DataFrame,
    *,
    n_series: int = _MULTIVARIATE_N_SERIES,
) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df must be a pandas DataFrame")
    if int(n_series) < 2:
        raise ValueError("n_series must be >= 2")
    if long_df.empty:
        raise ValueError("long_df is empty")

    ranked = (
        long_df.groupby("unique_id", sort=False)["y"]
        .agg(
            n_obs="size",
            n_nonzero=lambda s: int((s != 0).sum()),
            n_unique="nunique",
            y_std="std",
        )
        .reset_index()
        .assign(
            unique_id=lambda df: df["unique_id"].astype("string"),
            y_std=lambda df: df["y_std"].fillna(0.0),
        )
        .sort_values(
            ["n_nonzero", "n_unique", "y_std", "unique_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
    )
    selected = ranked["unique_id"].head(int(n_series)).tolist()
    if len(selected) < int(n_series):
        raise ValueError(f"Need at least {int(n_series)} series for multivariate validation")

    subset = long_df.loc[long_df["unique_id"].isin(selected)].copy()
    wide_df = long_to_wide(
        subset,
        freq=_VALIDATION_FREQ,
        strict_freq=False,
        missing="zero",
    )
    target_cols = [str(col) for col in wide_df.columns if str(col) != "ds"]
    return wide_df, target_cols


def _torch_installed() -> bool:
    return is_dependency_available("torch")


def resolve_runtime_device(device: str = "auto") -> str:
    requested = str(device).strip().lower() or "auto"
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    if requested == "cpu":
        return "cpu"

    if not _torch_installed():
        if requested == "cuda":
            raise RuntimeError("device='cuda' requested but torch is not installed")
        return "cpu"

    torch = require_dependency("torch", install_hint='pip install -e ".[torch]"')

    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return "cuda"
    return "cuda" if cuda_available else "cpu"


def _is_foundation_fixture_wrapper(spec: Any) -> bool:
    supported = _supported_model_params(spec)
    return {"backend", "checkpoint_path"}.issubset(supported) and "model_source" in supported


def _write_foundation_fixture(output_dir: str | Path | None) -> Path:
    base_dir = _DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = base_dir / "foundation-fixture.json"
    fixture_path.write_text(
        json.dumps({"bias": 1.5, "scale": 1.0, "use_trend": True}, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return fixture_path


def transform_validation_long_df_for_model(long_df: pd.DataFrame, *, model: str) -> pd.DataFrame:
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df must be a pandas DataFrame")

    model_key = str(model).strip().lower()
    out = long_df.copy()
    if "logistic" in model_key:
        def _scale(group: pd.Series) -> pd.Series:
            values = group.to_numpy(dtype=float, copy=False)
            max_value = float(values.max(initial=0.0))
            if max_value <= 0.0:
                return pd.Series(values, index=group.index, dtype=float)
            return pd.Series(values / max_value, index=group.index, dtype=float)

        out["y"] = out.groupby("unique_id", sort=False)["y"].transform(_scale)
        return out

    if any(token in model_key for token in ("gamma", "poisson", "tweedie")) or model_key in {
        "holt-winters-mul",
        "holt-winters-mul-auto",
    }:
        def _positive_unit_scale(group: pd.Series) -> pd.Series:
            values = group.to_numpy(dtype=float, copy=False)
            min_value = float(values.min(initial=0.0))
            shifted = values + ((1e-6 - min_value) if min_value <= 0.0 else 0.0)
            max_value = float(shifted.max(initial=0.0))
            if max_value > 0.0:
                shifted = shifted / max_value
            return pd.Series(shifted, index=group.index, dtype=float)

        out["y"] = out.groupby("unique_id", sort=False)["y"].transform(_positive_unit_scale)
        return out

    if "catboost" in model_key:
        def _jitter(group: pd.Series) -> pd.Series:
            values = group.to_numpy(dtype=float, copy=False)
            eps = 1e-6 * np.arange(values.size, dtype=float)
            return pd.Series(values + eps, index=group.index, dtype=float)

        out["y"] = out.groupby("unique_id", sort=False)["y"].transform(_jitter)
        return out

    return out


def build_model_params(
    spec: Any,
    *,
    device: str,
    foundation_fixture_path: str | Path | None = None,
) -> dict[str, Any]:
    model_key = str(getattr(spec, "key", "")).strip().lower()
    supported = _supported_model_params(spec)
    epochs = _GPU_EPOCHS if str(device) == "cuda" else _CPU_EPOCHS
    batch_size = _GPU_BATCH_SIZE if str(device) == "cuda" else _CPU_BATCH_SIZE
    candidates = {
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": _LIGHTWEIGHT_SEED,
        "patience": _LIGHTWEIGHT_PATIENCE,
        "warmup_epochs": 0,
        "ema_warmup_epochs": 0,
        "swa_start_epoch": -1,
        "min_epochs": 1,
        "n_estimators": _LIGHTWEIGHT_N_ESTIMATORS,
        "iterations": _LIGHTWEIGHT_ITERATIONS,
        "learning_rate": _LIGHTWEIGHT_LEARNING_RATE,
        "max_iter": _LIGHTWEIGHT_MAX_ITER,
        "random_state": _LIGHTWEIGHT_RANDOM_STATE,
        "lags": _LIGHTWEIGHT_CONTEXT_LENGTH,
        "context_length": _LIGHTWEIGHT_CONTEXT_LENGTH,
        "num_samples": _LIGHTWEIGHT_NUM_SAMPLES,
    }
    if model_key in {"autoreg", "stl-autoreg", "mstl-autoreg", "tbats-lite-autoreg"} and "lags" in supported:
        candidates["lags"] = 4
    if "segrnn" in model_key and "segment_len" in supported:
        candidates["segment_len"] = 8
    if "knn" in model_key and "n_neighbors" in supported:
        candidates["n_neighbors"] = 2
    if model_key in {"fourier-autoreg", "stl-autoreg", "mstl-autoreg", "tbats-lite-autoreg"}:
        if "periods" in supported:
            candidates["periods"] = (4,)
        if "orders" in supported:
            candidates["orders"] = 1
        if "lags" in supported:
            candidates["lags"] = min(int(candidates.get("lags", 4)), 2)
    if model_key in {
        "ets",
        "stl-ets",
        "mstl-ets",
        "tbats-lite-ets",
        "holt-winters-add",
        "holt-winters-add-auto",
        "holt-winters-mul",
        "holt-winters-mul-auto",
    } and "season_length" in supported:
        candidates["season_length"] = 4
    if model_key in {"seasonal-drift", "seasonal-mean", "seasonal-naive", "seasonal-naive-auto"} and "season_length" in supported:
        candidates["season_length"] = 4
    if "lstnet" in model_key and "highway_window" in supported:
        candidates["highway_window"] = 8
    if model_key == "sar-ols" and "season_length" in supported:
        candidates["season_length"] = 4
    if model_key in {"fourier-ets", "stl-ets", "mstl-ets", "tbats-lite-ets"}:
        if "periods" in supported:
            candidates["periods"] = (4,)
        if "orders" in supported:
            candidates["orders"] = 1
    if model_key in {"mstl-arima", "mstl-auto-arima", "mstl-sarimax", "mstl-uc"} and "periods" in supported:
        candidates["periods"] = (4,)
    if "crossformer" in model_key:
        if "segment_len" in supported:
            candidates["segment_len"] = 8
        if "stride" in supported:
            candidates["stride"] = 8
    if "pyraformer" in model_key:
        if "segment_len" in supported:
            candidates["segment_len"] = 8
        if "stride" in supported:
            candidates["stride"] = 8
    if model_key == "hf-timeseries-transformer-direct":
        if "context_length" in supported:
            candidates["context_length"] = 4
        if "lags_sequence" in supported:
            candidates["lags_sequence"] = (1, 2, 3)
    if "timexer" in model_key and "x_cols" in supported:
        candidates["x_cols"] = (
            "time_month_sin",
            "time_month_cos",
            "time_week_sin",
            "time_week_cos",
        )
    if "patchtst" in model_key:
        if "patch_len" in supported:
            candidates["patch_len"] = 8
        if "stride" in supported:
            candidates["stride"] = 8
    if model_key == "ssa" and "window_length" in supported:
        candidates["window_length"] = 8
    if foundation_fixture_path is not None and _is_foundation_fixture_wrapper(spec):
        candidates.update(
            {
                "backend": "fixture-json",
                "checkpoint_path": str(Path(foundation_fixture_path)),
                "local_files_only": True,
            }
        )
    return {key: value for key, value in candidates.items() if key in supported}


def _ok_row(
    *,
    model: str,
    interface: str,
    requires: tuple[str, ...],
    device: str,
    duration_seconds: float,
    payload: dict[str, Any],
) -> dict[str, Any]:
    metrics = {
        "mae": payload.get("mae"),
        "rmse": payload.get("rmse"),
        "mape": payload.get("mape"),
        "smape": payload.get("smape"),
    }
    for name, value in metrics.items():
        if value is None:
            raise ValueError(f"{model!r} did not return {name}")
        if not np.isfinite(float(value)):
            raise ValueError(f"{model!r} returned non-finite {name}")

    return {
        "model": str(model),
        "interface": str(interface),
        "requires": [str(item) for item in requires],
        "backend": _requires_label(type("Spec", (), {"requires": requires})()),
        "status": "ok",
        "n_points": int(payload.get("n_points", 0)),
        "mae": float(payload["mae"]),
        "rmse": float(payload["rmse"]),
        "mape": float(payload["mape"]),
        "smape": float(payload["smape"]),
        "device": str(device),
        "duration_seconds": round(float(duration_seconds), 6),
        "error_type": "",
        "error_message": "",
    }


def _error_row(
    *,
    model: str,
    interface: str,
    requires: tuple[str, ...],
    device: str,
    duration_seconds: float,
    error: Exception,
) -> dict[str, Any]:
    return {
        "model": str(model),
        "interface": str(interface),
        "requires": [str(item) for item in requires],
        "backend": _requires_label(type("Spec", (), {"requires": requires})()),
        "status": "error",
        "n_points": 0,
        "mae": None,
        "rmse": None,
        "mape": None,
        "smape": None,
        "device": str(device),
        "duration_seconds": round(float(duration_seconds), 6),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _evaluate_model(
    *,
    model: str,
    spec: Any,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    target_cols: list[str],
    device: str,
    foundation_fixture_path: str | Path | None,
) -> dict[str, Any]:
    params = build_model_params(
        spec,
        device=device,
        foundation_fixture_path=foundation_fixture_path,
    )
    model_long_df = transform_validation_long_df_for_model(long_df, model=model)
    started = time.perf_counter()
    try:
        interface = str(spec.interface).strip().lower()
        if interface == "multivariate":
            model_wide_df, model_target_cols = build_promotion_multivariate_wide_df(model_long_df)
            payload = eval_multivariate_model_df(
                model=str(model),
                df=model_wide_df,
                target_cols=model_target_cols,
                horizon=_VALIDATION_HORIZON,
                step=_VALIDATION_STEP,
                min_train_size=_VALIDATION_MIN_TRAIN_SIZE,
                max_windows=_VALIDATION_MAX_WINDOWS,
                model_params=params,
            )
        else:
            payload = eval_model_long_df(
                model=str(model),
                long_df=model_long_df,
                horizon=_VALIDATION_HORIZON,
                step=_VALIDATION_STEP,
                min_train_size=_VALIDATION_MIN_TRAIN_SIZE,
                max_windows=_VALIDATION_MAX_WINDOWS,
                model_params=params,
            )
        duration_seconds = time.perf_counter() - started
        return _ok_row(
            model=str(model),
            interface=str(spec.interface),
            requires=tuple(spec.requires),
            device=str(device),
            duration_seconds=duration_seconds,
            payload=payload,
        )
    except Exception as error:  # noqa: BLE001
        duration_seconds = time.perf_counter() - started
        return _error_row(
            model=str(model),
            interface=str(spec.interface),
            requires=tuple(spec.requires),
            device=str(device),
            duration_seconds=duration_seconds,
            error=error,
        )


def _normalize_models(models: Iterable[str] | None) -> list[str]:
    if models is None:
        return list_models()
    normalized = [str(model).strip() for model in models if str(model).strip()]
    if not normalized:
        raise ValueError("models must contain at least one non-empty model key")
    return normalized


def _summary_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# All Model Validation Summary",
        "",
        f"- dataset: `{payload['dataset']}`",
        f"- device: `{payload['device']}`",
        f"- total_models: `{summary['total_models']}`",
        f"- ok_models: `{summary['ok_models']}`",
        f"- failed_models: `{summary['failed_models']}`",
        f"- duration_seconds_total: `{summary['duration_seconds_total']}`",
        "",
        "## By Interface",
    ]
    for name, value in summary["by_interface"].items():
        lines.append(f"- {name}: {value}")
    lines.extend(["", "## By Backend"])
    for name, value in summary["by_backend"].items():
        lines.append(f"- {name}: {value}")
    return "\n".join(lines) + "\n"


def write_validation_outputs(payload: dict[str, Any], output_dir: str | Path | None) -> Path:
    output_path = _DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows_path = output_path / "rows.json"
    summary_path = output_path / "summary.json"
    summary_md_path = output_path / "summary.md"

    rows_path.write_text(
        json.dumps(payload["rows"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_md_path.write_text(_summary_markdown(payload), encoding="utf-8")
    return output_path


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] != "ok"]
    return {
        "total_models": len(rows),
        "ok_models": len(ok_rows),
        "failed_models": len(failed_rows),
        "by_interface": dict(Counter(str(row["interface"]) for row in rows)),
        "by_backend": dict(Counter(str(row["backend"]) for row in rows)),
        "duration_seconds_total": round(sum(float(row["duration_seconds"]) for row in rows), 6),
    }


def build_training_artifact_paths(
    *,
    output_dir: str | Path,
    model: str,
) -> dict[str, Path]:
    model_dir = Path(output_dir) / "models" / str(model)
    checkpoints_dir = model_dir / "checkpoints"
    return {
        "model_dir": model_dir,
        "result_path": model_dir / "result.json",
        "forecast_artifact_path": model_dir / "forecast_artifact.pkl",
        "checkpoints_dir": checkpoints_dir,
        "best_checkpoint_path": checkpoints_dir / "best.pt",
        "last_checkpoint_path": checkpoints_dir / "last.pt",
        "var_artifact_path": model_dir / "var.pkl",
    }


def build_training_model_params(
    spec: Any,
    *,
    device: str,
    artifact_paths: dict[str, Path],
    foundation_fixture_path: str | Path | None = None,
) -> dict[str, Any]:
    params = build_model_params(
        spec,
        device=device,
        foundation_fixture_path=foundation_fixture_path,
    )
    supported = _supported_model_params(spec)
    requires = {str(item).strip() for item in getattr(spec, "requires", ())}
    if "torch" in requires or "transformers" in requires:
        if "checkpoint_dir" in supported:
            params["checkpoint_dir"] = str(artifact_paths["checkpoints_dir"])
        if "save_best_checkpoint" in supported:
            params["save_best_checkpoint"] = True
        if "save_last_checkpoint" in supported:
            params["save_last_checkpoint"] = True
        if "resume_checkpoint_path" in supported:
            params["resume_checkpoint_path"] = ""
    return params


def write_training_progress(
    *,
    output_dir: str | Path,
    completed_models: int,
    total_models: int,
    ok_models: int,
    failed_models: int,
    last_model: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    progress_path = output_path / "progress.json"
    payload = {
        "completed_models": int(completed_models),
        "total_models": int(total_models),
        "ok_models": int(ok_models),
        "failed_models": int(failed_models),
        "last_model": str(last_model),
        "updated_at_epoch_seconds": time.time(),
    }
    progress_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return progress_path


def _progress_line(
    *,
    completed_models: int,
    total_models: int,
    ok_models: int,
    failed_models: int,
    last_model: str,
) -> str:
    return (
        f"[{int(completed_models)}/{int(total_models)}] "
        f"ok={int(ok_models)} failed={int(failed_models)} last={str(last_model)}"
    )


def _select_representative_local_long_df(long_df: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        long_df.groupby("unique_id", sort=False)["y"]
        .agg(
            n_obs="size",
            n_nonzero=lambda s: int((s != 0).sum()),
            n_unique="nunique",
            y_std="std",
        )
        .reset_index()
        .assign(
            unique_id=lambda df: df["unique_id"].astype("string"),
            y_std=lambda df: df["y_std"].fillna(0.0),
        )
        .sort_values(
            ["n_nonzero", "n_unique", "y_std", "unique_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
    )
    selected = str(ranked["unique_id"].iloc[0])
    return (
        long_df.loc[long_df["unique_id"].astype("string") == selected]
        .sort_values(["unique_id", "ds"], kind="mergesort")
        .reset_index(drop=True)
    )


def _calendar_feature_columns(ds_index: pd.Index) -> pd.DataFrame:
    ds = pd.to_datetime(ds_index)
    out = pd.DataFrame({"ds": ds})
    month_numbers = pd.Series(ds).dt.month.astype(float).to_numpy()
    week_numbers = pd.Series(ds).dt.isocalendar().week.astype(int).astype(float).to_numpy()
    month_angle = 2.0 * np.pi * (month_numbers - 1.0) / 12.0
    week_angle = 2.0 * np.pi * (week_numbers - 1.0) / 53.0
    out["time_month_sin"] = np.sin(month_angle)
    out["time_month_cos"] = np.cos(month_angle)
    out["time_week_sin"] = np.sin(week_angle)
    out["time_week_cos"] = np.cos(week_angle)
    return out


def _align_global_training_long_df(long_df: pd.DataFrame) -> pd.DataFrame:
    last_ds = long_df.groupby("unique_id", sort=False)["ds"].max()
    cutoff = pd.to_datetime(last_ds.min())
    return (
        long_df.loc[pd.to_datetime(long_df["ds"]) <= cutoff]
        .sort_values(["unique_id", "ds"], kind="mergesort")
        .reset_index(drop=True)
    )


def _build_future_covariate_long_df(
    *,
    long_df: pd.DataFrame,
    horizon: int,
    x_cols: tuple[str, ...],
) -> pd.DataFrame | None:
    if not x_cols:
        return None

    supported = {"time_month_sin", "time_month_cos", "time_week_sin", "time_week_cos"}
    unknown = sorted(set(x_cols) - supported)
    if unknown:
        raise ValueError(f"Unsupported validation future x_cols: {unknown}")

    from . import forecasting as _forecasting

    rows: list[pd.DataFrame] = []
    for uid, group in long_df.groupby("unique_id", sort=False):
        future_ds = _forecasting._infer_future_ds(group["ds"], int(horizon))
        future = _calendar_feature_columns(pd.Index(future_ds))
        future.insert(0, "unique_id", str(uid))
        future["y"] = np.nan
        rows.append(future.loc[:, ["unique_id", "ds", "y", *list(x_cols)]])
    return pd.concat(rows, axis=0, ignore_index=True)


def _existing_checkpoint_path(artifact_paths: dict[str, Path]) -> Path | None:
    for key in ("best_checkpoint_path", "last_checkpoint_path"):
        path = artifact_paths[key]
        if path.exists():
            return path
    return None


def _persist_var_artifact(
    *,
    wide_df: pd.DataFrame,
    target_cols: list[str],
    params: dict[str, Any],
    artifact_path: Path,
) -> Path:
    try:
        from statsmodels.tsa.api import VAR  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'VAR artifact persistence requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = wide_df.loc[:, list(target_cols)].to_numpy(dtype=float, copy=False)
    maxlags = int(params.get("maxlags", 1))
    trend = str(params.get("trend", "c"))
    ic = params.get("ic")
    ic_final = None if ic is None or str(ic).strip().lower() in {"", "none", "null"} else str(ic)
    res = VAR(x).fit(maxlags=maxlags, ic=ic_final, trend=trend)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as handle:
        pickle.dump(res, handle)
    return artifact_path


def _ok_training_row(
    *,
    model: str,
    interface: str,
    requires: tuple[str, ...],
    device: str,
    duration_seconds: float,
    artifact_kind: str,
    artifact_path: str,
    n_points: int,
) -> dict[str, Any]:
    return {
        "model": str(model),
        "interface": str(interface),
        "requires": [str(item) for item in requires],
        "backend": _requires_label(type("Spec", (), {"requires": requires})()),
        "status": "ok",
        "n_points": int(n_points),
        "device": str(device),
        "duration_seconds": round(float(duration_seconds), 6),
        "artifact_kind": str(artifact_kind),
        "artifact_path": str(artifact_path),
        "error_type": "",
        "error_message": "",
    }


def _error_training_row(
    *,
    model: str,
    interface: str,
    requires: tuple[str, ...],
    device: str,
    duration_seconds: float,
    error: Exception,
    artifact_kind: str = "none",
    artifact_path: str = "",
) -> dict[str, Any]:
    return {
        "model": str(model),
        "interface": str(interface),
        "requires": [str(item) for item in requires],
        "backend": _requires_label(type("Spec", (), {"requires": requires})()),
        "status": "error",
        "n_points": 0,
        "device": str(device),
        "duration_seconds": round(float(duration_seconds), 6),
        "artifact_kind": str(artifact_kind),
        "artifact_path": str(artifact_path),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _train_local_or_global_model(
    *,
    model: str,
    spec: Any,
    long_df: pd.DataFrame,
    device: str,
    artifact_paths: dict[str, Path],
    foundation_fixture_path: str | Path | None,
) -> dict[str, Any]:
    from ..contracts.params import normalize_covariate_roles
    from . import cli_workflows as _cli_workflows
    from . import forecasting as _forecasting

    params = build_training_model_params(
        spec,
        device=device,
        artifact_paths=artifact_paths,
        foundation_fixture_path=foundation_fixture_path,
    )
    model_long_df = transform_validation_long_df_for_model(long_df, model=model)
    interface = str(spec.interface).strip().lower()
    train_long_df = (
        _select_representative_local_long_df(model_long_df)
        if interface == "local"
        else _align_global_training_long_df(model_long_df)
    )
    _historic_x_cols, x_cols = normalize_covariate_roles(params)
    future_df = _build_future_covariate_long_df(
        long_df=train_long_df,
        horizon=_VALIDATION_HORIZON,
        x_cols=x_cols,
    )
    started = time.perf_counter()
    try:
        pred = _forecasting.forecast_model_long_df(
            model=str(model),
            long_df=train_long_df,
            future_df=future_df,
            horizon=_VALIDATION_HORIZON,
            model_params=params,
        )
        artifact_kind = "forecast_artifact"
        artifact_path = artifact_paths["forecast_artifact_path"]
        if interface == "local":
            _cli_workflows._save_local_forecast_artifact(
                artifact_path=str(artifact_path),
                model_key=str(model),
                params=params,
                long_df=train_long_df,
                future_df=future_df,
                future_x_cols=x_cols,
                horizon=_VALIDATION_HORIZON,
            )
        else:
            _cli_workflows._save_global_forecast_artifact(
                artifact_path=str(artifact_path),
                model_key=str(model),
                params=params,
                long_df=train_long_df,
                future_df=future_df,
                id_cols=("unique_id",),
                future_x_cols=x_cols,
                horizon=_VALIDATION_HORIZON,
            )
        checkpoint_path = _existing_checkpoint_path(artifact_paths)
        if checkpoint_path is not None:
            artifact_kind = "checkpoint"
            artifact_path = checkpoint_path
        duration_seconds = time.perf_counter() - started
        return _ok_training_row(
            model=str(model),
            interface=str(spec.interface),
            requires=tuple(spec.requires),
            device=str(device),
            duration_seconds=duration_seconds,
            artifact_kind=artifact_kind,
            artifact_path=str(artifact_path),
            n_points=int(len(pred)),
        )
    except Exception as error:  # noqa: BLE001
        duration_seconds = time.perf_counter() - started
        return _error_training_row(
            model=str(model),
            interface=str(spec.interface),
            requires=tuple(spec.requires),
            device=str(device),
            duration_seconds=duration_seconds,
            error=error,
        )


def _train_multivariate_model(
    *,
    model: str,
    spec: Any,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    target_cols: list[str],
    device: str,
    artifact_paths: dict[str, Path],
    foundation_fixture_path: str | Path | None,
) -> dict[str, Any]:
    from . import model_execution as _model_execution

    model_long_df = transform_validation_long_df_for_model(long_df, model=model)
    model_wide_df, model_target_cols = build_promotion_multivariate_wide_df(model_long_df)
    params = build_training_model_params(
        spec,
        device=device,
        artifact_paths=artifact_paths,
        foundation_fixture_path=foundation_fixture_path,
    )
    started = time.perf_counter()
    try:
        if str(model) == "var":
            artifact_path = _persist_var_artifact(
                wide_df=model_wide_df,
                target_cols=model_target_cols,
                params=params,
                artifact_path=artifact_paths["var_artifact_path"],
            )
            duration_seconds = time.perf_counter() - started
            return _ok_training_row(
                model=str(model),
                interface=str(spec.interface),
                requires=tuple(spec.requires),
                device=str(device),
                duration_seconds=duration_seconds,
                artifact_kind="var_pickle",
                artifact_path=str(artifact_path),
                n_points=int(_VALIDATION_HORIZON * len(model_target_cols)),
            )

        runner = _model_execution.make_multivariate_forecaster_runner(str(model), params)
        yhat = runner(model_wide_df.loc[:, list(model_target_cols)], _VALIDATION_HORIZON)
        checkpoint_path = _existing_checkpoint_path(artifact_paths)
        if checkpoint_path is None:
            raise RuntimeError(f"{model!r} did not produce a checkpoint file")
        duration_seconds = time.perf_counter() - started
        return _ok_training_row(
            model=str(model),
            interface=str(spec.interface),
            requires=tuple(spec.requires),
            device=str(device),
            duration_seconds=duration_seconds,
            artifact_kind="checkpoint",
            artifact_path=str(checkpoint_path),
            n_points=int(np.asarray(yhat).size),
        )
    except Exception as error:  # noqa: BLE001
        duration_seconds = time.perf_counter() - started
        return _error_training_row(
            model=str(model),
            interface=str(spec.interface),
            requires=tuple(spec.requires),
            device=str(device),
            duration_seconds=duration_seconds,
            error=error,
        )


def _train_single_model(
    *,
    model: str,
    spec: Any,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    target_cols: list[str],
    device: str,
    output_dir: str | Path,
    artifact_paths: dict[str, Path],
    foundation_fixture_path: str | Path | None,
) -> dict[str, Any]:
    artifact_paths["model_dir"].mkdir(parents=True, exist_ok=True)
    interface = str(spec.interface).strip().lower()
    if interface == "multivariate":
        row = _train_multivariate_model(
            model=model,
            spec=spec,
            long_df=long_df,
            wide_df=wide_df,
            target_cols=target_cols,
            device=device,
            artifact_paths=artifact_paths,
            foundation_fixture_path=foundation_fixture_path,
        )
    else:
        row = _train_local_or_global_model(
            model=model,
            spec=spec,
            long_df=long_df,
            device=device,
            artifact_paths=artifact_paths,
            foundation_fixture_path=foundation_fixture_path,
        )
    artifact_paths["result_path"].write_text(
        json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return row


def run_registry_training_validation(
    *,
    models: Iterable[str] | None = None,
    data_dir: str | Path | None = None,
    device: str = "auto",
    output_dir: str | Path | None = None,
    progress_every: int = _DEFAULT_PROGRESS_EVERY,
) -> dict[str, Any]:
    runtime_device = resolve_runtime_device(device)
    output_path = _DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    keys = _normalize_models(models)
    long_df = prepare_promotion_long_df(data_dir=data_dir)
    wide_df, target_cols = build_promotion_multivariate_wide_df(long_df)
    fixture_path = _write_foundation_fixture(output_path)

    rows: list[dict[str, Any]] = []
    ok_models = 0
    failed_models = 0
    progress_every_int = max(int(progress_every), 1)
    total_models = len(keys)

    for index, key in enumerate(keys, start=1):
        row = _train_single_model(
            model=key,
            spec=get_model_spec(key),
            long_df=long_df,
            wide_df=wide_df,
            target_cols=target_cols,
            device=runtime_device,
            output_dir=output_path,
            artifact_paths=build_training_artifact_paths(output_dir=output_path, model=key),
            foundation_fixture_path=fixture_path,
        )
        rows.append(row)
        if row["status"] == "ok":
            ok_models += 1
        else:
            failed_models += 1
        if index % progress_every_int == 0 or index == total_models:
            write_training_progress(
                output_dir=output_path,
                completed_models=index,
                total_models=total_models,
                ok_models=ok_models,
                failed_models=failed_models,
                last_model=str(key),
            )
            print(
                _progress_line(
                    completed_models=index,
                    total_models=total_models,
                    ok_models=ok_models,
                    failed_models=failed_models,
                    last_model=str(key),
                ),
                flush=True,
            )

    payload = {
        "dataset": _VALIDATION_DATASET_KEY,
        "device": runtime_device,
        "horizon": _VALIDATION_HORIZON,
        "step": _VALIDATION_STEP,
        "min_train_size": _VALIDATION_MIN_TRAIN_SIZE,
        "max_windows": _VALIDATION_MAX_WINDOWS,
        "multivariate_target_cols": list(target_cols),
        "rows": rows,
        "summary": _build_summary(rows),
    }
    write_validation_outputs(payload, output_path)
    return payload


def run_registry_validation(
    *,
    models: Iterable[str] | None = None,
    data_dir: str | Path | None = None,
    device: str = "auto",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    runtime_device = resolve_runtime_device(device)
    keys = _normalize_models(models)
    long_df = prepare_promotion_long_df(data_dir=data_dir)
    wide_df, target_cols = build_promotion_multivariate_wide_df(long_df)
    fixture_path = _write_foundation_fixture(output_dir)

    rows = [
        _evaluate_model(
            model=key,
            spec=get_model_spec(key),
            long_df=long_df,
            wide_df=wide_df,
            target_cols=target_cols,
            device=runtime_device,
            foundation_fixture_path=fixture_path,
        )
        for key in keys
    ]
    rows.sort(key=lambda row: (0 if row["status"] == "error" else 1, str(row["model"])))

    payload = {
        "dataset": _VALIDATION_DATASET_KEY,
        "device": runtime_device,
        "horizon": _VALIDATION_HORIZON,
        "step": _VALIDATION_STEP,
        "min_train_size": _VALIDATION_MIN_TRAIN_SIZE,
        "max_windows": _VALIDATION_MAX_WINDOWS,
        "multivariate_target_cols": list(target_cols),
        "rows": rows,
        "summary": _build_summary(rows),
    }
    write_validation_outputs(payload, output_dir)
    return payload


__all__ = [
    "build_training_artifact_paths",
    "build_training_model_params",
    "build_model_params",
    "build_promotion_multivariate_wide_df",
    "prepare_promotion_long_df",
    "resolve_runtime_device",
    "run_registry_validation",
    "run_registry_training_validation",
    "write_training_progress",
    "write_validation_outputs",
]
