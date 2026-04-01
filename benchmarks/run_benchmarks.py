#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Any

_TASK_GROUPS = {"point", "probabilistic", "covariate"}
_BENCHMARK_WORKLOADS = {"panel_cv"}
_BENCHMARK_SCALES = {"tiny", "small", "medium", "large"}
_BENCHMARK_BACKENDS = {"thread", "process"}
_BUDGET_MODES = {"warn", "fail"}
_cli_shared: Any | None = None
_batch_execution: Any | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path(root: Path) -> None:
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _get_cli_shared_module() -> Any:
    return _get_cached_module("_cli_shared", ".cli_shared")


def _get_batch_execution_module() -> Any:
    return _get_cached_module("_batch_execution", ".batch_execution")


def _get_cached_module(cache_name: str, module_name: str) -> Any:
    cached = globals().get(str(cache_name))
    if cached is not None:
        return cached

    from importlib import import_module

    module = import_module(str(module_name), "foresight")
    globals()[str(cache_name)] = module
    return module


def _load_benchmark_config(path: Path | None = None) -> dict[str, Any]:
    config_path = (
        path if path is not None else (_repo_root() / "benchmarks" / "benchmark_config.json")
    )
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("benchmark config must be a JSON object")
    return raw


def _validate_benchmark_dataset_case(config_name: str, dataset_case: Any) -> None:
    if not isinstance(dataset_case, dict):
        raise TypeError(f"benchmark config {config_name!r} dataset rows must be JSON objects")
    required = {"key", "y_col", "horizon", "step", "min_train_size", "max_windows"}
    missing = sorted(required.difference(dataset_case))
    if missing:
        raise KeyError(f"benchmark config {config_name!r} dataset rows missing keys: {missing}")


def _validate_benchmark_model_case(config_name: str, model_case: Any) -> None:
    if not isinstance(model_case, dict):
        raise TypeError(f"benchmark config {config_name!r} model rows must be JSON objects")
    if "key" not in model_case:
        raise KeyError(f"benchmark config {config_name!r} model rows must include 'key'")
    params = model_case.get("params", {})
    if not isinstance(params, dict):
        raise TypeError(f"benchmark config {config_name!r} model params must be a JSON object")


def _validate_benchmark_config(config_name: str, config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError(f"benchmark config {config_name!r} must be a JSON object")

    task_group = str(config.get("task_group", "point")).strip().lower() or "point"
    if task_group not in _TASK_GROUPS:
        raise ValueError(
            f"benchmark config {config_name!r} task_group must be one of: {', '.join(sorted(_TASK_GROUPS))}"
        )

    profiling = config.get("profiling", False)
    if not isinstance(profiling, bool):
        raise TypeError(f"benchmark config {config_name!r} profiling must be a boolean")

    workload = str(config.get("workload", "panel_cv")).strip().lower() or "panel_cv"
    if workload not in _BENCHMARK_WORKLOADS:
        raise ValueError(
            f"benchmark config {config_name!r} workload must be one of: {', '.join(sorted(_BENCHMARK_WORKLOADS))}"
        )

    scale = str(config.get("scale", "small")).strip().lower() or "small"
    if scale not in _BENCHMARK_SCALES:
        raise ValueError(
            f"benchmark config {config_name!r} scale must be one of: {', '.join(sorted(_BENCHMARK_SCALES))}"
        )

    budgets = config.get("budgets", {})
    if not isinstance(budgets, dict):
        raise TypeError(f"benchmark config {config_name!r} budgets must be a JSON object")
    normalized_budgets: dict[str, float] = {}
    for key, value in budgets.items():
        metric_key = str(key).strip()
        if not metric_key:
            raise ValueError(f"benchmark config {config_name!r} budget keys must be non-empty")
        metric_value = float(value)
        if metric_value <= 0.0:
            raise ValueError(f"benchmark config {config_name!r} budget {metric_key!r} must be > 0")
        normalized_budgets[metric_key] = metric_value

    datasets = config.get("datasets", [])
    models = config.get("models", [])
    if not isinstance(datasets, list):
        raise TypeError(f"benchmark config {config_name!r} datasets must be a list")
    if not isinstance(models, list):
        raise TypeError(f"benchmark config {config_name!r} models must be a list")

    for dataset_case in datasets:
        _validate_benchmark_dataset_case(config_name, dataset_case)
    for model_case in models:
        _validate_benchmark_model_case(config_name, model_case)

    out = dict(config)
    out["task_group"] = task_group
    out["profiling"] = bool(profiling)
    out["workload"] = workload
    out["scale"] = scale
    out["budgets"] = normalized_budgets
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:  # noqa: BLE001
        return None


def _summary_columns(level_suffixes: list[int], *, include_profile: bool) -> list[str]:
    interval_cols: list[str] = []
    for suffix in level_suffixes:
        interval_cols.extend(
            [
                f"coverage_{suffix}_mean",
                f"mean_width_{suffix}_mean",
                f"interval_score_{suffix}_mean",
            ]
        )
    profile_cols = (
        [
            "load_seconds_total",
            "load_seconds_mean",
            "prepare_seconds_total",
            "prepare_seconds_mean",
            "eval_seconds_total",
            "eval_seconds_mean",
            "dispatch_seconds_total",
            "dispatch_seconds_mean",
            "raw_cache_hits_total",
            "prepared_cache_hits_total",
            "peak_memory_mb_max",
            "points_per_second_mean",
            "windows_per_second_mean",
        ]
        if include_profile
        else []
    )
    return [
        "model",
        "task_group",
        "backend_family",
        "n_datasets",
        "ok_rows",
        "skip_rows",
        "n_points_total",
        "mae_mean",
        "rmse_mean",
        "mape_mean",
        "smape_mean",
        *interval_cols,
        *profile_cols,
        "cv_seconds_total",
        "cv_seconds_mean",
    ]


def _summarize_rows(
    rows: list[dict[str, Any]], *, conformal_levels: list[int], include_profile: bool
) -> list[dict[str, Any]]:
    by_model: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_key = (
            str(row.get("task_group", "point")),
            str(row.get("backend_family", "unknown")),
            str(row["model"]),
        )
        by_model[group_key].append(row)

    out: list[dict[str, Any]] = []
    for task_group, backend_family, model in sorted(by_model):
        summary = _build_benchmark_summary_context(
            by_model[(task_group, backend_family, model)],
            conformal_levels=conformal_levels,
            include_profile=include_profile,
        )
        summary.update(
            {
                "model": model,
                "task_group": task_group,
                "backend_family": backend_family,
            }
        )
        out.append(summary)

    return out


def _build_benchmark_summary_context(
    items: list[dict[str, Any]],
    *,
    conformal_levels: list[int],
    include_profile: bool,
) -> dict[str, Any]:
    ok_items = [item for item in items if str(item.get("status", "ok")).strip().lower() == "ok"]
    summary: dict[str, Any] = {
        "n_datasets": int(len({str(item["dataset"]) for item in items})),
        "ok_rows": int(len(ok_items)),
        "skip_rows": int(len(items) - len(ok_items)),
        "n_points_total": int(sum(int(item.get("n_points", 0) or 0) for item in ok_items)),
        "mae_mean": _mean(
            [float(item["mae"]) for item in ok_items if _as_float(item.get("mae")) is not None]
        ),
        "rmse_mean": _mean(
            [float(item["rmse"]) for item in ok_items if _as_float(item.get("rmse")) is not None]
        ),
        "mape_mean": _mean(
            [float(item["mape"]) for item in ok_items if _as_float(item.get("mape")) is not None]
        ),
        "smape_mean": _mean(
            [float(item["smape"]) for item in ok_items if _as_float(item.get("smape")) is not None]
        ),
        "cv_seconds_total": float(sum(float(item["cv_seconds"]) for item in items)),
        "cv_seconds_mean": _mean([float(item["cv_seconds"]) for item in items]),
    }
    if include_profile:
        summary.update(
            {
                "load_seconds_total": float(
                    sum(float(item.get("load_seconds", 0.0) or 0.0) for item in items)
                ),
                "load_seconds_mean": _mean(
                    [float(item.get("load_seconds", 0.0) or 0.0) for item in items]
                ),
                "prepare_seconds_total": float(
                    sum(float(item.get("prepare_seconds", 0.0) or 0.0) for item in items)
                ),
                "prepare_seconds_mean": _mean(
                    [float(item.get("prepare_seconds", 0.0) or 0.0) for item in items]
                ),
                "eval_seconds_total": float(
                    sum(float(item.get("eval_seconds", 0.0) or 0.0) for item in items)
                ),
                "eval_seconds_mean": _mean(
                    [float(item.get("eval_seconds", 0.0) or 0.0) for item in items]
                ),
                "dispatch_seconds_total": float(
                    sum(float(item.get("dispatch_seconds", 0.0) or 0.0) for item in items)
                ),
                "dispatch_seconds_mean": _mean(
                    [float(item.get("dispatch_seconds", 0.0) or 0.0) for item in items]
                ),
                "raw_cache_hits_total": float(
                    sum(float(item.get("raw_cache_hit", 0.0) or 0.0) for item in items)
                ),
                "prepared_cache_hits_total": float(
                    sum(float(item.get("prepared_cache_hit", 0.0) or 0.0) for item in items)
                ),
                "peak_memory_mb_max": max(
                    [float(item.get("peak_memory_mb", 0.0) or 0.0) for item in items],
                    default=0.0,
                ),
                "points_per_second_mean": _mean(
                    [float(item.get("points_per_second", 0.0) or 0.0) for item in items]
                ),
                "windows_per_second_mean": _mean(
                    [float(item.get("windows_per_second", 0.0) or 0.0) for item in items]
                ),
            }
        )
    for suffix in conformal_levels:
        for metric in ("coverage", "mean_width", "interval_score"):
            metric_key = f"{metric}_{suffix}"
            values = [
                float(item[metric_key])
                for item in ok_items
                if _as_float(item.get(metric_key)) is not None
            ]
            summary[f"{metric_key}_mean"] = _mean(values)
    return summary


def _benchmark_task_label(
    dataset_fields: dict[str, int | str],
    model_cases: tuple[dict[str, Any], ...],
) -> str:
    dataset_key = str(dataset_fields["dataset_key"])
    if len(model_cases) == 1:
        return f"{dataset_key}/{model_cases[0]['key']}"
    return f"{dataset_key}/[{len(model_cases)} models]"


def _benchmark_model_case_chunks(
    model_cases: list[dict[str, Any]],
    *,
    chunk_size: int,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    if chunk_size < 0:
        raise ValueError("--chunk-size must be >= 0")
    if not model_cases:
        return ()
    if chunk_size == 0:
        return (tuple(model_cases),)
    return tuple(tuple(model_cases[i : i + chunk_size]) for i in range(0, len(model_cases), chunk_size))


def _resolve_benchmark_chunk_size(
    raw_chunk_size: object,
    *,
    dataset_count: int,
    model_count: int,
    jobs: int,
) -> int:
    from foresight.batch_execution import resolve_auto_chunk_size

    return resolve_auto_chunk_size(
        raw_chunk_size,
        dataset_count=dataset_count,
        model_count=model_count,
        jobs=jobs,
    )


def _build_benchmark_tasks(
    *,
    dataset_fields_list: list[dict[str, int | str]],
    model_cases: list[dict[str, Any]],
    requested_chunk_size: object,
    chunk_size: int,
) -> list[Any]:
    from foresight.batch_execution import BatchTask

    tasks: list[Any] = []
    for dataset_fields in dataset_fields_list:
        for chunk in _benchmark_model_case_chunks(model_cases, chunk_size=chunk_size):
            tasks.append(
                BatchTask(
                    label=_benchmark_task_label(dataset_fields, chunk),
                    task_args=(dict(dataset_fields), tuple(dict(case) for case in chunk)),
                    task_scope="benchmark",
                    dataset=str(dataset_fields["dataset_key"]),
                    model_count=len(chunk),
                    requested_chunk_size=str(requested_chunk_size),
                    resolved_chunk_size=int(chunk_size),
                )
            )
    return tasks


def _benchmark_rows_for_task(
    dataset_fields: dict[str, int | str],
    model_cases: tuple[dict[str, Any], ...],
    data_dir: str | Path | None,
    conformal_levels: list[int],
    conformal_per_step: bool,
    task_group: str,
    profiling: bool,
    workload: str,
    scale: str,
    benchmark_row_builder: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    from foresight.dataset_long_df_cache import get_or_build_dataset_long_df
    from foresight.eval_forecast import eval_model_long_df
    from foresight.models.registry import get_model_spec

    task_started = time.perf_counter()
    frame_bundles: dict[tuple[Any, ...], dict[str, Any]] = {}
    frame_use_counts: dict[tuple[Any, ...], int] = {}
    rows: list[dict[str, Any]] = []
    for model_case in model_cases:
        model_key = str(model_case["key"])
        frame_key = _benchmark_frame_request_key(
            dataset_fields=dataset_fields,
            model_params=dict(model_case.get("params", {})),
            data_dir=data_dir,
        )
        frame_bundle = frame_bundles.get(frame_key)
        if frame_bundle is None:
            frame_bundle = _get_or_build_benchmark_frame(
                dataset_fields=dataset_fields,
                model_params=dict(model_case.get("params", {})),
                data_dir=data_dir,
                get_or_build_dataset_long_df=get_or_build_dataset_long_df,
            )
            frame_bundles[frame_key] = frame_bundle
            frame_use_counts[frame_key] = 0
        use_count = int(frame_use_counts.get(frame_key, 0))
        frame_use_counts[frame_key] = use_count + 1
        effective_frame_bundle = (
            frame_bundle
            if use_count == 0
            else _benchmark_reused_frame_bundle(frame_bundle)
        )
        try:
            spec = get_model_spec(model_key)
            requires = tuple(str(item).strip().lower() for item in getattr(spec, "requires", ()) if str(item).strip())
            backend_family = "core" if not requires else str(requires[0])
        except Exception:  # noqa: BLE001
            backend_family = "unknown"

        rows.append(
            _benchmark_result_row_from_frame_bundle(
                eval_model_long_df,
                effective_frame_bundle,
                dataset_fields,
                model_case,
                data_dir=data_dir,
                conformal_levels=conformal_levels,
                conformal_per_step=conformal_per_step,
                task_group=task_group,
                backend_family=backend_family,
                profiling=profiling,
                workload=workload,
                scale=scale,
            )
        )
    task_elapsed = float(time.perf_counter() - task_started)
    accounted_seconds = float(
        sum(float(row.get("cv_seconds", 0.0) or 0.0) for row in rows)
    )
    dispatch_seconds = max(0.0, task_elapsed - accounted_seconds)
    dispatch_per_row = 0.0 if not rows else dispatch_seconds / float(len(rows))
    for row in rows:
        row["dispatch_seconds"] = round(float(dispatch_per_row), 6)
    return rows, []


def _budget_metric_key(budget_key: str) -> str:
    if budget_key.endswith("_warn"):
        return budget_key[: -len("_warn")]
    if budget_key.endswith("_fail"):
        return budget_key[: -len("_fail")]
    return budget_key


def _benchmark_budget_findings(
    summary_rows: list[dict[str, Any]],
    budgets: dict[str, float],
) -> list[str]:
    findings: list[str] = []
    for budget_key, threshold in sorted(budgets.items()):
        metric_key = _budget_metric_key(str(budget_key))
        for row in summary_rows:
            actual = _as_float(row.get(metric_key))
            if actual is None or float(actual) <= float(threshold):
                continue
            findings.append(
                f"BUDGET {budget_key} exceeded for {row.get('model', 'unknown')}: "
                f"{metric_key}={float(actual):.6g} > {float(threshold):.6g}"
            )
    return findings


def _benchmark_dataset_case_fields(dataset_case: Any) -> dict[str, int | str]:
    return {
        "dataset_key": str(dataset_case["key"]),
        "y_col": str(dataset_case["y_col"]),
        "horizon": int(dataset_case["horizon"]),
        "step": int(dataset_case["step"]),
        "min_train_size": int(dataset_case["min_train_size"]),
        "max_windows": int(dataset_case["max_windows"]),
    }


def _get_or_build_benchmark_frame(
    *,
    dataset_fields: dict[str, int | str],
    model_params: dict[str, Any],
    data_dir: str | Path | None,
    get_or_build_dataset_long_df: Any,
) -> dict[str, Any]:
    dataset_key = str(dataset_fields["dataset_key"])
    y_col = str(dataset_fields["y_col"])
    frame_bundle = get_or_build_dataset_long_df(
        dataset=dataset_key,
        y_col=y_col,
        data_dir=data_dir,
        model_params=model_params,
    )

    return {
        "spec": frame_bundle["spec"],
        "y_col_final": str(frame_bundle["y_col_final"]),
        "long_df": frame_bundle["long_df"],
        "load_seconds": float(frame_bundle["load_seconds"]),
        "prepare_seconds": float(frame_bundle["prepare_seconds"]),
        "raw_cache_hit": bool(frame_bundle.get("raw_cache_hit", False)),
        "prepared_cache_hit": bool(frame_bundle.get("prepared_cache_hit", False)),
    }


def _benchmark_frame_request_key(
    *,
    dataset_fields: dict[str, int | str],
    model_params: dict[str, Any],
    data_dir: str | Path | None,
) -> tuple[Any, ...]:
    from foresight.contracts.params import normalize_covariate_roles, normalize_static_cols

    historic_x_cols, future_x_cols = normalize_covariate_roles(model_params)
    static_cols = normalize_static_cols(model_params)
    data_dir_s = "" if data_dir is None else str(data_dir).strip()
    return (
        str(dataset_fields["dataset_key"]),
        str(dataset_fields["y_col"]),
        data_dir_s,
        tuple(historic_x_cols),
        tuple(future_x_cols),
        tuple(static_cols),
    )


def _benchmark_result_row_from_frame_bundle(
    eval_model_long_df: Any,
    frame_bundle: dict[str, Any],
    dataset_fields: dict[str, int | str],
    model_case: Any,
    *,
    data_dir: str | Path | None,
    conformal_levels: list[int],
    conformal_per_step: bool,
    task_group: str,
    backend_family: str,
    profiling: bool,
    workload: str,
    scale: str,
) -> dict[str, Any]:
    model_key = str(model_case["key"])
    params = dict(model_case.get("params", {}))
    dataset_key = str(dataset_fields["dataset_key"])
    y_col = str(frame_bundle.get("y_col_final", dataset_fields["y_col"]))
    horizon = int(dataset_fields["horizon"])
    step = int(dataset_fields["step"])
    min_train_size = int(dataset_fields["min_train_size"])
    max_windows = int(dataset_fields["max_windows"])
    load_elapsed = float(frame_bundle.get("load_seconds", 0.0) or 0.0)
    prepare_elapsed = float(frame_bundle.get("prepare_seconds", 0.0) or 0.0)
    eval_started = 0.0
    peak_memory_mb = 0.0
    long_df = frame_bundle["long_df"]
    try:
        if profiling:
            tracemalloc.start()
        if long_df.empty:
            raise ValueError("Loaded 0 rows after to_long(dropna=True). Check dataset and y_col.")

        eval_started = time.perf_counter()
        payload = eval_model_long_df(
            model=model_key,
            long_df=long_df,
            horizon=horizon,
            step=step,
            min_train_size=min_train_size,
            max_windows=max_windows,
            model_params=params,
            conformal_levels=conformal_levels,
            conformal_per_step=conformal_per_step,
        )
        if profiling:
            _current, peak_bytes = tracemalloc.get_traced_memory()
            peak_memory_mb = float(peak_bytes) / (1024.0 * 1024.0)
    except Exception as e:  # noqa: BLE001
        if profiling and tracemalloc.is_tracing():
            tracemalloc.stop()
        return {
            "model": model_key,
            "task_group": str(task_group),
            "backend_family": str(backend_family),
            "workload": str(workload),
            "scale": str(scale),
            "status": "skip",
            "skip_reason": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "dataset": dataset_key,
            "y_col": y_col,
            "horizon": horizon,
            "step": step,
            "min_train_size": min_train_size,
            "max_windows": max_windows,
            "n_windows": 0,
            "n_series": 0,
            "n_series_skipped": 0,
            "n_points": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
            "smape": None,
            "load_seconds": round(float(load_elapsed), 6),
            "prepare_seconds": round(float(prepare_elapsed), 6),
            "eval_seconds": 0.0,
            "dispatch_seconds": 0.0,
            "raw_cache_hit": int(bool(frame_bundle.get("raw_cache_hit", False))),
            "prepared_cache_hit": int(bool(frame_bundle.get("prepared_cache_hit", False))),
            "cv_seconds": round(float(load_elapsed + prepare_elapsed), 6),
            "peak_memory_mb": round(float(peak_memory_mb), 6),
            "points_per_second": 0.0,
            "windows_per_second": 0.0,
        }
    finally:
        if profiling and tracemalloc.is_tracing():
            tracemalloc.stop()
    eval_elapsed = time.perf_counter() - eval_started
    elapsed = float(load_elapsed + prepare_elapsed + eval_elapsed)
    n_points = int(payload["n_points"])
    n_windows = int(payload.get("n_windows", 0) or 0)
    points_per_second = 0.0 if elapsed <= 0.0 else float(n_points) / elapsed
    windows_per_second = 0.0 if elapsed <= 0.0 else float(n_windows) / elapsed

    row: dict[str, Any] = {
        "model": model_key,
        "task_group": str(task_group),
        "backend_family": str(backend_family),
        "workload": str(workload),
        "scale": str(scale),
        "status": "ok",
        "skip_reason": "",
        "error_type": "",
        "error_message": "",
        "dataset": dataset_key,
        "y_col": y_col,
        "horizon": horizon,
        "step": step,
        "min_train_size": min_train_size,
        "max_windows": max_windows,
        "n_windows": n_windows,
        "n_series": int(payload["n_series"]),
        "n_series_skipped": int(payload["n_series_skipped"]),
        "n_points": n_points,
        "mae": float(payload["mae"]),
        "rmse": float(payload["rmse"]),
        "mape": float(payload["mape"]),
        "smape": float(payload["smape"]),
        "load_seconds": round(float(load_elapsed), 6),
        "prepare_seconds": round(float(prepare_elapsed), 6),
        "eval_seconds": round(float(eval_elapsed), 6),
        "dispatch_seconds": 0.0,
        "raw_cache_hit": int(bool(frame_bundle.get("raw_cache_hit", False))),
        "prepared_cache_hit": int(bool(frame_bundle.get("prepared_cache_hit", False))),
        "cv_seconds": round(float(elapsed), 6),
        "peak_memory_mb": round(float(peak_memory_mb), 6),
        "points_per_second": round(float(points_per_second), 6),
        "windows_per_second": round(float(windows_per_second), 6),
    }
    for suffix in conformal_levels:
        for metric in ("coverage", "mean_width", "interval_score"):
            payload_key = f"{metric}_{suffix}"
            if payload_key in payload:
                row[payload_key] = float(payload[payload_key])
    return row


def _benchmark_reused_frame_bundle(frame_bundle: dict[str, Any]) -> dict[str, Any]:
    reused = dict(frame_bundle)
    reused["load_seconds"] = 0.0
    reused["prepare_seconds"] = 0.0
    reused["raw_cache_hit"] = False
    reused["prepared_cache_hit"] = False
    return reused


def _benchmark_result_row(
    eval_model_long_df: Any,
    get_or_build_dataset_long_df: Any,
    dataset_fields: dict[str, int | str],
    model_case: Any,
    *,
    data_dir: str | Path | None,
    conformal_levels: list[int],
    conformal_per_step: bool,
    task_group: str,
    backend_family: str,
    profiling: bool,
    workload: str,
    scale: str,
) -> dict[str, Any]:
    frame_bundle = _get_or_build_benchmark_frame(
        dataset_fields=dataset_fields,
        model_params=dict(model_case.get("params", {})),
        data_dir=data_dir,
        get_or_build_dataset_long_df=get_or_build_dataset_long_df,
    )
    return _benchmark_result_row_from_frame_bundle(
        eval_model_long_df,
        frame_bundle,
        dataset_fields,
        model_case,
        data_dir=data_dir,
        conformal_levels=conformal_levels,
        conformal_per_step=conformal_per_step,
        task_group=task_group,
        backend_family=backend_family,
        profiling=profiling,
        workload=workload,
        scale=scale,
    )


def run_benchmark_suite(
    *,
    config_name: str,
    data_dir: str | Path | None = None,
    profile: bool | None = None,
    jobs: int = 1,
    backend: str = "process",
    chunk_size: int | str = 1,
    progress: bool = False,
    budget_mode: str = "warn",
) -> dict[str, Any]:
    root = _repo_root()
    _ensure_src_on_path(root)

    batch_execution = _get_batch_execution_module()

    all_config = _load_benchmark_config()
    if config_name not in all_config:
        raise KeyError(
            f"Unknown benchmark config: {config_name!r}. Available: {', '.join(sorted(all_config))}"
        )

    config = _validate_benchmark_config(config_name, all_config[config_name])

    datasets = config.get("datasets", [])
    models = config.get("models", [])
    task_group = str(config.get("task_group", "point")).strip() or "point"
    workload = str(config.get("workload", "panel_cv")).strip() or "panel_cv"
    scale = str(config.get("scale", "small")).strip() or "small"
    profiling = bool(config.get("profiling", False))
    budgets = dict(config.get("budgets", {}))
    if profile is not None:
        profiling = bool(profile)
    backend_s = str(backend).strip().lower() or "process"
    if backend_s not in _BENCHMARK_BACKENDS:
        raise ValueError(
            f"--backend must be one of: {', '.join(sorted(_BENCHMARK_BACKENDS))}"
        )
    if int(jobs) <= 0:
        raise ValueError("--jobs must be >= 1")
    budget_mode_s = str(budget_mode).strip().lower() or "warn"
    if budget_mode_s not in _BUDGET_MODES:
        raise ValueError(
            f"--budget-mode must be one of: {', '.join(sorted(_BUDGET_MODES))}"
        )
    dataset_fields_list = [_benchmark_dataset_case_fields(case) for case in datasets]
    resolved_chunk_size = _resolve_benchmark_chunk_size(
        chunk_size,
        dataset_count=len(dataset_fields_list),
        model_count=len(models),
        jobs=int(jobs),
    )
    conformal_levels = [int(float(level)) for level in config.get("conformal_levels", [])]
    conformal_per_step = bool(config.get("conformal_per_step", False))
    benchmark_row_builder = _benchmark_result_row
    task_stats: list[Any] = []

    tasks = _build_benchmark_tasks(
        dataset_fields_list=dataset_fields_list,
        model_cases=[dict(case) for case in models],
        requested_chunk_size=chunk_size,
        chunk_size=resolved_chunk_size,
    )
    rows, failures = batch_execution.run_batch_tasks(
        tasks,
        jobs=int(jobs),
        backend=backend_s,
        progress=bool(progress),
        strict=False,
        worker=_benchmark_rows_for_task,
        worker_args=(
            data_dir,
            conformal_levels,
            conformal_per_step,
            task_group,
            profiling,
            workload,
            scale,
            benchmark_row_builder,
        ),
        stats_out=task_stats,
    )

    rows.sort(key=lambda row: (str(row["dataset"]), str(row["model"])))
    summary = _summarize_rows(
        rows,
        conformal_levels=conformal_levels,
        include_profile=profiling,
    )
    budget_findings = _benchmark_budget_findings(summary, budgets)
    return {
        "config": config_name,
        "description": str(config.get("description", "")),
        "task_group": task_group,
        "workload": workload,
        "scale": scale,
        "profiling": profiling,
        "jobs": int(jobs),
        "backend": backend_s,
        "chunk_size": int(resolved_chunk_size),
        "progress": bool(progress),
        "budget_mode": budget_mode_s,
        "failures": int(failures),
        "budgets": budgets,
        "budget_findings": budget_findings,
        "datasets": [str(item["dataset_key"]) for item in dataset_fields_list],
        "models": [str(item["key"]) for item in models],
        "rows": rows,
        "task_reports": batch_execution.task_report_rows(
            task_stats,
            backend=backend_s,
            jobs=int(jobs),
        ),
        "summary": summary,
        "conformal_levels": conformal_levels,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a small reproducible benchmark sweep on packaged ForeSight datasets."
    )
    parser.add_argument(
        "--config",
        default="baseline",
        help="Benchmark config name from benchmarks/benchmark_config.json (default: baseline).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shortcut for --config smoke.",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "json", "md"],
        help="Summary output format (default: csv).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the formatted summary output.",
    )
    parser.add_argument(
        "--task-reports-output",
        default="",
        help="Optional path to write per-task execution stats.",
    )
    parser.add_argument(
        "--task-reports-format",
        choices=["csv", "json", "md"],
        default="json",
        help="Task report output format (default: json).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional dataset base directory override.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Force per-stage profiling columns in the summary output.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of benchmark worker tasks to run concurrently (default: 1).",
    )
    parser.add_argument(
        "--backend",
        choices=["thread", "process"],
        default="process",
        help="Parallel backend for benchmark task execution (default: process).",
    )
    parser.add_argument(
        "--chunk-size",
        type=str,
        default="1",
        help="Models per dataset task; use 0 for dataset-wide tasks or auto for jobs-aware chunking.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Emit per-task DONE progress lines to stderr.",
    )
    parser.add_argument(
        "--budget-mode",
        choices=["warn", "fail"],
        default="warn",
        help="Whether benchmark budget regressions warn or fail the command (default: warn).",
    )
    args = parser.parse_args(argv)

    config_name = "smoke" if bool(args.smoke) else str(args.config)

    cli_shared = _get_cli_shared_module()
    payload = run_benchmark_suite(
        config_name=config_name,
        data_dir=args.data_dir,
        profile=True if bool(args.profile) else None,
        jobs=int(args.jobs),
        backend=str(args.backend),
        chunk_size=str(args.chunk_size),
        progress=bool(args.progress),
        budget_mode=str(args.budget_mode),
    )
    cli_shared._emit_table(
        payload["summary"],
        columns=_summary_columns(
            list(payload["conformal_levels"]),
            include_profile=bool(payload.get("profiling", False)),
        ),
        output=str(args.output),
        fmt=str(args.format),
    )
    task_reports_output = str(args.task_reports_output).strip()
    if task_reports_output:
        _get_batch_execution_module().write_task_reports(
            list(payload.get("task_reports", [])),
            fmt=str(args.task_reports_format),
            output=task_reports_output,
        )
    budget_findings = list(
        payload.get(
            "budget_findings",
            _benchmark_budget_findings(
                list(payload.get("summary", [])),
                dict(payload.get("budgets", {})),
            ),
        )
    )
    for line in budget_findings:
        print(line, file=sys.stderr)
    return 1 if budget_findings and str(args.budget_mode) == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
