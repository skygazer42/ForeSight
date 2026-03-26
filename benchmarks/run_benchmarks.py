#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_TASK_GROUPS = {"point", "probabilistic", "covariate"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path(root: Path) -> None:
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


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
        items = by_model[(task_group, backend_family, model)]
        ok_items = [item for item in items if str(item.get("status", "ok")).strip().lower() == "ok"]
        summary: dict[str, Any] = {
            "model": model,
            "task_group": task_group,
            "backend_family": backend_family,
            "n_datasets": int(len({str(item["dataset"]) for item in items})),
            "ok_rows": int(len(ok_items)),
            "skip_rows": int(len(items) - len(ok_items)),
            "n_points_total": int(sum(int(item.get("n_points", 0) or 0) for item in ok_items)),
            "mae_mean": _mean(
                [float(item["mae"]) for item in ok_items if _as_float(item.get("mae")) is not None]
            ),
            "rmse_mean": _mean(
                [
                    float(item["rmse"])
                    for item in ok_items
                    if _as_float(item.get("rmse")) is not None
                ]
            ),
            "mape_mean": _mean(
                [
                    float(item["mape"])
                    for item in ok_items
                    if _as_float(item.get("mape")) is not None
                ]
            ),
            "smape_mean": _mean(
                [
                    float(item["smape"])
                    for item in ok_items
                    if _as_float(item.get("smape")) is not None
                ]
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
        out.append(summary)

    return out


def _format_csv(rows: list[dict[str, Any]], *, columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buf.getvalue().rstrip("\n")


def _format_markdown(rows: list[dict[str, Any]], *, columns: list[str]) -> str:
    def _fmt(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _format_summary(
    rows: list[dict[str, Any]],
    *,
    conformal_levels: list[int],
    fmt: str,
    include_profile: bool,
) -> str:
    columns = _summary_columns(conformal_levels, include_profile=include_profile)
    if fmt == "csv":
        return _format_csv(rows, columns=columns)
    if fmt == "json":
        return json.dumps(rows, ensure_ascii=False, sort_keys=True)
    if fmt == "md":
        return _format_markdown(rows, columns=columns)
    raise ValueError(f"Unknown format: {fmt!r}")


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
        "long_df": frame_bundle["long_df"],
        "load_seconds": float(frame_bundle["load_seconds"]),
        "prepare_seconds": float(frame_bundle["prepare_seconds"]),
    }


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
) -> dict[str, Any]:
    model_key = str(model_case["key"])
    params = dict(model_case.get("params", {}))
    dataset_key = str(dataset_fields["dataset_key"])
    y_col = str(dataset_fields["y_col"])
    horizon = int(dataset_fields["horizon"])
    step = int(dataset_fields["step"])
    min_train_size = int(dataset_fields["min_train_size"])
    max_windows = int(dataset_fields["max_windows"])
    load_elapsed = 0.0
    prepare_elapsed = 0.0
    try:
        frame_bundle = _get_or_build_benchmark_frame(
            dataset_fields=dataset_fields,
            model_params=params,
            data_dir=data_dir,
            get_or_build_dataset_long_df=get_or_build_dataset_long_df,
        )
        long_df = frame_bundle["long_df"]
        load_elapsed = float(frame_bundle["load_seconds"])
        prepare_elapsed = float(frame_bundle["prepare_seconds"])
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
    except Exception as e:  # noqa: BLE001
        return {
            "model": model_key,
            "task_group": str(task_group),
            "backend_family": str(backend_family),
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
            "cv_seconds": round(float(load_elapsed + prepare_elapsed), 6),
        }
    eval_elapsed = time.perf_counter() - eval_started
    elapsed = float(load_elapsed + prepare_elapsed + eval_elapsed)

    row: dict[str, Any] = {
        "model": model_key,
        "task_group": str(task_group),
        "backend_family": str(backend_family),
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
        "n_series": int(payload["n_series"]),
        "n_series_skipped": int(payload["n_series_skipped"]),
        "n_points": int(payload["n_points"]),
        "mae": float(payload["mae"]),
        "rmse": float(payload["rmse"]),
        "mape": float(payload["mape"]),
        "smape": float(payload["smape"]),
        "load_seconds": round(float(load_elapsed), 6),
        "prepare_seconds": round(float(prepare_elapsed), 6),
        "eval_seconds": round(float(eval_elapsed), 6),
        "cv_seconds": round(float(elapsed), 6),
    }
    for suffix in conformal_levels:
        for metric in ("coverage", "mean_width", "interval_score"):
            payload_key = f"{metric}_{suffix}"
            if payload_key in payload:
                row[payload_key] = float(payload[payload_key])
    return row


def run_benchmark_suite(
    *,
    config_name: str,
    data_dir: str | Path | None = None,
    profile: bool | None = None,
) -> dict[str, Any]:
    root = _repo_root()
    _ensure_src_on_path(root)

    from foresight.dataset_long_df_cache import get_or_build_dataset_long_df
    from foresight.eval_forecast import eval_model_long_df
    from foresight.models.registry import get_model_spec

    all_config = _load_benchmark_config()
    if config_name not in all_config:
        raise KeyError(
            f"Unknown benchmark config: {config_name!r}. Available: {', '.join(sorted(all_config))}"
        )

    config = _validate_benchmark_config(config_name, all_config[config_name])

    datasets = config.get("datasets", [])
    models = config.get("models", [])
    task_group = str(config.get("task_group", "point")).strip() or "point"
    profiling = bool(config.get("profiling", False))
    if profile is not None:
        profiling = bool(profile)
    dataset_fields_list = [_benchmark_dataset_case_fields(case) for case in datasets]
    conformal_levels = [int(float(level)) for level in config.get("conformal_levels", [])]
    conformal_per_step = bool(config.get("conformal_per_step", False))

    rows: list[dict[str, Any]] = []
    for dataset_fields in dataset_fields_list:
        for model_case in models:
            model_key = str(model_case["key"])
            try:
                backend_family = (
                    "core"
                    if not tuple(get_model_spec(model_key).requires)
                    else str(get_model_spec(model_key).requires[0]).strip().lower()
                )
            except Exception:  # noqa: BLE001
                backend_family = "unknown"
            rows.append(
                _benchmark_result_row(
                    eval_model_long_df,
                    get_or_build_dataset_long_df,
                    dataset_fields,
                    model_case,
                    data_dir=data_dir,
                    conformal_levels=conformal_levels,
                    conformal_per_step=conformal_per_step,
                    task_group=task_group,
                    backend_family=backend_family,
                )
            )

    rows.sort(key=lambda row: (str(row["dataset"]), str(row["model"])))
    summary = _summarize_rows(
        rows,
        conformal_levels=conformal_levels,
        include_profile=profiling,
    )
    return {
        "config": config_name,
        "description": str(config.get("description", "")),
        "task_group": task_group,
        "profiling": profiling,
        "datasets": [str(item["dataset_key"]) for item in dataset_fields_list],
        "models": [str(item["key"]) for item in models],
        "rows": rows,
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
        "--data-dir",
        default=None,
        help="Optional dataset base directory override.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Force per-stage profiling columns in the summary output.",
    )
    args = parser.parse_args(argv)

    config_name = "smoke" if bool(args.smoke) else str(args.config)
    payload = run_benchmark_suite(
        config_name=config_name,
        data_dir=args.data_dir,
        profile=True if bool(args.profile) else None,
    )
    text = _format_summary(
        payload["summary"],
        conformal_levels=list(payload["conformal_levels"]),
        fmt=str(args.format),
        include_profile=bool(payload.get("profiling", False)),
    )
    print(text)

    output = str(args.output).strip()
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
