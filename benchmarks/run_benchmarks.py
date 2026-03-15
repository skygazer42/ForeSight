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


def _summary_columns(level_suffixes: list[int]) -> list[str]:
    interval_cols: list[str] = []
    for suffix in level_suffixes:
        interval_cols.extend(
            [
                f"coverage_{suffix}_mean",
                f"mean_width_{suffix}_mean",
                f"interval_score_{suffix}_mean",
            ]
        )
    return [
        "model",
        "n_datasets",
        "n_points_total",
        "mae_mean",
        "rmse_mean",
        "mape_mean",
        "smape_mean",
        *interval_cols,
        "wall_clock_seconds_total",
        "wall_clock_seconds_mean",
    ]


def _summarize_rows(
    rows: list[dict[str, Any]], *, conformal_levels: list[int]
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)

    out: list[dict[str, Any]] = []
    for model in sorted(by_model):
        items = by_model[model]
        summary: dict[str, Any] = {
            "model": model,
            "n_datasets": int(len({str(item["dataset"]) for item in items})),
            "n_points_total": int(sum(int(item.get("n_points", 0) or 0) for item in items)),
            "mae_mean": _mean(
                [float(item["mae"]) for item in items if _as_float(item.get("mae")) is not None]
            ),
            "rmse_mean": _mean(
                [float(item["rmse"]) for item in items if _as_float(item.get("rmse")) is not None]
            ),
            "mape_mean": _mean(
                [float(item["mape"]) for item in items if _as_float(item.get("mape")) is not None]
            ),
            "smape_mean": _mean(
                [float(item["smape"]) for item in items if _as_float(item.get("smape")) is not None]
            ),
            "wall_clock_seconds_total": float(
                sum(float(item["wall_clock_seconds"]) for item in items)
            ),
            "wall_clock_seconds_mean": _mean([float(item["wall_clock_seconds"]) for item in items]),
        }
        for suffix in conformal_levels:
            for metric in ("coverage", "mean_width", "interval_score"):
                key = f"{metric}_{suffix}"
                values = [
                    float(item[key]) for item in items if _as_float(item.get(key)) is not None
                ]
                summary[f"{key}_mean"] = _mean(values)
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


def _format_summary(rows: list[dict[str, Any]], *, conformal_levels: list[int], fmt: str) -> str:
    columns = _summary_columns(conformal_levels)
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


def _benchmark_result_row(
    eval_model: Any,
    dataset_fields: dict[str, int | str],
    model_case: Any,
    *,
    data_dir: str | Path | None,
    conformal_levels: list[int],
    conformal_per_step: bool,
) -> dict[str, Any]:
    model_key = str(model_case["key"])
    params = dict(model_case.get("params", {}))
    started = time.perf_counter()
    payload = eval_model(
        model=model_key,
        dataset=str(dataset_fields["dataset_key"]),
        y_col=str(dataset_fields["y_col"]),
        horizon=int(dataset_fields["horizon"]),
        step=int(dataset_fields["step"]),
        min_train_size=int(dataset_fields["min_train_size"]),
        max_windows=int(dataset_fields["max_windows"]),
        model_params=params,
        data_dir=data_dir,
        conformal_levels=conformal_levels,
        conformal_per_step=conformal_per_step,
    )
    elapsed = time.perf_counter() - started

    row: dict[str, Any] = {
        "model": model_key,
        "dataset": str(dataset_fields["dataset_key"]),
        "y_col": str(dataset_fields["y_col"]),
        "horizon": int(dataset_fields["horizon"]),
        "step": int(dataset_fields["step"]),
        "min_train_size": int(dataset_fields["min_train_size"]),
        "max_windows": int(dataset_fields["max_windows"]),
        "n_series": int(payload["n_series"]),
        "n_series_skipped": int(payload["n_series_skipped"]),
        "n_points": int(payload["n_points"]),
        "mae": float(payload["mae"]),
        "rmse": float(payload["rmse"]),
        "mape": float(payload["mape"]),
        "smape": float(payload["smape"]),
        "wall_clock_seconds": round(float(elapsed), 6),
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
) -> dict[str, Any]:
    root = _repo_root()
    _ensure_src_on_path(root)

    from foresight.eval_forecast import eval_model

    all_config = _load_benchmark_config()
    if config_name not in all_config:
        raise KeyError(
            f"Unknown benchmark config: {config_name!r}. Available: {', '.join(sorted(all_config))}"
        )

    config = all_config[config_name]
    if not isinstance(config, dict):
        raise TypeError(f"benchmark config {config_name!r} must be a JSON object")

    datasets = config.get("datasets", [])
    models = config.get("models", [])
    dataset_fields_list = [_benchmark_dataset_case_fields(case) for case in datasets]
    conformal_levels = [int(float(level)) for level in list(config.get("conformal_levels", []))]
    conformal_per_step = bool(config.get("conformal_per_step", False))

    rows: list[dict[str, Any]] = []
    for dataset_fields in dataset_fields_list:
        for model_case in models:
            rows.append(
                _benchmark_result_row(
                    eval_model,
                    dataset_fields,
                    model_case,
                    data_dir=data_dir,
                    conformal_levels=conformal_levels,
                    conformal_per_step=conformal_per_step,
                )
            )

    rows.sort(key=lambda row: (str(row["dataset"]), str(row["model"])))
    summary = _summarize_rows(rows, conformal_levels=conformal_levels)
    return {
        "config": config_name,
        "description": str(config.get("description", "")),
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
    args = parser.parse_args(argv)

    config_name = "smoke" if bool(args.smoke) else str(args.config)
    payload = run_benchmark_suite(config_name=config_name, data_dir=args.data_dir)
    text = _format_summary(
        payload["summary"],
        conformal_levels=list(payload["conformal_levels"]),
        fmt=str(args.format),
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
