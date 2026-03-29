from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import cli_catalog as _cli_catalog
from . import cli_data as _cli_data
from . import cli_leaderboard as _cli_leaderboard
from . import cli_runtime as _cli_runtime
from . import cli_shared as _cli_shared

_MODEL_KEY_HELP = "Model key (see: `foresight models list`)"
_DATASET_KEY_HELP = "Dataset key"
_FORECAST_HORIZON_HELP = "Forecast horizon"
_WALK_FORWARD_STEP_HELP = "Walk-forward step size"
_MIN_TRAIN_SIZE_FIRST_WINDOW_HELP = "Minimum training size for first window"
_MAX_WINDOWS_LIMIT_HELP = "Optional limit on the number of walk-forward windows"
_OUTPUT_PATH_HELP = "Optional path to write output"
_OUTPUT_JSON_FORMAT_HELP = "Output format (default: json)"
_OUTPUT_ROWS_FORMAT_HELP = "Output format (default: csv)"
_TARGET_COLUMN_HELP = "Target column name"
_DEFAULT_TARGET_COLUMN_HELP = "Optional target column name (default: use dataset spec default_y)."
_EXPANDING_WINDOW_HELP = "Optional rolling train window size (default: expanding window)."
_MODEL_PARAM_EXAMPLE_HELP = (
    "Model parameter as key=value (repeatable). Example: --model-param season_length=12"
)
_METRICS_OUTPUT_PATH_HELP = "Optional path to write metrics output"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="foresight",
        description="ForeSight: time-series forecasting models and utilities.",
    )
    p.add_argument("--version", action="store_true", help="Print version and exit.")
    p.add_argument(
        "--list",
        action="store_true",
        help="Shortcut for `foresight models list`.",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Explicit shortcut for `foresight models list`.",
    )
    p.add_argument(
        "--list-datasets",
        action="store_true",
        help="Shortcut for `foresight datasets list`.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Show full tracebacks on errors.",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Base directory to resolve dataset files (overrides FORESIGHT_DATA_DIR).",
    )

    sub = p.add_subparsers(dest="command")
    _cli_catalog.register_catalog_subparsers(sub)

    doctor = sub.add_parser("doctor", help="Inspect environment, datasets, and optional dependencies")
    doctor.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help=_OUTPUT_JSON_FORMAT_HELP,
    )
    doctor.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    doctor.set_defaults(_handler=_cmd_doctor)

    cv = sub.add_parser("cv", help="Cross-validation utilities")
    cv_sub = cv.add_subparsers(dest="cv_command", required=True)

    cv_run = cv_sub.add_parser("run", help="Run rolling-origin CV and output predictions")
    cv_run.add_argument("--model", required=True, help=_MODEL_KEY_HELP)
    cv_run.add_argument("--dataset", required=True, help=_DATASET_KEY_HELP)
    cv_run.add_argument(
        "--y-col",
        type=str,
        default="",
        help=_DEFAULT_TARGET_COLUMN_HELP,
    )
    cv_run.add_argument("--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP)
    cv_run.add_argument("--step-size", type=int, default=1, help="CV step size (default: 1)")
    cv_run.add_argument("--min-train-size", type=int, required=True, help="Minimum train size")
    cv_run.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help=_EXPANDING_WINDOW_HELP,
    )
    cv_run.add_argument(
        "--n-windows",
        type=int,
        default=None,
        help="Optional limit to the last N CV windows.",
    )
    cv_run.add_argument(
        "--model-param",
        action="append",
        default=[],
        help=_MODEL_PARAM_EXAMPLE_HELP,
    )
    cv_run.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    cv_run.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format for predictions (default: csv)",
    )
    _cli_runtime.register_runtime_logging_args(cv_run)
    cv_run.set_defaults(_handler=_cmd_cv_run)

    cv_csv = cv_sub.add_parser("csv", help="Run rolling-origin CV on an arbitrary CSV file")
    cv_csv.add_argument("--model", required=True, help=_MODEL_KEY_HELP)
    cv_csv.add_argument("--path", required=True, help="Path to a CSV file")
    cv_csv.add_argument("--time-col", required=True, help="Time column name")
    cv_csv.add_argument("--y-col", required=True, help=_TARGET_COLUMN_HELP)
    cv_csv.add_argument(
        "--id-cols",
        type=str,
        default="",
        help="Optional comma-separated ID columns for panel data (e.g. store,dept)",
    )
    cv_csv.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse the time column as datetime.",
    )
    cv_csv.add_argument("--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP)
    cv_csv.add_argument("--step-size", type=int, default=1, help="CV step size (default: 1)")
    cv_csv.add_argument("--min-train-size", type=int, required=True, help="Minimum train size")
    cv_csv.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help=_EXPANDING_WINDOW_HELP,
    )
    cv_csv.add_argument(
        "--n-windows",
        type=int,
        default=None,
        help="Optional limit to the last N CV windows.",
    )
    cv_csv.add_argument(
        "--model-param",
        action="append",
        default=[],
        help=_MODEL_PARAM_EXAMPLE_HELP,
    )
    cv_csv.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    cv_csv.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format for predictions (default: csv)",
    )
    _cli_runtime.register_runtime_logging_args(cv_csv)
    cv_csv.set_defaults(_handler=_cmd_cv_csv)

    forecast_p = sub.add_parser("forecast", help="Forecast utilities")
    forecast_sub = forecast_p.add_subparsers(dest="forecast_command", required=True)

    forecast_csv = forecast_sub.add_parser("csv", help="Forecast a model on an arbitrary CSV file")
    forecast_csv.add_argument(
        "--model", required=True, help=_MODEL_KEY_HELP
    )
    forecast_csv.add_argument("--path", required=True, help="Path to a CSV file")
    forecast_csv.add_argument(
        "--future-path",
        type=str,
        default="",
        help="Optional path to a CSV file containing future timestamps/covariates only",
    )
    forecast_csv.add_argument("--time-col", required=True, help="Time column name")
    forecast_csv.add_argument("--y-col", required=True, help=_TARGET_COLUMN_HELP)
    forecast_csv.add_argument(
        "--id-cols",
        type=str,
        default="",
        help="Optional comma-separated id columns for panel data",
    )
    forecast_csv.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse time_col with pandas.to_datetime before forecasting",
    )
    forecast_csv.add_argument("--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP)
    forecast_csv.add_argument(
        "--interval-levels",
        type=str,
        default="",
        help="Optional central interval levels as percentages/floats, e.g. 80,90 or 0.8,0.9",
    )
    forecast_csv.add_argument(
        "--interval-min-train-size",
        type=int,
        default=None,
        help="Optional min_train_size used for local bootstrap forecast intervals",
    )
    forecast_csv.add_argument(
        "--interval-samples",
        type=int,
        default=1000,
        help="Bootstrap sample count for local forecast intervals (default: 1000)",
    )
    forecast_csv.add_argument(
        "--interval-seed",
        type=int,
        default=None,
        help="Optional random seed for local bootstrap forecast intervals",
    )
    forecast_csv.add_argument(
        "--model-param",
        action="append",
        default=[],
        help=_MODEL_PARAM_EXAMPLE_HELP,
    )
    forecast_csv.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    forecast_csv.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format for forecasts (default: csv)",
    )
    forecast_csv.add_argument(
        "--save-artifact",
        type=str,
        default="",
        help="Optional path to save a fitted forecasting artifact for reuse",
    )
    _cli_runtime.register_runtime_logging_args(forecast_csv)
    forecast_csv.set_defaults(_handler=_cmd_forecast_csv)

    forecast_artifact = forecast_sub.add_parser(
        "artifact", help="Forecast from a previously saved artifact"
    )
    forecast_artifact.add_argument("--artifact", required=True, help="Path to a saved artifact")
    forecast_artifact.add_argument(
        "--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP
    )
    forecast_artifact.add_argument(
        "--interval-levels",
        type=str,
        default="",
        help=(
            "Optional central interval levels for local artifacts or "
            "interval-capable global artifacts, e.g. 80,90 or 0.8,0.9"
        ),
    )
    forecast_artifact.add_argument(
        "--interval-min-train-size",
        type=int,
        default=None,
        help="Optional min_train_size used for local bootstrap forecast intervals",
    )
    forecast_artifact.add_argument(
        "--interval-samples",
        type=int,
        default=1000,
        help="Bootstrap sample count for local forecast intervals (default: 1000)",
    )
    forecast_artifact.add_argument(
        "--interval-seed",
        type=int,
        default=None,
        help="Optional random seed for local bootstrap forecast intervals",
    )
    forecast_artifact.add_argument(
        "--cutoff",
        type=str,
        default="",
        help="Optional cutoff override for global artifacts",
    )
    forecast_artifact.add_argument(
        "--future-path",
        type=str,
        default="",
        help=(
            "Optional CSV containing future timestamps/covariates for artifact reuse; "
            "local overrides replace saved future context, "
            "global overrides may include unique_id, the saved raw id columns, "
            "or omit ids for single-series artifacts"
        ),
    )
    forecast_artifact.add_argument(
        "--time-col",
        type=str,
        default="",
        help="Time column name used with --future-path (required when overriding future context)",
    )
    forecast_artifact.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse the artifact future-path time column as datetime.",
    )
    forecast_artifact.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    forecast_artifact.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format for forecasts (default: csv)",
    )
    _cli_runtime.register_runtime_logging_args(forecast_artifact)
    forecast_artifact.set_defaults(_handler=_cmd_forecast_artifact)

    artifact_p = sub.add_parser("artifact", help="Artifact utilities")
    artifact_sub = artifact_p.add_subparsers(dest="artifact_command", required=True)

    artifact_info = artifact_sub.add_parser("info", help="Show metadata about a saved artifact")
    artifact_info.add_argument("--artifact", required=True, help="Path to a saved artifact")
    artifact_info.add_argument(
        "--format",
        choices=["json", "md", "markdown"],
        default="json",
        help="Output format (default: json; use md/markdown for a grouped Markdown report)",
    )
    artifact_info.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    _cli_runtime.register_runtime_logging_args(artifact_info)
    artifact_info.set_defaults(_handler=_cmd_artifact_info)

    artifact_validate = artifact_sub.add_parser(
        "validate",
        help="Validate a saved artifact against the supported artifact contract",
    )
    artifact_validate.add_argument("--artifact", required=True, help="Path to a saved artifact")
    artifact_validate.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Output format (default: json)",
    )
    artifact_validate.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    _cli_runtime.register_runtime_logging_args(artifact_validate)
    artifact_validate.set_defaults(_handler=_cmd_artifact_validate)

    artifact_diff = artifact_sub.add_parser(
        "diff",
        help="Compare two saved artifacts by metadata and extra payload",
    )
    artifact_diff.add_argument(
        "--left-artifact",
        required=True,
        help="Path to the left artifact",
    )
    artifact_diff.add_argument(
        "--right-artifact",
        required=True,
        help="Path to the right artifact",
    )
    artifact_diff.add_argument(
        "--path-prefix",
        type=str,
        default="",
        help="Optional dot-path prefix filter, e.g. metadata.train_schema.runtime",
    )
    artifact_diff.add_argument(
        "--format",
        choices=["json", "csv", "md", "markdown"],
        default="json",
        help="Output format (default: json; use md/markdown for a grouped Markdown diff report)",
    )
    artifact_diff.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    _cli_runtime.register_runtime_logging_args(artifact_diff)
    artifact_diff.set_defaults(_handler=_cmd_artifact_diff)

    tuning = sub.add_parser("tuning", help="Hyperparameter tuning utilities")
    tuning_sub = tuning.add_subparsers(dest="tuning_command", required=True)

    tuning_run = tuning_sub.add_parser("run", help="Run deterministic grid search on a dataset")
    tuning_run.add_argument(
        "--model", required=True, help=_MODEL_KEY_HELP
    )
    tuning_run.add_argument("--dataset", required=True, help=_DATASET_KEY_HELP)
    tuning_run.add_argument(
        "--y-col",
        type=str,
        default="",
        help=_DEFAULT_TARGET_COLUMN_HELP,
    )
    tuning_run.add_argument("--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP)
    tuning_run.add_argument("--step", type=int, default=1, help=_WALK_FORWARD_STEP_HELP)
    tuning_run.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum train size before the first evaluation window",
    )
    tuning_run.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit to the last N backtest windows.",
    )
    tuning_run.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help=_EXPANDING_WINDOW_HELP,
    )
    tuning_run.add_argument(
        "--metric",
        choices=["mae", "rmse", "mape", "smape"],
        default="mae",
        help="Optimization metric (default: mae)",
    )
    tuning_run.add_argument(
        "--mode",
        choices=["min", "max"],
        default="min",
        help="Optimization direction (default: min)",
    )
    tuning_run.add_argument(
        "--model-param",
        action="append",
        default=[],
        help="Fixed model parameter as key=value (repeatable).",
    )
    tuning_run.add_argument(
        "--grid-param",
        action="append",
        default=[],
        help="Grid-search parameter as key=v1,v2,... (repeatable).",
    )
    tuning_run.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    tuning_run.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format for the tuning summary (default: json)",
    )
    _cli_runtime.register_runtime_logging_args(tuning_run)
    tuning_run.set_defaults(_handler=_cmd_tuning_run)

    _cli_data.register_data_subparsers(sub)

    detect_p = sub.add_parser("detect", help="Anomaly detection utilities")
    detect_sub = detect_p.add_subparsers(dest="detect_command", required=True)

    detect_run = detect_sub.add_parser("run", help="Detect anomalies on a registered dataset")
    detect_run.add_argument("--dataset", required=True, help=_DATASET_KEY_HELP)
    detect_run.add_argument(
        "--y-col",
        type=str,
        default="",
        help=_DEFAULT_TARGET_COLUMN_HELP,
    )
    detect_run.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional model key used for forecast-residual scoring",
    )
    detect_run.add_argument(
        "--score-method",
        choices=["forecast-residual", "rolling-mad", "rolling-zscore"],
        default="",
        help="Anomaly score method (default: forecast-residual if model is set, else rolling-zscore)",
    )
    detect_run.add_argument(
        "--threshold-method",
        choices=["mad", "quantile", "zscore"],
        default="",
        help="Threshold method (default: mad for forecast-residual, else zscore)",
    )
    detect_run.add_argument(
        "--threshold-k",
        type=float,
        default=3.0,
        help="Threshold multiplier for mad/zscore methods (default: 3.0)",
    )
    detect_run.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.99,
        help="Quantile threshold in (0,1) when --threshold-method=quantile (default: 0.99)",
    )
    detect_run.add_argument(
        "--window",
        type=int,
        default=12,
        help="Rolling history window for rolling score methods (default: 12)",
    )
    detect_run.add_argument(
        "--min-history",
        type=int,
        default=3,
        help="Minimum rolling history before scoring starts (default: 3)",
    )
    detect_run.add_argument(
        "--min-train-size",
        type=int,
        default=None,
        help="Minimum training size for forecast-residual scoring",
    )
    detect_run.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="Walk-forward step size for forecast-residual scoring (default: 1)",
    )
    detect_run.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help=_EXPANDING_WINDOW_HELP,
    )
    detect_run.add_argument(
        "--n-windows",
        type=int,
        default=None,
        help=_MAX_WINDOWS_LIMIT_HELP,
    )
    detect_run.add_argument(
        "--model-param",
        action="append",
        default=[],
        help=_MODEL_PARAM_EXAMPLE_HELP,
    )
    detect_run.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    detect_run.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help=_OUTPUT_ROWS_FORMAT_HELP,
    )
    _cli_runtime.register_runtime_logging_args(detect_run)
    detect_run.set_defaults(_handler=_cmd_detect_run)

    detect_csv = detect_sub.add_parser("csv", help="Detect anomalies on an arbitrary CSV file")
    detect_csv.add_argument("--path", required=True, help="Path to a CSV file")
    detect_csv.add_argument("--time-col", required=True, help="Time column name")
    detect_csv.add_argument("--y-col", required=True, help=_TARGET_COLUMN_HELP)
    detect_csv.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional model key used for forecast-residual scoring",
    )
    detect_csv.add_argument(
        "--id-cols",
        type=str,
        default="",
        help="Optional comma-separated ID columns for panel data (e.g. store,dept)",
    )
    detect_csv.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse the time column as datetime.",
    )
    detect_csv.add_argument(
        "--score-method",
        choices=["forecast-residual", "rolling-mad", "rolling-zscore"],
        default="",
        help="Anomaly score method (default: forecast-residual if model is set, else rolling-zscore)",
    )
    detect_csv.add_argument(
        "--threshold-method",
        choices=["mad", "quantile", "zscore"],
        default="",
        help="Threshold method (default: mad for forecast-residual, else zscore)",
    )
    detect_csv.add_argument(
        "--threshold-k",
        type=float,
        default=3.0,
        help="Threshold multiplier for mad/zscore methods (default: 3.0)",
    )
    detect_csv.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.99,
        help="Quantile threshold in (0,1) when --threshold-method=quantile (default: 0.99)",
    )
    detect_csv.add_argument(
        "--window",
        type=int,
        default=12,
        help="Rolling history window for rolling score methods (default: 12)",
    )
    detect_csv.add_argument(
        "--min-history",
        type=int,
        default=3,
        help="Minimum rolling history before scoring starts (default: 3)",
    )
    detect_csv.add_argument(
        "--min-train-size",
        type=int,
        default=None,
        help="Minimum training size for forecast-residual scoring",
    )
    detect_csv.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="Walk-forward step size for forecast-residual scoring (default: 1)",
    )
    detect_csv.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help=_EXPANDING_WINDOW_HELP,
    )
    detect_csv.add_argument(
        "--n-windows",
        type=int,
        default=None,
        help=_MAX_WINDOWS_LIMIT_HELP,
    )
    detect_csv.add_argument(
        "--model-param",
        action="append",
        default=[],
        help=_MODEL_PARAM_EXAMPLE_HELP,
    )
    detect_csv.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    detect_csv.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help=_OUTPUT_ROWS_FORMAT_HELP,
    )
    _cli_runtime.register_runtime_logging_args(detect_csv)
    detect_csv.set_defaults(_handler=_cmd_detect_csv)

    eval_p = sub.add_parser("eval", help="Evaluation utilities")
    eval_sub = eval_p.add_subparsers(dest="eval_command", required=True)

    eval_naive_last = eval_sub.add_parser("naive-last", help="Evaluate naive-last baseline")
    eval_naive_last.add_argument("--dataset", required=True, help=_DATASET_KEY_HELP)
    eval_naive_last.add_argument("--y-col", required=True, help=_TARGET_COLUMN_HELP)
    eval_naive_last.add_argument(
        "--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP
    )
    eval_naive_last.add_argument("--step", type=int, default=1, help=_WALK_FORWARD_STEP_HELP)
    eval_naive_last.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help=_MIN_TRAIN_SIZE_FIRST_WINDOW_HELP,
    )
    eval_naive_last.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=_MAX_WINDOWS_LIMIT_HELP,
    )
    eval_naive_last.add_argument(
        "--output",
        type=str,
        default="",
        help=_METRICS_OUTPUT_PATH_HELP,
    )
    eval_naive_last.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help=_OUTPUT_JSON_FORMAT_HELP,
    )
    _cli_runtime.register_runtime_logging_args(eval_naive_last)
    eval_naive_last.set_defaults(_handler=_cmd_eval_naive_last)

    eval_seasonal_naive = eval_sub.add_parser(
        "seasonal-naive", help="Evaluate seasonal naive baseline"
    )
    eval_seasonal_naive.add_argument("--dataset", required=True, help=_DATASET_KEY_HELP)
    eval_seasonal_naive.add_argument("--y-col", required=True, help=_TARGET_COLUMN_HELP)
    eval_seasonal_naive.add_argument(
        "--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP
    )
    eval_seasonal_naive.add_argument(
        "--step", type=int, default=1, help=_WALK_FORWARD_STEP_HELP
    )
    eval_seasonal_naive.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help=_MIN_TRAIN_SIZE_FIRST_WINDOW_HELP,
    )
    eval_seasonal_naive.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=_MAX_WINDOWS_LIMIT_HELP,
    )
    eval_seasonal_naive.add_argument(
        "--season-length",
        type=int,
        required=True,
        help="Season length for seasonal naive",
    )
    eval_seasonal_naive.add_argument(
        "--output",
        type=str,
        default="",
        help=_METRICS_OUTPUT_PATH_HELP,
    )
    eval_seasonal_naive.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help=_OUTPUT_JSON_FORMAT_HELP,
    )
    _cli_runtime.register_runtime_logging_args(eval_seasonal_naive)
    eval_seasonal_naive.set_defaults(_handler=_cmd_eval_seasonal_naive)

    eval_run = eval_sub.add_parser("run", help="Evaluate any registered model")
    eval_run.add_argument("--model", required=True, help=_MODEL_KEY_HELP)
    eval_run.add_argument("--dataset", required=True, help=_DATASET_KEY_HELP)
    eval_run.add_argument(
        "--y-col",
        type=str,
        default="",
        help=_DEFAULT_TARGET_COLUMN_HELP,
    )
    eval_run.add_argument("--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP)
    eval_run.add_argument("--step", type=int, default=1, help=_WALK_FORWARD_STEP_HELP)
    eval_run.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help=_MIN_TRAIN_SIZE_FIRST_WINDOW_HELP,
    )
    eval_run.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=_MAX_WINDOWS_LIMIT_HELP,
    )
    eval_run.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help=_EXPANDING_WINDOW_HELP,
    )
    eval_run.add_argument(
        "--conformal-levels",
        type=str,
        default="",
        help="Optional conformal levels as percentages or floats, e.g. 80,90 or 0.8,0.9",
    )
    eval_run.add_argument(
        "--conformal-pooled",
        action="store_true",
        help="Pool residuals across steps instead of per-step conformal radii.",
    )
    eval_run.add_argument(
        "--model-param",
        action="append",
        default=[],
        help=_MODEL_PARAM_EXAMPLE_HELP,
    )
    eval_run.add_argument(
        "--output",
        type=str,
        default="",
        help=_METRICS_OUTPUT_PATH_HELP,
    )
    eval_run.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help=_OUTPUT_JSON_FORMAT_HELP,
    )
    _cli_runtime.register_runtime_logging_args(eval_run)
    eval_run.set_defaults(_handler=_cmd_eval_run)

    eval_csv = eval_sub.add_parser("csv", help="Evaluate a model on an arbitrary CSV file")
    eval_csv.add_argument("--model", required=True, help=_MODEL_KEY_HELP)
    eval_csv.add_argument("--path", required=True, help="Path to a CSV file")
    eval_csv.add_argument("--time-col", required=True, help="Time column name")
    eval_csv.add_argument("--y-col", required=True, help=_TARGET_COLUMN_HELP)
    eval_csv.add_argument(
        "--id-cols",
        type=str,
        default="",
        help="Optional comma-separated ID columns for panel data (e.g. store,dept)",
    )
    eval_csv.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse the time column as datetime.",
    )
    eval_csv.add_argument("--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP)
    eval_csv.add_argument("--step", type=int, default=1, help=_WALK_FORWARD_STEP_HELP)
    eval_csv.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help=_MIN_TRAIN_SIZE_FIRST_WINDOW_HELP,
    )
    eval_csv.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=_MAX_WINDOWS_LIMIT_HELP,
    )
    eval_csv.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help=_EXPANDING_WINDOW_HELP,
    )
    eval_csv.add_argument(
        "--conformal-levels",
        type=str,
        default="",
        help="Optional conformal levels as percentages or floats, e.g. 80,90 or 0.8,0.9",
    )
    eval_csv.add_argument(
        "--conformal-pooled",
        action="store_true",
        help="Pool residuals across steps instead of per-step conformal radii.",
    )
    eval_csv.add_argument(
        "--model-param",
        action="append",
        default=[],
        help=_MODEL_PARAM_EXAMPLE_HELP,
    )
    eval_csv.add_argument(
        "--output",
        type=str,
        default="",
        help=_METRICS_OUTPUT_PATH_HELP,
    )
    eval_csv.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help=_OUTPUT_JSON_FORMAT_HELP,
    )
    _cli_runtime.register_runtime_logging_args(eval_csv)
    eval_csv.set_defaults(_handler=_cmd_eval_csv)

    _cli_leaderboard.register_leaderboard_subparsers(sub)

    return p


def _silence_stdout_broken_pipe() -> None:
    """
    Best-effort: redirect stdout to /dev/null so Python's final flush/close
    does not emit "Exception ignored ... BrokenPipeError" noise when stdout was
    closed early by a downstream consumer (e.g. `| head -n 1`).
    """

    try:
        stdout_fd = sys.stdout.fileno()
    except Exception:  # noqa: BLE001
        return

    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except Exception:  # noqa: BLE001
        return

    try:
        os.dup2(devnull_fd, stdout_fd)
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            os.close(devnull_fd)
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        # Avoid importing package metadata; keep it simple and dependency-free.
        from . import __version__

        try:
            print(f"ForeSight {__version__}")
            sys.stdout.flush()
        except BrokenPipeError:
            _silence_stdout_broken_pipe()
        return 0

    handler = getattr(args, "_handler", None)
    if handler is None:
        handler = _root_shortcut_handler(args)
    if handler is None:
        try:
            parser.print_help()
            sys.stdout.flush()
        except BrokenPipeError:
            _silence_stdout_broken_pipe()
        return 0

    try:
        code = int(handler(args))
        try:
            sys.stdout.flush()
        except BrokenPipeError:
            _silence_stdout_broken_pipe()
            return 0
        return code
    except BrokenPipeError:
        _silence_stdout_broken_pipe()
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001
        if getattr(args, "debug", False):
            raise
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def _root_shortcut_handler(args: argparse.Namespace) -> Any:
    wants_models = bool(getattr(args, "list", False) or getattr(args, "list_models", False))
    wants_datasets = bool(getattr(args, "list_datasets", False))

    if wants_models and wants_datasets:
        raise ValueError("Choose only one of: --list/--list-models or --list-datasets")

    if wants_models:
        shortcut_args = argparse.Namespace(
            format="tsv",
            output="",
            prefix="",
            interface="any",
            requires="",
            exclude_requires="",
            columns="",
            header=False,
            sort="key",
            desc=False,
            limit=0,
        )
        return lambda _args: _cli_catalog._cmd_models_list(shortcut_args)

    if wants_datasets:
        shortcut_args = argparse.Namespace(
            with_path=False,
            data_dir=str(getattr(args, "data_dir", "")),
        )
        return lambda _args: _cli_data._cmd_datasets_list(shortcut_args)

    return None


def _cmd_doctor(args: argparse.Namespace) -> int:
    from . import __version__
    from .datasets.registry import list_datasets, list_packaged_datasets, preview_dataset_path_info
    from .optional_deps import get_dependency_status, get_extra_status

    dependency_keys = ["ml", "xgb", "lgbm", "catboost", "stats", "torch", "transformers"]
    extra_keys = ["core", "ml", "xgb", "lgbm", "catboost", "stats", "torch", "transformers", "all"]
    data_dir = str(getattr(args, "data_dir", ""))

    datasets: dict[str, dict[str, Any]] = {}
    packaged = set(list_packaged_datasets())
    for key in list_datasets():
        path, source, available = preview_dataset_path_info(key, data_dir=data_dir)
        datasets[key] = {
            "available": available,
            "packaged": key in packaged,
            "path": str(path),
            "source": source,
        }

    package_path = Path(__file__).resolve().parent / "__init__.py"
    payload = {
        "package": {
            "name": "foresight-ts",
            "version": __version__,
            "module_path": str(package_path),
            "editable_like": "site-packages" not in str(package_path),
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "data_dir": {
            "argument": data_dir.strip() or None,
            "env": os.environ.get("FORESIGHT_DATA_DIR", "").strip() or None,
        },
        "dependencies": {name: get_dependency_status(name).as_dict() for name in dependency_keys},
        "extras": {name: get_extra_status(name).as_dict() for name in extra_keys},
        "datasets": datasets,
    }
    _cli_shared._emit(payload, output=str(args.output), fmt=str(args.format))
    return 0


def _log_payload(**kwargs: Any) -> dict[str, Any]:
    return _cli_runtime.compact_log_payload(**kwargs)


def _run_logged_command(
    args: argparse.Namespace,
    *,
    command: str,
    payload: dict[str, Any],
    action: Any,
) -> int:
    with _cli_runtime.command_scope(args, command=command, payload=payload):
        return int(action())


def _cmd_cv_run(args: argparse.Namespace) -> int:
    from .cv import cross_validation_predictions

    def _run() -> int:
        model_params = _cli_shared._parse_model_params(list(args.model_param))
        y_col = str(args.y_col).strip() or None
        df = cross_validation_predictions(
            model=str(args.model),
            dataset=str(args.dataset),
            y_col=y_col,
            horizon=int(args.horizon),
            step_size=int(args.step_size),
            min_train_size=int(args.min_train_size),
            max_train_size=args.max_train_size,
            n_windows=args.n_windows,
            model_params=model_params,
            data_dir=str(args.data_dir),
        )
        _cli_shared._emit_dataframe(df, output=str(args.output), fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="cv run",
        payload=_log_payload(
            model=str(args.model),
            dataset=str(args.dataset),
            horizon=int(args.horizon),
            step_size=int(args.step_size),
            min_train_size=int(args.min_train_size),
            n_windows=args.n_windows,
        ),
        action=_run,
    )


def _cmd_cv_csv(args: argparse.Namespace) -> int:
    from .io import parse_id_cols
    from .services.cli_workflows import cv_csv_workflow

    def _run() -> int:
        model_params = _cli_shared._parse_model_params(list(args.model_param))
        id_cols = parse_id_cols(str(args.id_cols))
        df = cv_csv_workflow(
            model=str(args.model),
            path=str(args.path),
            time_col=str(args.time_col),
            y_col=str(args.y_col),
            horizon=int(args.horizon),
            step_size=int(args.step_size),
            min_train_size=int(args.min_train_size),
            id_cols=id_cols,
            parse_dates=bool(args.parse_dates),
            model_params=model_params,
            max_train_size=args.max_train_size,
            n_windows=args.n_windows,
        )
        _cli_shared._emit_dataframe(df, output=str(args.output), fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="cv csv",
        payload=_log_payload(
            model=str(args.model),
            path=str(args.path),
            horizon=int(args.horizon),
            step_size=int(args.step_size),
            min_train_size=int(args.min_train_size),
            n_windows=args.n_windows,
        ),
        action=_run,
    )


def _cmd_forecast_csv(args: argparse.Namespace) -> int:
    from .io import parse_id_cols
    from .services.cli_workflows import forecast_csv_workflow

    def _run() -> int:
        model_params = _cli_shared._parse_model_params(list(args.model_param))
        id_cols = parse_id_cols(str(args.id_cols))
        pred = forecast_csv_workflow(
            model=str(args.model),
            path=str(args.path),
            time_col=str(args.time_col),
            y_col=str(args.y_col),
            id_cols=id_cols,
            parse_dates=bool(args.parse_dates),
            horizon=int(args.horizon),
            model_params=model_params,
            future_path=str(getattr(args, "future_path", "")).strip() or None,
            interval_levels=str(getattr(args, "interval_levels", "")).strip(),
            interval_min_train_size=getattr(args, "interval_min_train_size", None),
            interval_samples=int(getattr(args, "interval_samples", 1000)),
            interval_seed=getattr(args, "interval_seed", None),
            save_artifact_path=str(getattr(args, "save_artifact", "")).strip() or None,
        )
        _cli_shared._emit_dataframe(pred, output=str(args.output), fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="forecast csv",
        payload=_log_payload(
            model=str(args.model),
            path=str(args.path),
            horizon=int(args.horizon),
            future_path=str(getattr(args, "future_path", "")).strip() or None,
            save_artifact=str(getattr(args, "save_artifact", "")).strip() or None,
        ),
        action=_run,
    )


def _cmd_forecast_artifact(args: argparse.Namespace) -> int:
    from .services.cli_workflows import forecast_artifact_workflow

    def _run() -> int:
        pred = forecast_artifact_workflow(
            artifact=str(args.artifact),
            horizon=int(args.horizon),
            interval_levels=str(getattr(args, "interval_levels", "")).strip(),
            interval_min_train_size=getattr(args, "interval_min_train_size", None),
            interval_samples=int(getattr(args, "interval_samples", 1000)),
            interval_seed=getattr(args, "interval_seed", None),
            cutoff=getattr(args, "cutoff", None),
            future_path=str(getattr(args, "future_path", "")).strip() or None,
            time_col=str(getattr(args, "time_col", "")).strip() or None,
            parse_dates=bool(getattr(args, "parse_dates", False)),
        )
        _cli_shared._emit_dataframe(pred, output=str(args.output), fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="forecast artifact",
        payload=_log_payload(
            artifact=str(args.artifact),
            horizon=int(args.horizon),
            cutoff=getattr(args, "cutoff", None),
            future_path=str(getattr(args, "future_path", "")).strip() or None,
        ),
        action=_run,
    )


def _cmd_artifact_info(args: argparse.Namespace) -> int:
    from .services.cli_workflows import artifact_info_markdown_workflow, artifact_info_workflow

    def _run() -> int:
        fmt = str(args.format)
        if fmt in {"md", "markdown"}:
            text = artifact_info_markdown_workflow(artifact=str(args.artifact))
            _cli_shared._emit_text(text, output=str(args.output))
            return 0
        payload = artifact_info_workflow(artifact=str(args.artifact))
        _cli_shared._emit(payload, output=str(args.output), fmt=fmt)
        return 0

    return _run_logged_command(
        args,
        command="artifact info",
        payload=_log_payload(artifact=str(args.artifact)),
        action=_run,
    )


def _cmd_artifact_validate(args: argparse.Namespace) -> int:
    from .services.cli_workflows import artifact_validate_workflow

    def _run() -> int:
        payload = artifact_validate_workflow(artifact=str(args.artifact))
        _cli_shared._emit(payload, output=str(args.output), fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="artifact validate",
        payload=_log_payload(artifact=str(args.artifact)),
        action=_run,
    )


def _cmd_artifact_diff(args: argparse.Namespace) -> int:
    from .services.cli_workflows import (
        artifact_diff_markdown_workflow,
        artifact_diff_rows_workflow,
        artifact_diff_workflow,
    )

    def _run() -> int:
        left_artifact = str(args.left_artifact)
        right_artifact = str(args.right_artifact)
        fmt = "md" if str(args.format) == "markdown" else str(args.format)
        if fmt == "json":
            payload = artifact_diff_workflow(
                left_artifact=left_artifact,
                right_artifact=right_artifact,
                path_prefix=str(getattr(args, "path_prefix", "")).strip() or None,
            )
            _cli_shared._emit(payload, output=str(args.output), fmt=fmt)
            return 0
        if fmt == "md":
            text = artifact_diff_markdown_workflow(
                left_artifact=left_artifact,
                right_artifact=right_artifact,
                path_prefix=str(getattr(args, "path_prefix", "")).strip() or None,
            )
            _cli_shared._emit_text(text, output=str(args.output))
            return 0

        rows = artifact_diff_rows_workflow(
            left_artifact=left_artifact,
            right_artifact=right_artifact,
            path_prefix=str(getattr(args, "path_prefix", "")).strip() or None,
        )
        _cli_shared._emit_table(
            rows,
            columns=["path", "left", "right"],
            output=str(args.output),
            fmt=fmt,
        )
        return 0

    return _run_logged_command(
        args,
        command="artifact diff",
        payload=_log_payload(
            left_artifact=str(args.left_artifact),
            right_artifact=str(args.right_artifact),
            path_prefix=str(getattr(args, "path_prefix", "")).strip() or None,
        ),
        action=_run,
    )


def _cmd_tuning_run(args: argparse.Namespace) -> int:
    from .tuning import tune_model

    def _run() -> int:
        model_params = _cli_shared._parse_model_params(list(args.model_param))
        grid_params = _cli_shared._parse_grid_params(list(args.grid_param))
        y_col = str(args.y_col).strip() or None

        payload = tune_model(
            model=str(args.model),
            dataset=str(args.dataset),
            y_col=y_col,
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            search_space=grid_params,
            metric=str(args.metric),
            mode=str(args.mode),
            model_params=model_params,
            data_dir=str(args.data_dir),
            max_windows=args.max_windows,
            max_train_size=args.max_train_size,
        )

        fmt = str(args.format)
        if fmt == "json":
            _cli_shared._emit(payload, output=str(args.output), fmt=fmt)
        else:
            row = {
                "model": payload["model"],
                "dataset": payload["dataset"],
                "metric": payload["metric"],
                "mode": payload["mode"],
                "horizon": payload["horizon"],
                "step": payload["step"],
                "min_train_size": payload["min_train_size"],
                "max_windows": payload["max_windows"],
                "max_train_size": payload["max_train_size"],
                "n_trials": payload["n_trials"],
                "best_score": payload["best_score"],
                "best_params": json.dumps(payload["best_params"], ensure_ascii=False, sort_keys=True),
            }
            _cli_shared._emit_table(
                [row],
                columns=[
                    "model",
                    "dataset",
                    "metric",
                    "mode",
                    "horizon",
                    "step",
                    "min_train_size",
                    "max_windows",
                    "max_train_size",
                    "n_trials",
                    "best_score",
                    "best_params",
                ],
                output=str(args.output),
                fmt=fmt,
            )
        return 0

    return _run_logged_command(
        args,
        command="tuning run",
        payload=_log_payload(
            model=str(args.model),
            dataset=str(args.dataset),
            metric=str(args.metric),
            mode=str(args.mode),
            horizon=int(args.horizon),
            step=int(args.step),
        ),
        action=_run,
    )


def _cmd_detect_run(args: argparse.Namespace) -> int:
    from .detect import detect_anomalies

    def _run() -> int:
        model_params = _cli_shared._parse_model_params(list(args.model_param))
        y_col = str(args.y_col).strip() or None
        pred = detect_anomalies(
            dataset=str(args.dataset),
            y_col=y_col,
            model=str(args.model).strip() or None,
            model_params=model_params,
            data_dir=str(args.data_dir),
            score_method=str(args.score_method).strip() or None,
            threshold_method=str(args.threshold_method).strip() or None,
            threshold_k=float(args.threshold_k),
            threshold_quantile=float(args.threshold_quantile),
            window=int(args.window),
            min_history=int(args.min_history),
            min_train_size=args.min_train_size,
            step_size=int(args.step_size),
            max_train_size=args.max_train_size,
            n_windows=args.n_windows,
        )
        _cli_shared._emit_dataframe(pred, output=str(args.output), fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="detect run",
        payload=_log_payload(
            dataset=str(args.dataset),
            model=str(args.model).strip() or None,
            score_method=str(args.score_method).strip() or None,
            threshold_method=str(args.threshold_method).strip() or None,
        ),
        action=_run,
    )


def _cmd_detect_csv(args: argparse.Namespace) -> int:
    from .io import parse_id_cols
    from .services.cli_workflows import detect_csv_workflow

    def _run() -> int:
        model_params = _cli_shared._parse_model_params(list(args.model_param))
        id_cols = parse_id_cols(str(args.id_cols))
        pred = detect_csv_workflow(
            path=str(args.path),
            time_col=str(args.time_col),
            y_col=str(args.y_col),
            model=str(args.model).strip() or None,
            id_cols=id_cols,
            parse_dates=bool(args.parse_dates),
            model_params=model_params,
            score_method=str(args.score_method).strip() or None,
            threshold_method=str(args.threshold_method).strip() or None,
            threshold_k=float(args.threshold_k),
            threshold_quantile=float(args.threshold_quantile),
            window=int(args.window),
            min_history=int(args.min_history),
            min_train_size=args.min_train_size,
            step_size=int(args.step_size),
            max_train_size=args.max_train_size,
            n_windows=args.n_windows,
        )
        _cli_shared._emit_dataframe(pred, output=str(args.output), fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="detect csv",
        payload=_log_payload(
            path=str(args.path),
            model=str(args.model).strip() or None,
            score_method=str(args.score_method).strip() or None,
            threshold_method=str(args.threshold_method).strip() or None,
        ),
        action=_run,
    )


def _cmd_eval_naive_last(args: argparse.Namespace) -> int:
    from .eval import eval_naive_last

    def _run() -> int:
        payload = eval_naive_last(
            dataset=str(args.dataset),
            y_col=str(args.y_col),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            max_windows=args.max_windows,
            data_dir=str(args.data_dir),
        )
        _cli_shared._emit(payload, output=args.output, fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="eval naive-last",
        payload=_log_payload(
            dataset=str(args.dataset),
            y_col=str(args.y_col),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
        ),
        action=_run,
    )


def _cmd_eval_seasonal_naive(args: argparse.Namespace) -> int:
    from .eval import eval_seasonal_naive

    def _run() -> int:
        payload = eval_seasonal_naive(
            dataset=str(args.dataset),
            y_col=str(args.y_col),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            season_length=int(args.season_length),
            max_windows=args.max_windows,
            data_dir=str(args.data_dir),
        )
        _cli_shared._emit(payload, output=args.output, fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="eval seasonal-naive",
        payload=_log_payload(
            dataset=str(args.dataset),
            y_col=str(args.y_col),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            season_length=int(args.season_length),
        ),
        action=_run,
    )


def _cmd_eval_run(args: argparse.Namespace) -> int:
    from .eval_forecast import eval_model

    def _run() -> int:
        model_params = _cli_shared._parse_model_params(list(args.model_param))
        y_col = str(args.y_col).strip() or None
        payload = eval_model(
            model=str(args.model),
            dataset=str(args.dataset),
            y_col=y_col,
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            max_windows=args.max_windows,
            max_train_size=args.max_train_size,
            conformal_levels=str(args.conformal_levels).strip() or None,
            conformal_per_step=(not bool(args.conformal_pooled)),
            model_params=model_params,
            data_dir=str(args.data_dir),
        )
        _cli_shared._emit(payload, output=args.output, fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="eval run",
        payload=_log_payload(
            model=str(args.model),
            dataset=str(args.dataset),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            max_windows=args.max_windows,
        ),
        action=_run,
    )


def _cmd_eval_csv(args: argparse.Namespace) -> int:
    from .io import parse_id_cols
    from .services.cli_workflows import eval_csv_workflow

    def _run() -> int:
        model_params = _cli_shared._parse_model_params(list(args.model_param))
        id_cols = parse_id_cols(str(args.id_cols))
        payload = eval_csv_workflow(
            model=str(args.model),
            path=str(args.path),
            time_col=str(args.time_col),
            y_col=str(args.y_col),
            id_cols=id_cols,
            parse_dates=bool(args.parse_dates),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            model_params=model_params,
            max_windows=args.max_windows,
            max_train_size=args.max_train_size,
            conformal_levels=str(args.conformal_levels).strip() or None,
            conformal_per_step=(not bool(args.conformal_pooled)),
        )
        _cli_shared._emit(payload, output=str(args.output), fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="eval csv",
        payload=_log_payload(
            model=str(args.model),
            path=str(args.path),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            max_windows=args.max_windows,
        ),
        action=_run,
    )
