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
from . import cli_shared as _cli_shared


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

    cv = sub.add_parser("cv", help="Cross-validation utilities")
    cv_sub = cv.add_subparsers(dest="cv_command", required=True)

    cv_run = cv_sub.add_parser("run", help="Run rolling-origin CV and output predictions")
    cv_run.add_argument("--model", required=True, help="Model key (see: `foresight models list`)")
    cv_run.add_argument("--dataset", required=True, help="Dataset key")
    cv_run.add_argument(
        "--y-col",
        type=str,
        default="",
        help="Optional target column name (default: use dataset spec default_y).",
    )
    cv_run.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    cv_run.add_argument("--step-size", type=int, default=1, help="CV step size (default: 1)")
    cv_run.add_argument("--min-train-size", type=int, required=True, help="Minimum train size")
    cv_run.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help="Optional rolling train window size (default: expanding window).",
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
        help="Model parameter as key=value (repeatable). Example: --model-param season_length=12",
    )
    cv_run.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
    )
    cv_run.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format for predictions (default: csv)",
    )
    cv_run.set_defaults(_handler=_cmd_cv_run)

    forecast_p = sub.add_parser("forecast", help="Forecast utilities")
    forecast_sub = forecast_p.add_subparsers(dest="forecast_command", required=True)

    forecast_csv = forecast_sub.add_parser("csv", help="Forecast a model on an arbitrary CSV file")
    forecast_csv.add_argument(
        "--model", required=True, help="Model key (see: `foresight models list`)"
    )
    forecast_csv.add_argument("--path", required=True, help="Path to a CSV file")
    forecast_csv.add_argument(
        "--future-path",
        type=str,
        default="",
        help="Optional path to a CSV file containing future timestamps/covariates only",
    )
    forecast_csv.add_argument("--time-col", required=True, help="Time column name")
    forecast_csv.add_argument("--y-col", required=True, help="Target column name")
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
    forecast_csv.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
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
        help="Model parameter as key=value (repeatable). Example: --model-param season_length=12",
    )
    forecast_csv.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
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
    forecast_csv.set_defaults(_handler=_cmd_forecast_csv)

    forecast_artifact = forecast_sub.add_parser(
        "artifact", help="Forecast from a previously saved artifact"
    )
    forecast_artifact.add_argument("--artifact", required=True, help="Path to a saved artifact")
    forecast_artifact.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    forecast_artifact.add_argument(
        "--interval-levels",
        type=str,
        default="",
        help="Optional central interval levels for local artifacts, e.g. 80,90 or 0.8,0.9",
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
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
    )
    forecast_artifact.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format for forecasts (default: csv)",
    )
    forecast_artifact.set_defaults(_handler=_cmd_forecast_artifact)

    tuning = sub.add_parser("tuning", help="Hyperparameter tuning utilities")
    tuning_sub = tuning.add_subparsers(dest="tuning_command", required=True)

    tuning_run = tuning_sub.add_parser("run", help="Run deterministic grid search on a dataset")
    tuning_run.add_argument(
        "--model", required=True, help="Model key (see: `foresight models list`)"
    )
    tuning_run.add_argument("--dataset", required=True, help="Dataset key")
    tuning_run.add_argument(
        "--y-col",
        type=str,
        default="",
        help="Optional target column name (default: use dataset spec default_y).",
    )
    tuning_run.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    tuning_run.add_argument("--step", type=int, default=1, help="Walk-forward step size")
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
        help="Optional rolling train window size (default: expanding window).",
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
        help="Optional path to write output",
    )
    tuning_run.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format for the tuning summary (default: json)",
    )
    tuning_run.set_defaults(_handler=_cmd_tuning_run)

    _cli_data.register_data_subparsers(sub)

    eval_p = sub.add_parser("eval", help="Evaluation utilities")
    eval_sub = eval_p.add_subparsers(dest="eval_command", required=True)

    eval_naive_last = eval_sub.add_parser("naive-last", help="Evaluate naive-last baseline")
    eval_naive_last.add_argument("--dataset", required=True, help="Dataset key")
    eval_naive_last.add_argument("--y-col", required=True, help="Target column name")
    eval_naive_last.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    eval_naive_last.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    eval_naive_last.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    eval_naive_last.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
    )
    eval_naive_last.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write metrics output",
    )
    eval_naive_last.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    eval_naive_last.set_defaults(_handler=_cmd_eval_naive_last)

    eval_seasonal_naive = eval_sub.add_parser(
        "seasonal-naive", help="Evaluate seasonal naive baseline"
    )
    eval_seasonal_naive.add_argument("--dataset", required=True, help="Dataset key")
    eval_seasonal_naive.add_argument("--y-col", required=True, help="Target column name")
    eval_seasonal_naive.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    eval_seasonal_naive.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    eval_seasonal_naive.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    eval_seasonal_naive.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
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
        help="Optional path to write metrics output",
    )
    eval_seasonal_naive.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    eval_seasonal_naive.set_defaults(_handler=_cmd_eval_seasonal_naive)

    eval_run = eval_sub.add_parser("run", help="Evaluate any registered model")
    eval_run.add_argument("--model", required=True, help="Model key (see: `foresight models list`)")
    eval_run.add_argument("--dataset", required=True, help="Dataset key")
    eval_run.add_argument(
        "--y-col",
        type=str,
        default="",
        help="Optional target column name (default: use dataset spec default_y).",
    )
    eval_run.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    eval_run.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    eval_run.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    eval_run.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
    )
    eval_run.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help="Optional rolling train window size (default: expanding window).",
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
        help="Model parameter as key=value (repeatable). Example: --model-param season_length=12",
    )
    eval_run.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write metrics output",
    )
    eval_run.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    eval_run.set_defaults(_handler=_cmd_eval_run)

    eval_csv = eval_sub.add_parser("csv", help="Evaluate a model on an arbitrary CSV file")
    eval_csv.add_argument("--model", required=True, help="Model key (see: `foresight models list`)")
    eval_csv.add_argument("--path", required=True, help="Path to a CSV file")
    eval_csv.add_argument("--time-col", required=True, help="Time column name")
    eval_csv.add_argument("--y-col", required=True, help="Target column name")
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
    eval_csv.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    eval_csv.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    eval_csv.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    eval_csv.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
    )
    eval_csv.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help="Optional rolling train window size (default: expanding window).",
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
        help="Model parameter as key=value (repeatable). Example: --model-param season_length=12",
    )
    eval_csv.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write metrics output",
    )
    eval_csv.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
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


def _cmd_cv_run(args: argparse.Namespace) -> int:
    from .cv import cross_validation_predictions

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


def _cmd_forecast_csv(args: argparse.Namespace) -> int:
    from .io import parse_id_cols
    from .services.cli_workflows import forecast_csv_workflow

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


def _cmd_forecast_artifact(args: argparse.Namespace) -> int:
    from .services.cli_workflows import forecast_artifact_workflow

    pred = forecast_artifact_workflow(
        artifact=str(args.artifact),
        horizon=int(args.horizon),
        interval_levels=str(getattr(args, "interval_levels", "")).strip(),
        interval_min_train_size=getattr(args, "interval_min_train_size", None),
        interval_samples=int(getattr(args, "interval_samples", 1000)),
        interval_seed=getattr(args, "interval_seed", None),
        cutoff=getattr(args, "cutoff", None),
    )
    _cli_shared._emit_dataframe(pred, output=str(args.output), fmt=str(args.format))
    return 0


def _cmd_tuning_run(args: argparse.Namespace) -> int:
    from .tuning import tune_model

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
        return 0

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


def _cmd_eval_naive_last(args: argparse.Namespace) -> int:
    from .eval import eval_naive_last

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


def _cmd_eval_seasonal_naive(args: argparse.Namespace) -> int:
    from .eval import eval_seasonal_naive

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


def _cmd_eval_run(args: argparse.Namespace) -> int:
    from .eval_forecast import eval_model

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


def _cmd_eval_csv(args: argparse.Namespace) -> int:
    from .io import parse_id_cols
    from .services.cli_workflows import eval_csv_workflow

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

