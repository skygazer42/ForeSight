from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from . import cli_runtime as _cli_runtime
from . import cli_shared as _cli_shared
from .dataset_long_df_cache import get_or_build_dataset_long_df

_FORECAST_HORIZON_HELP = "Forecast horizon"
_WALK_FORWARD_STEP_HELP = "Walk-forward step size"
_MIN_TRAIN_SIZE_FIRST_WINDOW_HELP = "Minimum training size for first window"
_MAX_WINDOWS_LIMIT_HELP = "Optional limit on the number of walk-forward windows"
_OUTPUT_JSON_FORMAT_HELP = "Output format (default: json)"
_TASK_GROUP_CHOICES = ["all", "point", "probabilistic", "covariate"]
_batch_execution: Any | None = None


def _get_batch_execution_module() -> Any:
    from .module_cache import get_cached_module

    return get_cached_module(globals(), "_batch_execution", ".batch_execution", __package__)


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


def _leaderboard_status_sort_key(status: object) -> int:
    return 0 if str(status).strip().lower() == "ok" else 1


def _normalize_task_group_filter(raw: object) -> str | None:
    value = str(raw).strip().lower()
    if not value or value == "all":
        return None
    if value not in set(_TASK_GROUP_CHOICES):
        raise ValueError(f"--task-group must be one of: {', '.join(_TASK_GROUP_CHOICES)}")
    return value


def _leaderboard_backend_family_for_model_spec(spec: Any) -> str:
    requires = tuple(str(item).strip().lower() for item in getattr(spec, "requires", ()) if str(item).strip())
    if not requires:
        return "core"
    return str(requires[0])


def _leaderboard_requested_covariate_roles(
    model_params: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    from .contracts.params import (
        normalize_covariate_roles as _normalize_covariate_roles,
    )
    from .contracts.params import (
        normalize_static_cols as _normalize_static_cols,
    )

    historic_x_cols, future_x_cols = _normalize_covariate_roles(model_params)
    static_cols = _normalize_static_cols(model_params)
    return historic_x_cols, future_x_cols, static_cols


def _leaderboard_has_requested_covariates(model_params: dict[str, Any]) -> bool:
    historic_x_cols, future_x_cols, static_cols = _leaderboard_requested_covariate_roles(
        model_params
    )
    return bool(historic_x_cols or future_x_cols or static_cols)


def _leaderboard_has_requested_quantiles(model_params: dict[str, Any]) -> bool:
    raw = model_params.get("quantiles")
    if raw is None:
        return False
    if isinstance(raw, str):
        return bool([part.strip() for part in raw.split(",") if part.strip()])
    if isinstance(raw, list | tuple):
        return bool(raw)
    return True


def _leaderboard_task_group_for_model_spec(spec: Any, *, model_params: dict[str, Any]) -> str:
    capabilities = dict(getattr(spec, "capabilities", {}))
    if bool(capabilities.get("requires_future_covariates", False)) or _leaderboard_has_requested_covariates(
        model_params
    ):
        return "covariate"
    if _leaderboard_has_requested_quantiles(model_params):
        return "probabilistic"
    return "point"


def _leaderboard_model_metadata(
    model_key: str,
    *,
    model_params: dict[str, Any],
) -> tuple[str, str]:
    from .models.registry import get_model_spec

    try:
        spec = get_model_spec(str(model_key))
    except Exception:  # noqa: BLE001
        return ("point", "unknown")
    return (
        _leaderboard_task_group_for_model_spec(spec, model_params=model_params),
        _leaderboard_backend_family_for_model_spec(spec),
    )


def _get_or_build_leaderboard_long_df(
    *,
    dataset: str,
    y_col: str | None,
    data_dir: str,
    model_params: dict[str, Any],
) -> tuple[Any, str]:
    frame_bundle = get_or_build_dataset_long_df(
        dataset=str(dataset),
        y_col=y_col,
        data_dir=str(data_dir),
        model_params=model_params,
    )
    return frame_bundle["long_df"], str(frame_bundle["y_col_final"])


def _leaderboard_skip_row(
    *,
    model_key: str,
    dataset_key: str,
    y_col_final: str,
    horizon: int,
    step: int,
    min_train_size: int,
    max_windows: int | None,
    data_dir_s: str,
    model_params: dict[str, Any],
    task_group: str,
    backend_family: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "model": str(model_key),
        "task_group": str(task_group),
        "backend_family": str(backend_family),
        "status": "skip",
        "skip_reason": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "dataset": str(dataset_key),
        "y_col": str(y_col_final),
        "horizon": int(horizon),
        "step": int(step),
        "min_train_size": int(min_train_size),
        "max_windows": None if max_windows is None else int(max_windows),
        "n_series": 0,
        "n_series_skipped": 0,
        "n_windows": 0,
        "n_points": 0,
        "mae": None,
        "rmse": None,
        "mape": None,
        "smape": None,
        "data_dir": str(data_dir_s),
        "model_params": dict(model_params),
    }


def _leaderboard_finalize_ok_row(
    payload: dict[str, Any],
    *,
    dataset_key: str,
    y_col_final: str,
    data_dir_s: str,
    model_params: dict[str, Any],
    task_group: str,
    backend_family: str,
) -> dict[str, Any]:
    payload["dataset"] = dataset_key
    payload["y_col"] = y_col_final
    payload["data_dir"] = data_dir_s
    payload["model_params"] = dict(model_params)
    payload["task_group"] = str(task_group)
    payload["backend_family"] = str(backend_family)
    payload["status"] = "ok"
    payload["skip_reason"] = ""
    payload["error_type"] = ""
    payload["error_message"] = ""
    return {k: v for k, v in payload.items() if not str(k).endswith("_by_step")}


def register_leaderboard_subparsers(sub: Any) -> None:
    leaderboard = sub.add_parser("leaderboard", help="Run a small builtin leaderboard")
    leaderboard_sub = leaderboard.add_subparsers(dest="leaderboard_command", required=True)

    leaderboard_naive = leaderboard_sub.add_parser("naive", help="Run naive baselines leaderboard")
    leaderboard_naive.add_argument("--dataset", required=True, help="Dataset key")
    leaderboard_naive.add_argument("--y-col", required=True, help="Target column name")
    leaderboard_naive.add_argument(
        "--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP
    )
    leaderboard_naive.add_argument("--step", type=int, default=1, help=_WALK_FORWARD_STEP_HELP)
    leaderboard_naive.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help=_MIN_TRAIN_SIZE_FIRST_WINDOW_HELP,
    )
    leaderboard_naive.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=_MAX_WINDOWS_LIMIT_HELP,
    )
    leaderboard_naive.add_argument(
        "--season-length",
        type=int,
        required=True,
        help="Season length for seasonal naive",
    )
    leaderboard_naive.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write JSON leaderboard",
    )
    leaderboard_naive.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help=_OUTPUT_JSON_FORMAT_HELP,
    )
    leaderboard_naive.set_defaults(_handler=_cmd_leaderboard_naive)

    leaderboard_models = leaderboard_sub.add_parser(
        "models", help="Run a leaderboard across registered models"
    )
    leaderboard_models.add_argument("--dataset", required=True, help="Dataset key")
    leaderboard_models.add_argument(
        "--y-col",
        type=str,
        default="",
        help="Optional target column name (default: use dataset spec default_y).",
    )
    leaderboard_models.add_argument(
        "--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP
    )
    leaderboard_models.add_argument(
        "--step", type=int, default=1, help=_WALK_FORWARD_STEP_HELP
    )
    leaderboard_models.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help=_MIN_TRAIN_SIZE_FIRST_WINDOW_HELP,
    )
    leaderboard_models.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=_MAX_WINDOWS_LIMIT_HELP,
    )
    leaderboard_models.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model keys (default: all core models).",
    )
    leaderboard_models.add_argument(
        "--include-optional",
        action="store_true",
        help="Include models requiring optional extras (e.g. statsmodels).",
    )
    leaderboard_models.add_argument(
        "--model-param",
        action="append",
        default=[],
        help=(
            "Model parameter as key=value (repeatable). Example: --model-param window=7. "
            "Applied to every selected model; models that don't accept the parameter may SKIP."
        ),
    )
    leaderboard_models.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write leaderboard output",
    )
    leaderboard_models.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help=_OUTPUT_JSON_FORMAT_HELP,
    )
    leaderboard_models.add_argument(
        "--task-group",
        choices=_TASK_GROUP_CHOICES,
        default="all",
        help="Optional protocol group filter (default: all).",
    )
    _cli_runtime.register_runtime_logging_args(leaderboard_models)
    leaderboard_models.set_defaults(_handler=_cmd_leaderboard_models)

    leaderboard_sweep = leaderboard_sub.add_parser(
        "sweep", help="Run a leaderboard across multiple datasets and models"
    )
    leaderboard_sweep.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset keys (e.g. catfish,ice_cream_interest)",
    )
    leaderboard_sweep.add_argument(
        "--y-col",
        type=str,
        default="",
        help="Optional target column name (default: use dataset spec default_y per dataset).",
    )
    leaderboard_sweep.add_argument(
        "--horizon", type=int, required=True, help=_FORECAST_HORIZON_HELP
    )
    leaderboard_sweep.add_argument("--step", type=int, default=1, help=_WALK_FORWARD_STEP_HELP)
    leaderboard_sweep.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help=_MIN_TRAIN_SIZE_FIRST_WINDOW_HELP,
    )
    leaderboard_sweep.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=_MAX_WINDOWS_LIMIT_HELP,
    )
    leaderboard_sweep.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model keys (default: all core models).",
    )
    leaderboard_sweep.add_argument(
        "--include-optional",
        action="store_true",
        help="Include models requiring optional extras (e.g. statsmodels).",
    )
    leaderboard_sweep.add_argument(
        "--model-param",
        action="append",
        default=[],
        help=(
            "Model parameter as key=value (repeatable). Example: --model-param window=7. "
            "Applied to every selected model; models that don't accept the parameter may SKIP "
            "unless --strict is set."
        ),
    )
    leaderboard_sweep.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Optional path to an existing sweep output (.json/.csv) to resume from. "
            "Rows matching the current datasets/models and evaluation settings are kept, and "
            "their (dataset, model) pairs are skipped."
        ),
    )
    leaderboard_sweep.add_argument(
        "--summary-output",
        type=str,
        default="",
        help="Optional path to write an aggregated model summary table (across datasets).",
    )
    leaderboard_sweep.add_argument(
        "--summary-format",
        choices=["json", "csv", "md"],
        default="json",
        help="Summary output format (default: json).",
    )
    leaderboard_sweep.add_argument(
        "--summary-sort",
        type=str,
        default="mae_rank_mean",
        help="Summary sort column (default: mae_rank_mean). Use --summary-sort=-col for descending.",
    )
    leaderboard_sweep.add_argument(
        "--summary-limit",
        type=int,
        default=0,
        help="Optional limit for summary rows (default: 0 = no limit).",
    )
    leaderboard_sweep.add_argument(
        "--summary-min-datasets",
        type=int,
        default=0,
        help="Optional minimum dataset coverage for a model in the summary (default: 0 = no filter).",
    )
    leaderboard_sweep.add_argument(
        "--failures-output",
        type=str,
        default="",
        help="Optional path to write SKIP/failure lines (one per line).",
    )
    leaderboard_sweep.add_argument(
        "--task-reports-output",
        type=str,
        default="",
        help="Optional path to write per-task execution stats.",
    )
    leaderboard_sweep.add_argument(
        "--task-reports-format",
        choices=["json", "csv", "md"],
        default="json",
        help="Task report output format (default: json).",
    )
    leaderboard_sweep.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write leaderboard output",
    )
    leaderboard_sweep.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help=_OUTPUT_JSON_FORMAT_HELP,
    )
    leaderboard_sweep.add_argument(
        "--task-group",
        choices=_TASK_GROUP_CHOICES,
        default="all",
        help="Optional protocol group filter (default: all).",
    )
    leaderboard_sweep.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers (default: 1)",
    )
    leaderboard_sweep.add_argument(
        "--backend",
        choices=["process", "thread"],
        default="process",
        help="Parallel backend when --jobs>1 (default: process)",
    )
    leaderboard_sweep.add_argument(
        "--progress",
        action="store_true",
        help="Print progress to stderr (safe for piping JSON/CSV/MD on stdout)",
    )
    leaderboard_sweep.add_argument(
        "--chunk-size",
        type=str,
        default="1",
        help=(
            "Number of models to evaluate per task, per dataset (default: 1). "
            "Use 0 to run all models for a dataset in a single task, or auto to choose "
            "a jobs-aware chunk size."
        ),
    )
    leaderboard_sweep.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first error instead of skipping failed (dataset, model) pairs.",
    )
    _cli_runtime.register_runtime_logging_args(leaderboard_sweep)
    leaderboard_sweep.set_defaults(_handler=_cmd_leaderboard_sweep)

    leaderboard_summarize = leaderboard_sub.add_parser(
        "summarize", help="Aggregate leaderboard rows across datasets"
    )
    leaderboard_summarize.add_argument(
        "--input",
        type=str,
        default="-",
        help="Input path (.json/.csv) or '-' for stdin (default: -).",
    )
    leaderboard_summarize.add_argument(
        "--input-format",
        choices=["auto", "json", "csv"],
        default="auto",
        help="Input format (default: auto).",
    )
    leaderboard_summarize.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write summary output",
    )
    leaderboard_summarize.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help=_OUTPUT_JSON_FORMAT_HELP,
    )
    leaderboard_summarize.add_argument(
        "--sort",
        type=str,
        default="mae_rank_mean",
        help="Sort column (default: mae_rank_mean). Use --sort=-col for descending.",
    )
    leaderboard_summarize.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of models in the summary (default: 0 = no limit).",
    )
    leaderboard_summarize.add_argument(
        "--min-datasets",
        type=int,
        default=0,
        help="Optional minimum dataset coverage for a model (default: 0 = no filter).",
    )
    leaderboard_summarize.add_argument(
        "--task-group",
        choices=_TASK_GROUP_CHOICES,
        default="all",
        help="Optional protocol group filter before aggregation (default: all).",
    )
    leaderboard_summarize.set_defaults(_handler=_cmd_leaderboard_summarize)


def _cmd_leaderboard_naive(args: argparse.Namespace) -> int:
    from .eval import eval_naive_last, eval_seasonal_naive

    rows = [
        eval_naive_last(
            dataset=str(args.dataset),
            y_col=str(args.y_col),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            max_windows=args.max_windows,
            data_dir=str(args.data_dir),
        ),
        eval_seasonal_naive(
            dataset=str(args.dataset),
            y_col=str(args.y_col),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            season_length=int(args.season_length),
            max_windows=args.max_windows,
            data_dir=str(args.data_dir),
        ),
    ]
    _cli_shared._emit(rows, output=args.output, fmt=str(args.format))
    return 0


def _cmd_leaderboard_models(args: argparse.Namespace) -> int:
    from .eval_forecast import eval_model_long_df
    from .models.registry import get_model_spec, list_models

    def _run() -> int:
        with _cli_runtime.phase_scope("params", payload=_log_payload(dataset=str(args.dataset))):
            y_col = str(args.y_col).strip() or None
            model_params = _cli_shared._parse_model_params(list(getattr(args, "model_param", [])))

            if str(args.models).strip():
                keys = [k.strip() for k in str(args.models).split(",") if k.strip()]
            else:
                keys = [k for k in list_models() if args.include_optional or not get_model_spec(k).requires]

            rows: list[dict[str, Any]] = []
            task_group_filter = _normalize_task_group_filter(getattr(args, "task_group", "all"))
            filtered_keys: list[str] = []
            for key in keys:
                task_group, _backend_family = _leaderboard_model_metadata(
                    str(key),
                    model_params=model_params,
                )
                if task_group_filter is not None and task_group != task_group_filter:
                    continue
                filtered_keys.append(str(key))

        if not filtered_keys:
            with _cli_runtime.phase_scope("emit", payload=_log_payload(rows=0, format=str(args.format))):
                _cli_shared._emit(rows, output=str(args.output), fmt=str(args.format))
            return 0

        with _cli_runtime.phase_scope(
            "prepare",
            payload=_log_payload(dataset=str(args.dataset), models=len(filtered_keys)),
        ):
            long_df, y_col_final = _get_or_build_leaderboard_long_df(
                dataset=str(args.dataset),
                y_col=y_col,
                data_dir=str(args.data_dir),
                model_params=model_params,
            )

        with _cli_runtime.phase_scope(
            "evaluate",
            payload=_log_payload(dataset=str(args.dataset), models=len(filtered_keys)),
        ):
            for key in filtered_keys:
                task_group, backend_family = _leaderboard_model_metadata(
                    str(key),
                    model_params=model_params,
                )
                try:
                    payload = eval_model_long_df(
                        model=str(key),
                        long_df=long_df,
                        horizon=int(args.horizon),
                        step=int(args.step),
                        min_train_size=int(args.min_train_size),
                        max_windows=args.max_windows,
                        model_params=dict(model_params),
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"SKIP {key}: {type(e).__name__}: {e}", file=sys.stderr)
                    rows.append(
                        _leaderboard_skip_row(
                            model_key=str(key),
                            dataset_key=str(args.dataset),
                            y_col_final=y_col_final,
                            horizon=int(args.horizon),
                            step=int(args.step),
                            min_train_size=int(args.min_train_size),
                            max_windows=args.max_windows,
                            data_dir_s=str(args.data_dir),
                            model_params=model_params,
                            task_group=task_group,
                            backend_family=backend_family,
                            error=e,
                        )
                    )
                    continue

                rows.append(
                    _leaderboard_finalize_ok_row(
                        payload,
                        dataset_key=str(args.dataset),
                        y_col_final=y_col_final,
                        data_dir_s=str(args.data_dir),
                        model_params=model_params,
                        task_group=task_group,
                        backend_family=backend_family,
                    )
                )

        rows.sort(
            key=lambda r: (
                _leaderboard_status_sort_key(r.get("status")),
                float(r.get("mae", float("inf"))) if r.get("mae") is not None else float("inf"),
                str(r.get("model", "")),
            )
        )
        with _cli_runtime.phase_scope(
            "emit",
            payload=_log_payload(rows=len(rows), format=str(args.format)),
        ):
            _cli_shared._emit(rows, output=str(args.output), fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="leaderboard models",
        payload=_log_payload(
            dataset=str(args.dataset),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            max_windows=args.max_windows,
            task_group=str(getattr(args, "task_group", "all")),
        ),
        action=_run,
    )


def _leaderboard_sweep_worker(
    dataset: str,
    model_keys: tuple[str, ...],
    y_col: str | None,
    horizon: int,
    step: int,
    min_train_size: int,
    max_windows: int | None,
    data_dir: str,
    strict: bool,
    model_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    from .eval_forecast import eval_model_long_df

    dataset_key = str(dataset)
    data_dir_s = str(data_dir).strip()
    long_df, y_col_final = _get_or_build_leaderboard_long_df(
        dataset=dataset_key,
        y_col=y_col,
        data_dir=data_dir_s,
        model_params=model_params,
    )

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for model_key in model_keys:
        task_group, backend_family = _leaderboard_model_metadata(
            str(model_key),
            model_params=model_params,
        )
        try:
            payload = eval_model_long_df(
                model=str(model_key),
                long_df=long_df,
                horizon=int(horizon),
                step=int(step),
                min_train_size=int(min_train_size),
                max_windows=max_windows,
                model_params=dict(model_params),
            )
        except Exception as e:  # noqa: BLE001
            if strict:
                raise RuntimeError(f"{dataset_key}/{model_key}: {type(e).__name__}: {e}") from e
            errors.append(f"SKIP {dataset_key}/{model_key}: {type(e).__name__}: {e}")
            rows.append(
                _leaderboard_skip_row(
                    model_key=str(model_key),
                    dataset_key=dataset_key,
                    y_col_final=y_col_final,
                    horizon=int(horizon),
                    step=int(step),
                    min_train_size=int(min_train_size),
                    max_windows=max_windows,
                    data_dir_s=data_dir_s,
                    model_params=model_params,
                    task_group=task_group,
                    backend_family=backend_family,
                    error=e,
                )
            )
            continue

        rows.append(
            _leaderboard_finalize_ok_row(
                payload,
                dataset_key=dataset_key,
                y_col_final=y_col_final,
                data_dir_s=data_dir_s,
                model_params=model_params,
                task_group=task_group,
                backend_family=backend_family,
            )
        )

    return (rows, errors)


def _resolve_leaderboard_sweep_model_keys(args: argparse.Namespace) -> list[str]:
    from .models.registry import get_model_spec, list_models

    if str(args.models).strip():
        return [k.strip() for k in str(args.models).split(",") if k.strip()]

    return [k for k in list_models() if args.include_optional or not get_model_spec(k).requires]


def _read_leaderboard_sweep_resume_rows(path: Path) -> list[dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8")
    lower = path.name.lower()
    fmt = "csv" if lower.endswith(".csv") else "json"
    if not (lower.endswith(".csv") or lower.endswith(".json")):
        try:
            json.loads(raw_text)
            fmt = "json"
        except Exception:  # noqa: BLE001
            fmt = "csv"

    if fmt == "json":
        parsed = json.loads(raw_text) if str(raw_text).strip() else []
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
        raise TypeError("--resume json must be a dict row or a list of dict rows")

    reader = csv.DictReader(io.StringIO(raw_text))
    return [dict(r) for r in reader]


def _leaderboard_sweep_dataset_y_cols(
    datasets: list[str],
    y_col: str | None,
) -> dict[str, str]:
    from .datasets.registry import get_dataset_spec

    dataset_y_cols: dict[str, str] = {}
    for dataset in datasets:
        spec = get_dataset_spec(str(dataset))
        dataset_y_cols[str(dataset)] = str(y_col).strip() if y_col is not None else str(spec.default_y)
    return dataset_y_cols


def _leaderboard_resume_value_as_int(v: object) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    lower_s = s.lower()
    if lower_s in {"none", "null"}:
        return None
    try:
        return int(float(s))
    except Exception:  # noqa: BLE001
        return None


def _leaderboard_resume_value_as_str(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _leaderboard_resume_value_as_params_dict(v: object) -> dict[str, Any] | None:
    if v is None:
        return None
    if isinstance(v, dict):
        return {str(k): val for k, val in v.items()}
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(parsed, dict):
            return {str(k): val for k, val in parsed.items()}
    return None


def _leaderboard_sweep_expected_resume_context(
    *,
    datasets: list[str],
    keys: list[str],
    y_col: str | None,
    horizon: int,
    step: int,
    min_train_size: int,
    max_windows: int | None,
    data_dir: str,
    model_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "datasets_set": set(datasets),
        "keys_set": set(keys),
        "dataset_y_cols": _leaderboard_sweep_dataset_y_cols(datasets, y_col),
        "horizon_i": horizon,
        "step_i": step,
        "min_train_size_i": min_train_size,
        "max_windows_i": max_windows,
        "expected_data_dir": data_dir,
        "expected_params_norm": json.loads(
            json.dumps(model_params, ensure_ascii=False, sort_keys=True)
        ),
    }


def _resume_row_matches_leaderboard_sweep(
    row: dict[str, Any],
    *,
    datasets_set: set[str],
    keys_set: set[str],
    dataset_y_cols: dict[str, str],
    horizon_i: int,
    step_i: int,
    min_train_size_i: int,
    max_windows_i: int | None,
    expected_data_dir: str,
    expected_params_norm: dict[str, Any],
) -> tuple[bool, str, str]:
    ds = _leaderboard_resume_value_as_str(row.get("dataset"))
    model = _leaderboard_resume_value_as_str(row.get("model"))
    if not ds or not model:
        return (False, ds, model)
    if ds not in datasets_set or model not in keys_set:
        return (False, ds, model)
    if _leaderboard_resume_value_as_str(row.get("y_col")) != dataset_y_cols.get(ds, ""):
        return (False, ds, model)
    if _leaderboard_resume_value_as_int(row.get("horizon")) != horizon_i:
        return (False, ds, model)
    if _leaderboard_resume_value_as_int(row.get("step")) != step_i:
        return (False, ds, model)
    if _leaderboard_resume_value_as_int(row.get("min_train_size")) != min_train_size_i:
        return (False, ds, model)
    if _leaderboard_resume_value_as_int(row.get("max_windows")) != max_windows_i:
        return (False, ds, model)
    if _leaderboard_resume_value_as_str(row.get("data_dir")) != expected_data_dir:
        return (False, ds, model)

    row_params = _leaderboard_resume_value_as_params_dict(row.get("model_params"))
    if row_params is None:
        return (not bool(expected_params_norm), ds, model)

    row_params_norm = json.loads(json.dumps(row_params, ensure_ascii=False, sort_keys=True))
    return (row_params_norm == expected_params_norm, ds, model)


def _load_leaderboard_sweep_resume_state(
    *,
    resume_path: str,
    datasets: list[str],
    keys: list[str],
    y_col: str | None,
    horizon: int,
    step: int,
    min_train_size: int,
    max_windows: int | None,
    data_dir: str,
    model_params: dict[str, Any],
    progress: bool,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not resume_path:
        return ([], set())

    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"--resume path not found: {path}")

    rows_raw = _read_leaderboard_sweep_resume_rows(path)
    context = _leaderboard_sweep_expected_resume_context(
        datasets=datasets,
        keys=keys,
        y_col=y_col,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        max_windows=max_windows,
        data_dir=data_dir,
        model_params=model_params,
    )

    indexed_resume_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows_raw:
        matches, ds, model = _resume_row_matches_leaderboard_sweep(row, **context)
        if not matches:
            continue
        indexed_resume_rows[(ds, model)] = row

    resume_rows = list(indexed_resume_rows.values())
    done_pairs = set(indexed_resume_rows)

    if progress:
        print(f"RESUME keep={len(resume_rows)} skip={len(done_pairs)}", file=sys.stderr)

    return (resume_rows, done_pairs)


def _build_leaderboard_sweep_tasks(
    datasets: list[str],
    keys: list[str],
    done_pairs: set[tuple[str, str]],
    *,
    chunk_size: int | str,
    jobs: int = 1,
) -> list[Any]:
    batch_execution = _get_batch_execution_module()
    tasks: list[Any] = []
    for dataset in datasets:
        remaining = [k for k in keys if (dataset, k) not in done_pairs]
        if not remaining:
            continue
        effective_chunk_size = _resolve_leaderboard_chunk_size(
            chunk_size,
            dataset_count=len(datasets),
            model_count=len(remaining),
            jobs=jobs,
        )
        if effective_chunk_size == 0:
            model_keys = tuple(remaining)
            tasks.append(
                batch_execution.BatchTask(
                    label=_parallel_task_label(dataset, model_keys),
                    task_args=(dataset, model_keys),
                    task_scope="leaderboard_sweep",
                    dataset=str(dataset),
                    model_count=len(model_keys),
                    requested_chunk_size=str(chunk_size),
                    resolved_chunk_size=int(effective_chunk_size),
                )
            )
            continue
        for i in range(0, len(remaining), effective_chunk_size):
            chunk = tuple(remaining[i : i + effective_chunk_size])
            if chunk:
                tasks.append(
                    batch_execution.BatchTask(
                        label=_parallel_task_label(dataset, chunk),
                        task_args=(dataset, chunk),
                        task_scope="leaderboard_sweep",
                        dataset=str(dataset),
                        model_count=len(chunk),
                        requested_chunk_size=str(chunk_size),
                        resolved_chunk_size=int(effective_chunk_size),
                    )
                )
    return tasks


def _resolve_leaderboard_chunk_size(
    raw_chunk_size: object,
    *,
    dataset_count: int,
    model_count: int,
    jobs: int,
) -> int:
    return _get_batch_execution_module().resolve_auto_chunk_size(
        raw_chunk_size,
        dataset_count=dataset_count,
        model_count=model_count,
        jobs=jobs,
    )


def _leaderboard_sweep_mae_key(v: object) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return float("inf")


def _merge_leaderboard_sweep_rows(
    resume_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in resume_rows:
        ds = str(row.get("dataset", "")).strip()
        model = str(row.get("model", "")).strip()
        if ds and model:
            merged[(ds, model)] = row
    for row in rows:
        ds = str(row.get("dataset", "")).strip()
        model = str(row.get("model", "")).strip()
        if ds and model:
            merged[(ds, model)] = row

    final_rows = list(merged.values()) if merged else rows
    final_rows.sort(
        key=lambda row: (
            str(row.get("dataset", "")),
            _leaderboard_status_sort_key(row.get("status")),
            _leaderboard_sweep_mae_key(row.get("mae", float("inf"))),
            str(row.get("model", "")),
        )
    )
    return final_rows


def _write_leaderboard_sweep_summary(
    args: argparse.Namespace,
    final_rows: list[dict[str, Any]],
) -> None:
    summary_output = str(getattr(args, "summary_output", "")).strip()
    if not summary_output:
        return

    summary_format = str(getattr(args, "summary_format", "json")).strip()
    summary_sort = str(getattr(args, "summary_sort", "mae_rank_mean"))
    summary_limit = int(getattr(args, "summary_limit", 0))
    summary_min_datasets = int(getattr(args, "summary_min_datasets", 0))

    summary_rows = (
        []
        if not final_rows
        else _summarize_leaderboard_rows(
            final_rows,
            sort=summary_sort,
            limit=summary_limit,
            min_datasets=summary_min_datasets,
        )
    )
    text = _cli_shared._format_table(
        summary_rows,
        columns=_leaderboard_summary_columns(),
        fmt=summary_format,
    )
    _cli_shared._write_output(text, output=summary_output)


def _write_leaderboard_sweep_task_reports(
    args: argparse.Namespace,
    task_stats: list[Any],
) -> None:
    task_reports_output = str(getattr(args, "task_reports_output", "")).strip()
    if not task_reports_output:
        return

    task_reports_format = str(getattr(args, "task_reports_format", "json")).strip()
    _get_batch_execution_module().emit_task_reports(
        task_stats,
        backend=str(args.backend),
        jobs=int(args.jobs),
        fmt=task_reports_format,
        output=task_reports_output,
    )


def _cmd_leaderboard_sweep(args: argparse.Namespace) -> int:
    def _run() -> int:
        with _cli_runtime.phase_scope(
            "params",
            payload=_log_payload(datasets=str(args.datasets), jobs=int(args.jobs)),
        ):
            datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
            if not datasets:
                raise ValueError("--datasets must contain at least one dataset key")

            y_col = str(args.y_col).strip() or None
            keys = _resolve_leaderboard_sweep_model_keys(args)

            jobs = int(args.jobs)
            if jobs <= 0:
                raise ValueError("--jobs must be >= 1")

            chunk_size = str(getattr(args, "chunk_size", "1"))
            _resolve_leaderboard_chunk_size(
                chunk_size,
                dataset_count=len(datasets),
                model_count=len(keys),
                jobs=jobs,
            )
            strict = bool(getattr(args, "strict", False))
            model_params = _cli_shared._parse_model_params(list(getattr(args, "model_param", [])))
            task_group_filter = _normalize_task_group_filter(getattr(args, "task_group", "all"))
            if task_group_filter is not None:
                keys = [
                    key
                    for key in keys
                    if _leaderboard_model_metadata(str(key), model_params=model_params)[0]
                    == task_group_filter
                ]
            resume_path = str(getattr(args, "resume", "")).strip()

        with _cli_runtime.phase_scope("resume", payload=_log_payload(resume=resume_path or None)):
            resume_rows, done_pairs = _load_leaderboard_sweep_resume_state(
                resume_path=resume_path,
                datasets=datasets,
                keys=keys,
                y_col=y_col,
                horizon=int(args.horizon),
                step=int(args.step),
                min_train_size=int(args.min_train_size),
                max_windows=None if args.max_windows is None else int(args.max_windows),
                data_dir=str(args.data_dir).strip(),
                model_params=model_params,
                progress=bool(args.progress),
            )

            tasks = _build_leaderboard_sweep_tasks(
                datasets,
                keys,
                done_pairs,
                chunk_size=chunk_size,
                jobs=jobs,
            )

        failure_lines: list[str] = []
        task_stats: list[Any] = []
        with _cli_runtime.phase_scope(
            "evaluate",
            payload=_log_payload(tasks=len(tasks), jobs=jobs, backend=str(args.backend)),
        ):
            rows, _failures = _run_parallel_tasks(
                tasks,
                jobs=jobs,
                backend=str(args.backend),
                progress=bool(args.progress),
                strict=strict,
                worker=_leaderboard_sweep_worker,
                worker_args=(
                    y_col,
                    int(args.horizon),
                    int(args.step),
                    int(args.min_train_size),
                    args.max_windows,
                    str(args.data_dir),
                    strict,
                    model_params,
                ),
                errors_out=failure_lines,
                stats_out=task_stats,
            )

        final_rows = _merge_leaderboard_sweep_rows(resume_rows, rows)

        with _cli_runtime.phase_scope(
            "emit",
            payload=_log_payload(rows=len(final_rows), format=str(args.format)),
        ):
            failures_output = str(getattr(args, "failures_output", "")).strip()
            if failures_output:
                _cli_shared._write_output("\n".join(failure_lines), output=failures_output)

            _write_leaderboard_sweep_summary(args, final_rows)
            _write_leaderboard_sweep_task_reports(args, task_stats)
            _cli_shared._emit(final_rows, output=str(args.output), fmt=str(args.format))
        return 0

    return _run_logged_command(
        args,
        command="leaderboard sweep",
        payload=_log_payload(
            datasets=str(args.datasets),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            max_windows=args.max_windows,
            jobs=int(args.jobs),
            backend=str(args.backend),
        ),
        action=_run,
    )


def _cmd_leaderboard_summarize(args: argparse.Namespace) -> int:
    input_path = str(getattr(args, "input", "-")).strip() or "-"
    input_fmt = str(getattr(args, "input_format", "auto")).strip().lower()

    raw_text = _read_leaderboard_summarize_input_text(input_path)
    fmt = _detect_leaderboard_summarize_input_format(input_path, input_fmt, raw_text)
    rows_raw = _parse_leaderboard_summarize_rows(raw_text, fmt)
    task_group_filter = _normalize_task_group_filter(getattr(args, "task_group", "all"))
    if task_group_filter is not None:
        rows_raw = [
            row
            for row in rows_raw
            if (str(row.get("task_group", "point")).strip().lower() or "point") == task_group_filter
        ]

    if not rows_raw:
        raise ValueError("No rows found in input")

    summary = _summarize_leaderboard_rows(
        rows_raw,
        sort=str(getattr(args, "sort", "mae_mean")),
        limit=int(getattr(args, "limit", 0)),
        min_datasets=int(getattr(args, "min_datasets", 0)),
    )
    _cli_shared._emit_table(
        summary,
        columns=_leaderboard_summary_columns(),
        output=str(args.output),
        fmt=str(args.format),
    )
    return 0


def _read_leaderboard_summarize_input_text(input_path: str) -> str:
    if input_path == "-":
        return sys.stdin.read()
    return Path(input_path).read_text(encoding="utf-8")


def _detect_leaderboard_summarize_input_format(
    input_path: str,
    input_fmt: str,
    raw_text: str,
) -> str:
    fmt = input_fmt
    if fmt != "auto":
        return fmt

    lower = input_path.lower()
    if input_path != "-" and lower.endswith(".csv"):
        return "csv"
    if input_path != "-" and lower.endswith(".json"):
        return "json"

    try:
        json.loads(raw_text)
        return "json"
    except Exception:  # noqa: BLE001
        return "csv"


def _parse_leaderboard_summarize_rows(raw_text: str, fmt: str) -> list[dict[str, Any]]:
    if fmt == "json":
        parsed = json.loads(raw_text) if str(raw_text).strip() else []
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
        raise TypeError("JSON input must be a dict row or a list of dict rows")

    if fmt == "csv":
        reader = csv.DictReader(io.StringIO(raw_text))
        return [dict(r) for r in reader]

    raise ValueError("--input-format must be one of: auto, json, csv")


def _leaderboard_summary_as_float(v: object) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:  # noqa: BLE001
        return None


def _leaderboard_summary_as_int(v: object) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:  # noqa: BLE001
        return None


def _leaderboard_summary_relative_metric_to_best(
    value: float,
    best: float,
    *,
    zero_tol: float = 1e-12,
) -> float:
    if math.isclose(best, 0.0, abs_tol=zero_tol):
        return 1.0 if math.isclose(value, 0.0, abs_tol=zero_tol) else float("inf")
    return float(value / best)


def _clean_leaderboard_summary_rows(
    rows_raw: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    cleaned: list[dict[str, Any]] = []
    bad = 0
    for row in rows_raw:
        model = str(row.get("model", "")).strip()
        dataset = str(row.get("dataset", "")).strip()
        status = str(row.get("status", "ok")).strip().lower() or "ok"
        if not model or not dataset:
            bad += 1
            continue
        if status != "ok":
            continue

        cleaned.append(
            {
                "model": model,
                "dataset": dataset,
                "task_group": str(row.get("task_group", "point")).strip() or "point",
                "mae": _leaderboard_summary_as_float(row.get("mae")),
                "rmse": _leaderboard_summary_as_float(row.get("rmse")),
                "mape": _leaderboard_summary_as_float(row.get("mape")),
                "smape": _leaderboard_summary_as_float(row.get("smape")),
                "n_points": _leaderboard_summary_as_int(row.get("n_points")),
            }
        )
    return (cleaned, bad)


def _leaderboard_summary_best_by_dataset_metric(
    cleaned: list[dict[str, Any]],
    metrics: list[str],
    *,
    rows_by_dataset: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> dict[tuple[str, str], float]:
    best_by_dataset_metric: dict[tuple[str, str, str], float] = {}
    grouped = (
        rows_by_dataset
        if rows_by_dataset is not None
        else {
            group_key: [
                row
                for row in cleaned
                if str(row["task_group"]) == group_key[0] and str(row["dataset"]) == group_key[1]
            ]
            for group_key in sorted({(str(row["task_group"]), str(row["dataset"])) for row in cleaned})
        }
    )
    for task_group, dataset in sorted(grouped):
        rows_ds = grouped[(task_group, dataset)]
        for metric in metrics:
            values = [float(row[metric]) for row in rows_ds if row.get(metric) is not None]
            if values:
                best_by_dataset_metric[(task_group, dataset, metric)] = float(min(values))
    return best_by_dataset_metric


def _leaderboard_summary_rank_by_dataset_metric_model(
    cleaned: list[dict[str, Any]],
    metrics: list[str],
    *,
    rows_by_dataset: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> dict[tuple[str, str, str], float]:
    rank_by_dataset_metric_model: dict[tuple[str, str, str, str], float] = {}
    grouped = (
        rows_by_dataset
        if rows_by_dataset is not None
        else {
            group_key: [
                row
                for row in cleaned
                if str(row["task_group"]) == group_key[0] and str(row["dataset"]) == group_key[1]
            ]
            for group_key in sorted({(str(row["task_group"]), str(row["dataset"])) for row in cleaned})
        }
    )
    for task_group, dataset in sorted(grouped):
        rows_ds = grouped[(task_group, dataset)]
        for metric in metrics:
            vals = [
                (str(row["model"]), row.get(metric)) for row in rows_ds if row.get(metric) is not None
            ]
            if not vals:
                continue
            vals_sorted = sorted(vals, key=lambda t: float(t[1]))  # type: ignore[arg-type]
            rank = 1
            prev: float | None = None
            for model, v_obj in vals_sorted:
                v = float(v_obj)  # type: ignore[arg-type]
                if prev is None:
                    prev = v
                elif v != prev:
                    rank += 1
                    prev = v
                rank_by_dataset_metric_model[(task_group, dataset, metric, model)] = float(rank)
    return rank_by_dataset_metric_model


def _leaderboard_summary_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _leaderboard_summary_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _leaderboard_summary_weighted_mean(pairs: list[tuple[float, int]]) -> float | None:
    weight_sum = float(sum(weight for _value, weight in pairs))
    if weight_sum <= 0:
        return None
    return float(sum(value * float(weight) for value, weight in pairs) / weight_sum)


def _leaderboard_summary_metric_values(
    items: list[dict[str, Any]],
    metric: str,
) -> list[float]:
    return [float(item[metric]) for item in items if item.get(metric) is not None]


def _leaderboard_summary_metric_weighted_pairs(
    items: list[dict[str, Any]],
    metric: str,
) -> list[tuple[float, int]]:
    return [
        (float(item[metric]), int(item["n_points"]))
        for item in items
        if item.get(metric) is not None
        and item.get("n_points") is not None
        and int(item["n_points"]) > 0
    ]


def _leaderboard_summary_metric_relative_values(
    items: list[dict[str, Any]],
    metric: str,
    *,
    best_by_dataset_metric: dict[tuple[str, str, str], float],
) -> list[float]:
    rels: list[float] = []
    for item in items:
        v_obj = item.get(metric)
        if v_obj is None:
            continue
        best = best_by_dataset_metric.get(
            (str(item["task_group"]), str(item["dataset"]), metric)
        )
        if best is None:
            continue
        rels.append(
            _leaderboard_summary_relative_metric_to_best(float(v_obj), float(best))
        )
    return rels


def _leaderboard_summary_metric_relative_pairs(
    items: list[dict[str, Any]],
    metric: str,
    *,
    best_by_dataset_metric: dict[tuple[str, str, str], float],
) -> list[tuple[float, int]]:
    rel_pairs: list[tuple[float, int]] = []
    for item in items:
        v_obj = item.get(metric)
        if v_obj is None:
            continue
        weight = item.get("n_points")
        if weight is None or int(weight) <= 0:
            continue
        best = best_by_dataset_metric.get(
            (str(item["task_group"]), str(item["dataset"]), metric)
        )
        if best is None:
            continue
        rel_pairs.append(
            (
                _leaderboard_summary_relative_metric_to_best(float(v_obj), float(best)),
                int(weight),
            )
        )
    return rel_pairs


def _leaderboard_summary_metric_ranks(
    items: list[dict[str, Any]],
    metric: str,
    *,
    model: str,
    rank_by_dataset_metric_model: dict[tuple[str, str, str, str], float],
) -> list[float]:
    return [
        float(
            rank_by_dataset_metric_model[
                (str(item["task_group"]), str(item["dataset"]), metric, model)
            ]
        )
        for item in items
        if (str(item["task_group"]), str(item["dataset"]), metric, model)
        in rank_by_dataset_metric_model
    ]


def _leaderboard_summary_metric_rank_pairs(
    items: list[dict[str, Any]],
    metric: str,
    *,
    model: str,
    rank_by_dataset_metric_model: dict[tuple[str, str, str, str], float],
) -> list[tuple[float, int]]:
    return [
        (
            float(
                rank_by_dataset_metric_model[
                    (str(item["task_group"]), str(item["dataset"]), metric, model)
                ]
            ),
            int(item["n_points"]),
        )
        for item in items
        if (str(item["task_group"]), str(item["dataset"]), metric, model)
        in rank_by_dataset_metric_model
        and item.get("n_points") is not None
        and int(item["n_points"]) > 0
    ]


def _build_leaderboard_metric_contexts(
    items: list[dict[str, Any]],
    *,
    metrics: list[str],
    model: str,
    best_by_dataset_metric: dict[tuple[str, str, str], float],
    rank_by_dataset_metric_model: dict[tuple[str, str, str, str], float],
) -> dict[str, dict[str, list[Any]]]:
    contexts: dict[str, dict[str, list[Any]]] = {
        str(metric): {
            "values": [],
            "weighted_pairs": [],
            "relative_values": [],
            "relative_pairs": [],
            "rank_values": [],
            "rank_pairs": [],
        }
        for metric in metrics
    }

    for item in items:
        task_group = str(item["task_group"])
        dataset = str(item["dataset"])
        n_points = item.get("n_points")
        weight = None if n_points is None else int(n_points)
        for metric in metrics:
            metric_name = str(metric)
            context = contexts[metric_name]
            value_obj = item.get(metric_name)
            if value_obj is not None:
                value_f = float(value_obj)
                context["values"].append(value_f)
                if weight is not None and weight > 0:
                    context["weighted_pairs"].append((value_f, weight))

                best = best_by_dataset_metric.get((task_group, dataset, metric_name))
                if best is not None:
                    relative_f = _leaderboard_summary_relative_metric_to_best(value_f, float(best))
                    context["relative_values"].append(relative_f)
                    if weight is not None and weight > 0:
                        context["relative_pairs"].append((relative_f, weight))

            rank = rank_by_dataset_metric_model.get((task_group, dataset, metric_name, model))
            if rank is not None:
                rank_f = float(rank)
                context["rank_values"].append(rank_f)
                if weight is not None and weight > 0:
                    context["rank_pairs"].append((rank_f, weight))

    return contexts


def _populate_leaderboard_metric_summary(
    row: dict[str, Any],
    *,
    metric: str,
    metric_contexts: dict[str, dict[str, list[Any]]],
) -> None:
    context = metric_contexts[str(metric)]
    values = [float(value) for value in context["values"]]
    row[f"{metric}_mean"] = _leaderboard_summary_mean(values)
    row[f"{metric}_median"] = _leaderboard_summary_median(values)
    row[f"{metric}_wmean"] = _leaderboard_summary_weighted_mean(list(context["weighted_pairs"]))

    rel_values = [float(value) for value in context["relative_values"]]
    row[f"{metric}_rel_mean"] = _leaderboard_summary_mean(rel_values)
    row[f"{metric}_rel_median"] = _leaderboard_summary_median(rel_values)
    row[f"{metric}_rel_wmean"] = _leaderboard_summary_weighted_mean(list(context["relative_pairs"]))

    ranks = [float(value) for value in context["rank_values"]]
    row[f"{metric}_rank_mean"] = _leaderboard_summary_mean(ranks)
    row[f"{metric}_rank_wmean"] = _leaderboard_summary_weighted_mean(list(context["rank_pairs"]))


def _leaderboard_summary_metric_group_average(
    row: dict[str, Any],
    metrics: list[str],
    suffix: str,
) -> float | None:
    values = [row.get(f"{metric}_{suffix}") for metric in metrics]
    present = [float(v) for v in values if v is not None]
    if len(present) != len(metrics):
        return None
    return float(sum(present) / float(len(metrics)))


def _leaderboard_model_summary_row(
    task_group: str,
    model: str,
    items: list[dict[str, Any]],
    *,
    metrics: list[str],
    n_datasets_total: int,
    best_by_dataset_metric: dict[tuple[str, str, str], float],
    rank_by_dataset_metric_model: dict[tuple[str, str, str, str], float],
) -> dict[str, Any]:
    datasets = {str(item["dataset"]) for item in items}
    metric_contexts = _build_leaderboard_metric_contexts(
        items,
        metrics=metrics,
        model=model,
        best_by_dataset_metric=best_by_dataset_metric,
        rank_by_dataset_metric_model=rank_by_dataset_metric_model,
    )
    row: dict[str, Any] = {
        "model": model,
        "task_group": str(task_group),
        "n_datasets": int(len(datasets)),
        "n_datasets_total": int(n_datasets_total),
        "dataset_coverage": (
            None
            if n_datasets_total <= 0
            else float(int(len(datasets)) / float(int(n_datasets_total)))
        ),
        "n_rows": int(len(items)),
    }

    for metric in metrics:
        _populate_leaderboard_metric_summary(
            row,
            metric=metric,
            metric_contexts=metric_contexts,
        )

    row["score_rank_mean"] = _leaderboard_summary_metric_group_average(
        row, metrics, "rank_mean"
    )
    row["score_rank_wmean"] = _leaderboard_summary_metric_group_average(
        row, metrics, "rank_wmean"
    )
    row["score_rel_mean"] = _leaderboard_summary_metric_group_average(
        row, metrics, "rel_mean"
    )
    row["score_rel_wmean"] = _leaderboard_summary_metric_group_average(
        row, metrics, "rel_wmean"
    )
    return row


def _leaderboard_summary_sort_number(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def _leaderboard_summary_sort_value_key(
    v: float | None,
    *,
    descending: bool,
) -> tuple[int, float]:
    if v is None:
        return (1, 0.0)

    value = float(v)
    if descending:
        value = -value
    return (0, value)


def _sort_leaderboard_summary_rows(
    rows: list[dict[str, Any]],
    *,
    sort: str,
    limit: int,
    min_datasets: int,
) -> list[dict[str, Any]]:
    if not rows:
        return rows

    sort_s = str(sort).strip() or "mae_rank_mean"
    descending = False
    if sort_s.startswith("-"):
        descending = True
        sort_s = sort_s[1:].strip()
    if sort_s not in rows[0]:
        raise ValueError(f"--sort must be a known summary column, got: {sort_s!r}")

    secondary = "mae_mean" if "mae_mean" in rows[0] else ""

    def _sort_key(row: dict[str, Any]) -> tuple[int, float, int, float, str]:
        missing, val_key = _leaderboard_summary_sort_value_key(
            _leaderboard_summary_sort_number(row.get(sort_s)),
            descending=descending,
        )
        secondary_value = _leaderboard_summary_sort_number(row.get(secondary)) if secondary else None
        smissing, sval_key = _leaderboard_summary_sort_value_key(
            secondary_value,
            descending=descending,
        )
        return (missing, val_key, smissing, sval_key, str(row.get("model", "")))

    rows.sort(key=_sort_key)

    min_datasets_i = int(min_datasets)
    if min_datasets_i > 0:
        rows = [row for row in rows if int(row.get("n_datasets", 0) or 0) >= min_datasets_i]

    limit_i = int(limit)
    if limit_i > 0:
        rows = rows[:limit_i]

    return rows


def _summarize_leaderboard_rows(
    rows_raw: list[dict[str, Any]],
    *,
    sort: str,
    limit: int,
    min_datasets: int = 0,
) -> list[dict[str, Any]]:
    cleaned, bad = _clean_leaderboard_summary_rows(rows_raw)
    if not cleaned:
        raise ValueError(f"No valid rows found (bad_rows={bad})")

    by_group_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    rows_by_dataset: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cleaned:
        task_group = str(row["task_group"])
        model = str(row["model"])
        dataset = str(row["dataset"])
        by_group_model.setdefault((task_group, model), []).append(row)
        rows_by_dataset.setdefault((task_group, dataset), []).append(row)

    n_datasets_total_by_group: dict[str, int] = {}
    for task_group, _dataset in rows_by_dataset:
        n_datasets_total_by_group[task_group] = int(n_datasets_total_by_group.get(task_group, 0) + 1)

    metrics = ["mae", "rmse", "mape", "smape"]
    best_by_dataset_metric = _leaderboard_summary_best_by_dataset_metric(
        cleaned,
        metrics,
        rows_by_dataset=rows_by_dataset,
    )
    rank_by_dataset_metric_model = _leaderboard_summary_rank_by_dataset_metric_model(
        cleaned,
        metrics,
        rows_by_dataset=rows_by_dataset,
    )

    out = [
        _leaderboard_model_summary_row(
            task_group,
            model,
            items,
            metrics=metrics,
            n_datasets_total=int(n_datasets_total_by_group.get(task_group, 0)),
            best_by_dataset_metric=best_by_dataset_metric,
            rank_by_dataset_metric_model=rank_by_dataset_metric_model,
        )
        for (task_group, model), items in by_group_model.items()
    ]

    return _sort_leaderboard_summary_rows(
        out,
        sort=sort,
        limit=limit,
        min_datasets=min_datasets,
    )


def _leaderboard_summary_columns() -> list[str]:
    return [
        "model",
        "n_datasets",
        "n_datasets_total",
        "dataset_coverage",
        "n_rows",
        "task_group",
        "score_rank_mean",
        "score_rank_wmean",
        "score_rel_mean",
        "score_rel_wmean",
        "mae_mean",
        "mae_median",
        "mae_wmean",
        "mae_rel_mean",
        "mae_rel_median",
        "mae_rel_wmean",
        "rmse_mean",
        "rmse_median",
        "rmse_wmean",
        "rmse_rel_mean",
        "rmse_rel_median",
        "rmse_rel_wmean",
        "mape_mean",
        "mape_median",
        "mape_wmean",
        "mape_rel_mean",
        "mape_rel_median",
        "mape_rel_wmean",
        "smape_mean",
        "smape_median",
        "smape_wmean",
        "smape_rel_mean",
        "smape_rel_median",
        "smape_rel_wmean",
        "mae_rank_mean",
        "mae_rank_wmean",
        "rmse_rank_mean",
        "rmse_rank_wmean",
        "mape_rank_mean",
        "mape_rank_wmean",
        "smape_rank_mean",
        "smape_rank_wmean",
    ]


def _run_parallel_tasks(
    tasks: list[Any],
    *,
    jobs: int,
    backend: str,
    progress: bool,
    strict: bool,
    worker: Any,
    worker_args: tuple[Any, ...] = (),
    errors_out: list[str] | None = None,
    stats_out: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    n = len(tasks)
    if n <= 0:
        return ([], 0)

    batch_execution = _get_batch_execution_module()
    normalized_tasks = [
        task
        if isinstance(task, batch_execution.BatchTask)
        else batch_execution.BatchTask(
            label=_parallel_task_label(task[0], task[1]),
            task_args=(task[0], task[1]),
        )
        for task in tasks
    ]

    if jobs <= 1:
        return _run_parallel_tasks_sequential(
            normalized_tasks,
            progress=progress,
            strict=strict,
            worker=worker,
            worker_args=worker_args,
            errors_out=errors_out,
            stats_out=stats_out,
        )

    max_workers = min(int(jobs), max(1, n))
    backend_s = str(backend).strip().lower()
    executor = _build_parallel_task_executor(backend_s=backend_s, max_workers=max_workers)

    import concurrent.futures

    done = 0
    failures = 0
    out: list[dict[str, Any]] = []
    try:
        fut_to_task = {}
        for task in normalized_tasks:
            fut = executor.submit(worker, *task.task_args, *worker_args)
            fut_to_task[fut] = (task, time.perf_counter())
        for fut in concurrent.futures.as_completed(fut_to_task):
            task, submitted_at = fut_to_task[fut]
            label = task.label
            rows, task_failures, elapsed_seconds = _resolve_parallel_task_result(
                fut,
                label=label,
                strict=strict,
                sibling_futures=fut_to_task.keys(),
                errors_out=errors_out,
            )
            effective_elapsed = (
                float(elapsed_seconds)
                if float(elapsed_seconds) > 0.0
                else float(time.perf_counter() - submitted_at)
            )
            out.extend(rows)
            failures += task_failures
            if stats_out is not None:
                stats_out.append(
                    batch_execution.BatchTaskStat(
                        label=label,
                        elapsed_seconds=effective_elapsed,
                        row_count=len(rows),
                        failure_count=task_failures,
                        task_scope=str(task.task_scope),
                        dataset=str(task.dataset),
                        model_count=int(task.model_count),
                        requested_chunk_size=str(task.requested_chunk_size),
                        resolved_chunk_size=int(task.resolved_chunk_size),
                    )
                )
            done += 1
            if progress:
                print(f"DONE {done}/{n} {label}", file=sys.stderr)
    finally:
        executor.shutdown(cancel_futures=True)

    return (out, failures)


def _parallel_task_label(dataset: str, model_keys: tuple[str, ...]) -> str:
    if len(model_keys) == 1:
        return f"{dataset}/{model_keys[0]}"
    return f"{dataset}/[{len(model_keys)} models]"


def _record_parallel_task_errors(
    errors: list[str],
    *,
    errors_out: list[str] | None,
) -> int:
    return _get_batch_execution_module().record_task_errors(errors, errors_out=errors_out)


def _run_parallel_tasks_sequential(
    tasks: list[Any],
    *,
    progress: bool,
    strict: bool,
    worker: Any,
    worker_args: tuple[Any, ...],
    errors_out: list[str] | None,
    stats_out: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    batch_execution = _get_batch_execution_module()
    batch_tasks = [
        task
        if isinstance(task, batch_execution.BatchTask)
        else batch_execution.BatchTask(
            label=_parallel_task_label(task[0], task[1]),
            task_args=(task[0], task[1]),
        )
        for task in tasks
    ]
    return batch_execution.run_batch_tasks_sequential(
        batch_tasks,
        progress=progress,
        strict=strict,
        worker=worker,
        worker_args=worker_args,
        errors_out=errors_out,
        stats_out=stats_out,
    )


def _build_parallel_task_executor(*, backend_s: str, max_workers: int) -> Any:
    return _get_batch_execution_module().build_task_executor(
        backend_s=backend_s,
        max_workers=max_workers,
    )


def _resolve_parallel_task_result(
    fut: Any,
    *,
    label: str,
    strict: bool,
    sibling_futures: Any,
    errors_out: list[str] | None,
) -> tuple[list[dict[str, Any]], int, float]:
    rows, failures, elapsed_seconds = _get_batch_execution_module().resolve_task_result(
        fut,
        label=label,
        strict=strict,
        sibling_futures=sibling_futures,
        errors_out=errors_out,
    )
    return rows, failures, elapsed_seconds
