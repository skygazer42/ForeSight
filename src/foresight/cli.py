from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="foresight",
        description="ForeSight: time-series forecasting models and utilities.",
    )
    p.add_argument("--version", action="store_true", help="Print version and exit.")
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

    models = sub.add_parser("models", help="Model registry utilities")
    models_sub = models.add_subparsers(dest="models_command", required=True)

    models_list = models_sub.add_parser("list", help="List available models")
    models_list.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        help="Output format (default: tsv)",
    )
    models_list.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
    )
    models_list.set_defaults(_handler=_cmd_models_list)

    models_info = models_sub.add_parser("info", help="Show details about a model")
    models_info.add_argument("key", help="Model key (see: `foresight models list`)")
    models_info.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Output format (default: json)",
    )
    models_info.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
    )
    models_info.set_defaults(_handler=_cmd_models_info)

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

    datasets = sub.add_parser("datasets", help="Dataset utilities")
    datasets_sub = datasets.add_subparsers(dest="datasets_command", required=True)

    datasets_list = datasets_sub.add_parser("list", help="List available datasets")
    datasets_list.add_argument(
        "--with-path",
        action="store_true",
        help="Include resolved local file paths in the output.",
    )
    datasets_list.set_defaults(_handler=_cmd_datasets_list)

    datasets_preview = datasets_sub.add_parser("preview", help="Preview a dataset (head)")
    datasets_preview.add_argument("key", help="Dataset key (see: `foresight datasets list`)")
    datasets_preview.add_argument("--nrows", type=int, default=20, help="Number of rows to preview")
    datasets_preview.set_defaults(_handler=_cmd_datasets_preview)

    datasets_path = datasets_sub.add_parser("path", help="Print the local path for a dataset")
    datasets_path.add_argument("key", help="Dataset key (see: `foresight datasets list`)")
    datasets_path.set_defaults(_handler=_cmd_datasets_path)

    datasets_validate = datasets_sub.add_parser(
        "validate", help="Smoke-check local dataset files and basic schemas"
    )
    datasets_validate.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional dataset key to validate (default: validate all).",
    )
    datasets_validate.add_argument(
        "--nrows",
        type=int,
        default=5,
        help="Number of rows to load for large datasets (default: 5)",
    )
    datasets_validate.add_argument(
        "--check-time",
        action="store_true",
        help="Check per-series ds ordering and duplicates (uses dataset spec).",
    )
    datasets_validate.set_defaults(_handler=_cmd_datasets_validate)

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

    leaderboard = sub.add_parser("leaderboard", help="Run a small builtin leaderboard")
    leaderboard_sub = leaderboard.add_subparsers(dest="leaderboard_command", required=True)

    leaderboard_naive = leaderboard_sub.add_parser("naive", help="Run naive baselines leaderboard")
    leaderboard_naive.add_argument("--dataset", required=True, help="Dataset key")
    leaderboard_naive.add_argument("--y-col", required=True, help="Target column name")
    leaderboard_naive.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    leaderboard_naive.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    leaderboard_naive.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    leaderboard_naive.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
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
        help="Output format (default: json)",
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
    leaderboard_models.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    leaderboard_models.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    leaderboard_models.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    leaderboard_models.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
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
        "--output",
        type=str,
        default="",
        help="Optional path to write leaderboard output",
    )
    leaderboard_models.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    leaderboard_models.set_defaults(_handler=_cmd_leaderboard_models)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        # Avoid importing package metadata; keep it simple and dependency-free.
        from . import __version__

        print(f"ForeSight {__version__}")
        return 0

    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 0

    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001
        if getattr(args, "debug", False):
            raise
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def _coerce_model_param_value(raw: str) -> Any:
    s = str(raw).strip()
    lower = s.lower()

    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None

    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return tuple(_coerce_model_param_value(p) for p in parts)

    try:
        return int(s)
    except Exception:  # noqa: BLE001
        pass
    try:
        return float(s)
    except Exception:  # noqa: BLE001
        pass
    return s


def _parse_model_params(items: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        if "=" not in str(item):
            raise ValueError(f"--model-param must be key=value, got: {item!r}")
        key, value = str(item).split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--model-param must be key=value, got: {item!r}")
        params[key] = _coerce_model_param_value(value)
    return params


def _emit_text(text: str, *, output: str) -> None:
    print(text)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


def _emit_dataframe(df: Any, *, output: str, fmt: str) -> None:
    if fmt == "csv":
        text = df.to_csv(index=False).rstrip("\n")
        _emit_text(text, output=output)
        return
    if fmt == "json":
        text = df.to_json(orient="records", date_format="iso")
        _emit_text(text, output=output)
        return
    raise ValueError(f"Unknown dataframe format: {fmt!r}")


def _cmd_models_list(args: argparse.Namespace) -> int:
    from .models.registry import get_model_spec, list_models

    rows: list[dict[str, Any]] = []
    for key in list_models():
        spec = get_model_spec(key)
        rows.append(
            {
                "key": spec.key,
                "interface": str(spec.interface),
                "requires": ",".join(spec.requires),
                "description": spec.description,
                "default_params": dict(spec.default_params),
            }
        )

    fmt = str(args.format)
    if fmt == "json":
        _emit(rows, output=str(args.output), fmt="json")
        return 0

    lines = [f"{r['key']}\t{r['requires']}\t{r['description']}" for r in rows]
    _emit_text("\n".join(lines), output=str(args.output))
    return 0


def _cmd_models_info(args: argparse.Namespace) -> int:
    from .models.registry import get_model_spec

    spec = get_model_spec(str(args.key))
    payload = {
        "key": spec.key,
        "interface": str(spec.interface),
        "description": spec.description,
        "requires": list(spec.requires),
        "default_params": dict(spec.default_params),
        "param_help": dict(spec.param_help),
    }
    _emit(payload, output=str(args.output), fmt=str(args.format))
    return 0


def _cmd_cv_run(args: argparse.Namespace) -> int:
    from .cv import cross_validation_predictions

    model_params = _parse_model_params(list(args.model_param))
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
    _emit_dataframe(df, output=str(args.output), fmt=str(args.format))
    return 0


def _cmd_datasets_list(args: argparse.Namespace) -> int:
    from .datasets.registry import describe_dataset, list_datasets, resolve_dataset_path

    for key in list_datasets():
        if args.with_path:
            path = resolve_dataset_path(key, data_dir=str(args.data_dir))
            print(f"{key}\t{path.as_posix()}\t{describe_dataset(key)}")
        else:
            print(f"{key}\t{describe_dataset(key)}")
    return 0


def _cmd_datasets_preview(args: argparse.Namespace) -> int:
    from .datasets.loaders import load_dataset

    df = load_dataset(args.key, nrows=int(args.nrows), data_dir=str(args.data_dir))
    print(df.head(int(args.nrows)).to_string(index=False))
    return 0


def _cmd_datasets_path(args: argparse.Namespace) -> int:
    from .datasets.registry import resolve_dataset_path

    path = resolve_dataset_path(str(args.key), data_dir=str(args.data_dir))
    print(path.as_posix())
    return 0


def _cmd_datasets_validate(args: argparse.Namespace) -> int:
    from .datasets.loaders import load_dataset
    from .datasets.registry import get_dataset_spec, list_datasets

    nrows = int(args.nrows)
    keys = [str(args.dataset)] if str(args.dataset) else list_datasets()
    failures = 0
    for key in keys:
        try:
            spec = get_dataset_spec(key)
            df = load_dataset(key, nrows=nrows, data_dir=str(args.data_dir))
            if len(df) <= 0:
                raise ValueError("loaded 0 rows")
            missing = sorted(spec.expected_columns.difference(df.columns))
            if missing:
                raise ValueError(f"missing columns: {missing}")

            if spec.parse_dates:
                import pandas as pd

                for col in spec.parse_dates:
                    if col not in df.columns:
                        raise ValueError(f"missing parse_dates column: {col!r}")
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        raise ValueError(f"parse_dates column is not datetime: {col!r}")
                    if df[col].isna().any():
                        raise ValueError(f"parse_dates column contains NaT/NA values: {col!r}")

            if bool(args.check_time):
                from .data.format import to_long, validate_long_df

                long_df = to_long(
                    df,
                    time_col=spec.time_col,
                    y_col=spec.default_y,
                    id_cols=tuple(spec.group_cols),
                    dropna=True,
                )
                long_df = long_df.sort_values(["unique_id", "ds"], kind="mergesort")
                validate_long_df(long_df, require_sorted=True, require_unique_ds=True)
            print(f"OK {key} rows={len(df)} cols={len(df.columns)}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {key}: {type(e).__name__}: {e}", file=sys.stderr)

    return 1 if failures else 0


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
    _emit(payload, output=args.output, fmt=str(args.format))

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
    _emit(payload, output=args.output, fmt=str(args.format))
    return 0


def _cmd_eval_run(args: argparse.Namespace) -> int:
    from .eval_forecast import eval_model

    model_params = _parse_model_params(list(args.model_param))
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
    _emit(payload, output=args.output, fmt=str(args.format))
    return 0


def _cmd_eval_csv(args: argparse.Namespace) -> int:
    from .data.format import to_long
    from .eval_forecast import eval_model_long_df
    from .io import ensure_datetime, load_csv, parse_id_cols
    from .models.registry import get_model_spec

    model_params = _parse_model_params(list(args.model_param))
    id_cols = parse_id_cols(str(args.id_cols))

    df = load_csv(str(args.path))
    if bool(args.parse_dates):
        ensure_datetime(df, str(args.time_col))

    model_spec = get_model_spec(str(args.model))
    x_cols: tuple[str, ...] = ()
    if model_spec.interface == "global" and "x_cols" in model_params:
        raw = model_params.get("x_cols")
        if raw is not None:
            if isinstance(raw, str):
                x_cols = tuple([p.strip() for p in raw.split(",") if p.strip()])
            elif isinstance(raw, list | tuple):
                x_cols = tuple([str(p).strip() for p in raw if str(p).strip()])
            else:
                s = str(raw).strip()
                x_cols = (s,) if s else ()

    long_df = to_long(
        df,
        time_col=str(args.time_col),
        y_col=str(args.y_col),
        id_cols=id_cols,
        x_cols=x_cols,
        dropna=True,
    )
    payload = eval_model_long_df(
        model=str(args.model),
        long_df=long_df,
        horizon=int(args.horizon),
        step=int(args.step),
        min_train_size=int(args.min_train_size),
        max_windows=args.max_windows,
        max_train_size=args.max_train_size,
        conformal_levels=str(args.conformal_levels).strip() or None,
        conformal_per_step=(not bool(args.conformal_pooled)),
        model_params=model_params,
    )
    payload.update(
        {
            "dataset": str(args.path),
            "time_col": str(args.time_col),
            "y_col": str(args.y_col),
            "id_cols": list(id_cols),
        }
    )
    _emit(payload, output=str(args.output), fmt=str(args.format))
    return 0


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
    _emit(rows, output=args.output, fmt=str(args.format))
    return 0


def _cmd_leaderboard_models(args: argparse.Namespace) -> int:
    from .eval_forecast import eval_model
    from .models.registry import get_model_spec, list_models

    y_col = str(args.y_col).strip() or None

    if str(args.models).strip():
        keys = [k.strip() for k in str(args.models).split(",") if k.strip()]
    else:
        keys = [k for k in list_models() if args.include_optional or not get_model_spec(k).requires]

    rows: list[dict[str, Any]] = []
    for key in keys:
        try:
            payload = eval_model(
                model=str(key),
                dataset=str(args.dataset),
                y_col=y_col,
                horizon=int(args.horizon),
                step=int(args.step),
                min_train_size=int(args.min_train_size),
                max_windows=args.max_windows,
                data_dir=str(args.data_dir),
            )
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {key}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        slim = {k: v for k, v in payload.items() if not str(k).endswith("_by_step")}
        rows.append(slim)

    rows.sort(key=lambda r: float(r.get("mae", float("inf"))))
    _emit(rows, output=str(args.output), fmt=str(args.format))
    return 0


def _emit(payload: object, *, output: str, fmt: str) -> None:
    text = _format_payload(payload, fmt=fmt)
    print(text)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


def _format_payload(payload: object, *, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if fmt == "csv":
        rows: list[dict]
        if isinstance(payload, dict):
            rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise TypeError("csv format expects a dict row or list of dict rows")
        return _format_csv(rows)
    if fmt == "md":
        rows_md: list[dict]
        if isinstance(payload, dict):
            rows_md = [payload]
        elif isinstance(payload, list):
            rows_md = payload
        else:
            raise TypeError("md format expects a dict row or list of dict rows")
        return _format_markdown(rows_md)
    raise ValueError(f"Unknown format: {fmt!r}")


def _leaderboard_columns() -> list[str]:
    # Stable output makes diffs and automation easier.
    return [
        "model",
        "dataset",
        "y_col",
        "horizon",
        "step",
        "min_train_size",
        "max_windows",
        "season_length",
        "n_series",
        "n_series_skipped",
        "n_windows",
        "n_points",
        "mae",
        "rmse",
        "mape",
        "smape",
    ]


def _format_csv(rows: list[dict]) -> str:
    cols = _leaderboard_columns()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in cols})
    return buf.getvalue().rstrip("\n")


def _format_markdown(rows: list[dict]) -> str:
    cols = _leaderboard_columns()

    def _fmt(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(_fmt(row.get(k, "")) for k in cols) + " |" for row in rows]
    return "\n".join([header, sep, *body])
