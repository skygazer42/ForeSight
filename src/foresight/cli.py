from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path


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

    datasets = sub.add_parser("datasets", help="Dataset utilities")
    datasets_sub = datasets.add_subparsers(dest="datasets_command", required=True)

    datasets_list = datasets_sub.add_parser("list", help="List available datasets")
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
        "--output",
        type=str,
        default="",
        help="Optional path to write JSON metrics",
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
        "--season-length",
        type=int,
        required=True,
        help="Season length for seasonal naive",
    )
    eval_seasonal_naive.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write JSON metrics",
    )
    eval_seasonal_naive.set_defaults(_handler=_cmd_eval_seasonal_naive)

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


def _cmd_datasets_list(_args: argparse.Namespace) -> int:
    from .datasets.registry import describe_dataset, list_datasets

    for key in list_datasets():
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
        data_dir=str(args.data_dir),
    )
    _emit(payload, output=args.output, fmt="json")

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
        data_dir=str(args.data_dir),
    )
    _emit(payload, output=args.output, fmt="json")
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
            data_dir=str(args.data_dir),
        ),
        eval_seasonal_naive(
            dataset=str(args.dataset),
            y_col=str(args.y_col),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            season_length=int(args.season_length),
            data_dir=str(args.data_dir),
        ),
    ]
    _emit(rows, output=args.output, fmt=str(args.format))
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
        if not isinstance(payload, list):
            raise TypeError("csv format expects a list of dict rows")
        return _format_csv(payload)
    if fmt == "md":
        if not isinstance(payload, list):
            raise TypeError("md format expects a list of dict rows")
        return _format_markdown(payload)
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
        "season_length",
        "n_windows",
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
