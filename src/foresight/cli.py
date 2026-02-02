from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="foresight",
        description="ForeSight: time-series forecasting models and utilities.",
    )
    p.add_argument("--version", action="store_true", help="Print version and exit.")

    sub = p.add_subparsers(dest="command")

    datasets = sub.add_parser("datasets", help="Dataset utilities")
    datasets_sub = datasets.add_subparsers(dest="datasets_command", required=True)

    datasets_list = datasets_sub.add_parser("list", help="List available datasets")
    datasets_list.set_defaults(_handler=_cmd_datasets_list)

    datasets_preview = datasets_sub.add_parser("preview", help="Preview a dataset (head)")
    datasets_preview.add_argument("key", help="Dataset key (see: `foresight datasets list`)")
    datasets_preview.add_argument("--nrows", type=int, default=20, help="Number of rows to preview")
    datasets_preview.set_defaults(_handler=_cmd_datasets_preview)

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

    return int(handler(args))


def _cmd_datasets_list(_args: argparse.Namespace) -> int:
    from .datasets.registry import describe_dataset, list_datasets

    for key in list_datasets():
        print(f"{key}\t{describe_dataset(key)}")
    return 0


def _cmd_datasets_preview(args: argparse.Namespace) -> int:
    from .datasets.loaders import load_dataset

    df = load_dataset(args.key, nrows=int(args.nrows))
    print(df.head(int(args.nrows)).to_string(index=False))
    return 0


def _cmd_eval_naive_last(args: argparse.Namespace) -> int:
    from .eval import eval_naive_last

    payload = eval_naive_last(
        dataset=str(args.dataset),
        y_col=str(args.y_col),
        horizon=int(args.horizon),
        step=int(args.step),
        min_train_size=int(args.min_train_size),
    )
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    print(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    return 0
