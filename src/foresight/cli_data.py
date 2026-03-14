from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd

from . import cli_shared as _cli_shared

_OUTPUT_PATH_HELP = "Optional path to write output"
_OUTPUT_CSV_FORMAT_HELP = "Output format (default: csv)"


def register_data_subparsers(sub: Any) -> None:
    _register_datasets_parser(sub)
    _register_data_parser(sub)


def _register_datasets_parser(sub: Any) -> None:
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


def _register_data_parser(sub: Any) -> None:
    data_p = sub.add_parser("data", help="Time-series data utilities")
    data_sub = data_p.add_subparsers(dest="data_command", required=True)

    data_to_long = data_sub.add_parser("to-long", help="Convert an arbitrary CSV into long format")
    data_to_long.add_argument("--path", required=True, help="Path to a CSV file")
    data_to_long.add_argument("--time-col", required=True, help="Time column name")
    data_to_long.add_argument("--y-col", required=True, help="Target column name")
    data_to_long.add_argument(
        "--id-cols",
        type=str,
        default="",
        help="Optional comma-separated id columns for panel data",
    )
    data_to_long.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse time_col with pandas.to_datetime before converting",
    )
    data_to_long.add_argument(
        "--x-cols",
        type=str,
        default="",
        help="Optional comma-separated covariate columns (alias for future_x_cols)",
    )
    data_to_long.add_argument(
        "--historic-x-cols",
        type=str,
        default="",
        help="Optional comma-separated historic covariate columns",
    )
    data_to_long.add_argument(
        "--future-x-cols",
        type=str,
        default="",
        help="Optional comma-separated future covariate columns",
    )
    data_to_long.add_argument(
        "--keepna",
        action="store_true",
        help="Keep NA rows instead of dropping (default: drop NA in ds/y/x cols)",
    )
    data_to_long.add_argument(
        "--prepare",
        action="store_true",
        help="Apply prepare_long_df() after conversion (frequency + missing policies)",
    )
    data_to_long.add_argument(
        "--freq",
        type=str,
        default="",
        help="Optional frequency string for --prepare, e.g. D, W, M",
    )
    data_to_long.add_argument(
        "--strict-freq",
        action="store_true",
        help="Reject irregular timestamps when inferring frequency",
    )
    data_to_long.add_argument(
        "--y-missing",
        type=str,
        default="error",
        help="Missing policy for y when --prepare (error|drop|ffill|zero|interpolate)",
    )
    data_to_long.add_argument(
        "--x-missing",
        type=str,
        default="error",
        help="Missing policy for covariates when --prepare (error|drop|ffill|zero|interpolate)",
    )
    data_to_long.add_argument(
        "--historic-x-missing",
        type=str,
        default="",
        help="Optional missing policy override for historic_x_cols",
    )
    data_to_long.add_argument(
        "--future-x-missing",
        type=str,
        default="",
        help="Optional missing policy override for future_x_cols",
    )
    data_to_long.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    data_to_long.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help=_OUTPUT_CSV_FORMAT_HELP,
    )
    data_to_long.set_defaults(_handler=_cmd_data_to_long)

    data_prepare_long = data_sub.add_parser(
        "prepare-long", help="Regularize an existing long-format CSV (unique_id/ds/y)"
    )
    data_prepare_long.add_argument("--path", required=True, help="Path to a long-format CSV file")
    data_prepare_long.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse ds with pandas.to_datetime before preparing",
    )
    data_prepare_long.add_argument(
        "--freq",
        type=str,
        default="",
        help="Optional frequency string, e.g. D, W, M",
    )
    data_prepare_long.add_argument(
        "--strict-freq",
        action="store_true",
        help="Reject irregular timestamps when inferring frequency",
    )
    data_prepare_long.add_argument(
        "--y-missing",
        type=str,
        default="error",
        help="Missing policy for y (error|drop|ffill|zero|interpolate)",
    )
    data_prepare_long.add_argument(
        "--x-missing",
        type=str,
        default="error",
        help="Missing policy for covariates (error|drop|ffill|zero|interpolate)",
    )
    data_prepare_long.add_argument(
        "--historic-x-cols",
        type=str,
        default="",
        help="Optional comma-separated historic covariate columns",
    )
    data_prepare_long.add_argument(
        "--future-x-cols",
        type=str,
        default="",
        help="Optional comma-separated future covariate columns",
    )
    data_prepare_long.add_argument(
        "--historic-x-missing",
        type=str,
        default="",
        help="Optional missing policy override for historic_x_cols",
    )
    data_prepare_long.add_argument(
        "--future-x-missing",
        type=str,
        default="",
        help="Optional missing policy override for future_x_cols",
    )
    data_prepare_long.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    data_prepare_long.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help=_OUTPUT_CSV_FORMAT_HELP,
    )
    data_prepare_long.set_defaults(_handler=_cmd_data_prepare_long)

    data_infer_freq = data_sub.add_parser(
        "infer-freq", help="Infer a regular datetime frequency from timestamps"
    )
    data_infer_freq.add_argument("--path", required=True, help="Path to a CSV file")
    data_infer_freq.add_argument("--time-col", required=True, help="Time column name")
    data_infer_freq.add_argument(
        "--id-cols",
        type=str,
        default="",
        help="Optional comma-separated id columns for panel data",
    )
    data_infer_freq.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse time_col with pandas.to_datetime before inferring",
    )
    data_infer_freq.add_argument(
        "--strict",
        action="store_true",
        help="Fail on irregular series instead of emitting an empty freq",
    )
    data_infer_freq.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    data_infer_freq.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help=_OUTPUT_CSV_FORMAT_HELP,
    )
    data_infer_freq.set_defaults(_handler=_cmd_data_infer_freq)

    data_splits = data_sub.add_parser("splits", help="Time-series split utilities")
    data_splits_sub = data_splits.add_subparsers(dest="splits_command", required=True)

    data_splits_rolling = data_splits_sub.add_parser(
        "rolling-origin", help="Print rolling-origin split indices"
    )
    data_splits_rolling.add_argument("--n-obs", type=int, required=True, help="Series length")
    data_splits_rolling.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    data_splits_rolling.add_argument(
        "--step-size", type=int, default=1, help="Step size between windows"
    )
    data_splits_rolling.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum train size before first window",
    )
    data_splits_rolling.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help="Optional rolling train window size (default: expanding window)",
    )
    data_splits_rolling.add_argument(
        "--output",
        type=str,
        default="",
        help=_OUTPUT_PATH_HELP,
    )
    data_splits_rolling.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help=_OUTPUT_CSV_FORMAT_HELP,
    )
    data_splits_rolling.set_defaults(_handler=_cmd_data_splits_rolling_origin)


def _cmd_datasets_list(args: argparse.Namespace) -> int:
    from .datasets.registry import (
        describe_dataset,
        get_dataset_spec,
        list_datasets,
        resolve_dataset_path,
    )

    for key in list_datasets():
        if args.with_path:
            try:
                path = resolve_dataset_path(key, data_dir=str(args.data_dir))
                path_s = path.as_posix()
            except FileNotFoundError:
                path_s = get_dataset_spec(key).rel_path.as_posix()
            print(f"{key}\t{path_s}\t{describe_dataset(key)}")
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


def _cmd_data_to_long(args: argparse.Namespace) -> int:
    from .data.format import to_long
    from .io import ensure_datetime, load_csv, parse_cols, parse_id_cols

    df = load_csv(str(args.path))
    if bool(args.parse_dates):
        ensure_datetime(df, str(args.time_col))

    id_cols = parse_id_cols(str(args.id_cols))
    x_cols = parse_cols(str(args.x_cols))
    historic_x_cols = parse_cols(str(args.historic_x_cols))
    future_x_cols = parse_cols(str(args.future_x_cols))

    freq = str(getattr(args, "freq", "")).strip() or None
    historic_x_missing = str(getattr(args, "historic_x_missing", "")).strip() or None
    future_x_missing = str(getattr(args, "future_x_missing", "")).strip() or None

    out = to_long(
        df,
        time_col=str(args.time_col),
        y_col=str(args.y_col),
        id_cols=id_cols,
        x_cols=x_cols,
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
        dropna=not bool(getattr(args, "keepna", False)),
        prepare=bool(getattr(args, "prepare", False)),
        freq=freq,
        strict_freq=bool(getattr(args, "strict_freq", False)),
        y_missing=str(getattr(args, "y_missing", "error")),
        x_missing=str(getattr(args, "x_missing", "error")),
        historic_x_missing=historic_x_missing,
        future_x_missing=future_x_missing,
    )

    _cli_shared._emit_dataframe(out, output=str(getattr(args, "output", "")), fmt=str(args.format))
    return 0


def _cmd_data_prepare_long(args: argparse.Namespace) -> int:
    from .data.prep import prepare_long_df
    from .io import ensure_datetime, load_csv, parse_cols

    df = load_csv(str(args.path))
    if bool(args.parse_dates):
        ensure_datetime(df, "ds")

    freq = str(getattr(args, "freq", "")).strip() or None
    historic_x_missing = str(getattr(args, "historic_x_missing", "")).strip() or None
    future_x_missing = str(getattr(args, "future_x_missing", "")).strip() or None

    historic_x_cols = parse_cols(str(getattr(args, "historic_x_cols", "")))
    future_x_cols = parse_cols(str(getattr(args, "future_x_cols", "")))

    out = prepare_long_df(
        df,
        freq=freq,
        strict_freq=bool(getattr(args, "strict_freq", False)),
        y_missing=str(getattr(args, "y_missing", "error")),
        x_missing=str(getattr(args, "x_missing", "error")),
        historic_x_cols=tuple(historic_x_cols),
        future_x_cols=tuple(future_x_cols),
        historic_x_missing=historic_x_missing,
        future_x_missing=future_x_missing,
    )

    _cli_shared._emit_dataframe(out, output=str(getattr(args, "output", "")), fmt=str(args.format))
    return 0


def _cmd_data_infer_freq(args: argparse.Namespace) -> int:
    from .data.prep import infer_series_frequency
    from .io import ensure_datetime, load_csv, parse_id_cols

    df = load_csv(str(args.path))
    time_col = str(args.time_col)
    if bool(args.parse_dates):
        ensure_datetime(df, time_col)

    id_cols = parse_id_cols(str(getattr(args, "id_cols", "")))

    if not id_cols:
        uid = pd.Series(["series=0"] * len(df), index=df.index, dtype="string")
    else:
        parts: list[pd.Series] = []
        for col in id_cols:
            if col not in df.columns:
                raise KeyError(f"id col not found: {col!r}")
            parts.append(col + "=" + df[col].astype("string"))
        uid = parts[0]
        for part in parts[1:]:
            uid = uid + "|" + part

    strict = bool(getattr(args, "strict", False))
    df2 = df.copy()
    df2["unique_id"] = uid.astype("string")

    rows: list[dict[str, Any]] = []
    for unique_id, group in df2.groupby("unique_id", sort=True):
        ds = group[time_col]
        has_duplicates = bool(ds.duplicated().any())
        error = ""
        freq = None
        try:
            if has_duplicates:
                raise ValueError("ds contains duplicates")
            freq = infer_series_frequency(ds, strict=bool(strict))
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"

        rows.append(
            {
                "unique_id": str(unique_id),
                "freq": freq,
                "n_obs": int(len(group)),
                "n_unique_ds": int(pd.Index(ds).nunique()),
                "has_duplicates": has_duplicates,
                "error": error,
            }
        )

    out = pd.DataFrame(rows)
    _cli_shared._emit_dataframe(out, output=str(getattr(args, "output", "")), fmt=str(args.format))
    return 0


def _cmd_data_splits_rolling_origin(args: argparse.Namespace) -> int:
    from .splits import rolling_origin_splits

    rows: list[dict[str, Any]] = []
    for i, split in enumerate(
        rolling_origin_splits(
            int(args.n_obs),
            horizon=int(args.horizon),
            step_size=int(args.step_size),
            min_train_size=int(args.min_train_size),
            max_train_size=getattr(args, "max_train_size", None),
        )
    ):
        rows.append(
            {
                "window": int(i),
                "train_start": int(split.train_start),
                "train_end": int(split.train_end),
                "test_start": int(split.test_start),
                "test_end": int(split.test_end),
            }
        )

    out = pd.DataFrame(rows)
    _cli_shared._emit_dataframe(out, output=str(getattr(args, "output", "")), fmt=str(args.format))
    return 0
