from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd

from . import cli_shared as _cli_shared

_OUTPUT_PATH_HELP = "Optional path to write output"
_OUTPUT_CSV_FORMAT_HELP = "Output format (default: csv)"
_CSV_PATH_HELP = "Path to a CSV file"
_LONG_CSV_PATH_HELP = "Path to a long-format CSV file"
_STRICT_FREQ_HELP = "Reject irregular timestamps when inferring frequency"


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
    data_to_long.add_argument("--path", required=True, help=_CSV_PATH_HELP)
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
        help=_STRICT_FREQ_HELP,
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
    data_prepare_long.add_argument("--path", required=True, help=_LONG_CSV_PATH_HELP)
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
        help=_STRICT_FREQ_HELP,
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

    data_align_long = data_sub.add_parser(
        "align-long", help="Align an existing long-format CSV to a regular frequency"
    )
    data_align_long.add_argument("--path", required=True, help=_LONG_CSV_PATH_HELP)
    data_align_long.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse ds with pandas.to_datetime before aligning",
    )
    data_align_long.add_argument(
        "--freq",
        type=str,
        default="",
        help="Optional frequency string, e.g. D, W, M",
    )
    data_align_long.add_argument(
        "--agg",
        type=str,
        default="last",
        help="Aggregation for duplicate timestamps / resampling buckets",
    )
    data_align_long.add_argument(
        "--columns",
        type=str,
        default="",
        help="Optional comma-separated numeric columns to align (default: all numeric value columns)",
    )
    data_align_long.add_argument(
        "--strict-freq",
        action="store_true",
        help=_STRICT_FREQ_HELP,
    )
    data_align_long.add_argument("--output", type=str, default="", help=_OUTPUT_PATH_HELP)
    data_align_long.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help=_OUTPUT_CSV_FORMAT_HELP,
    )
    data_align_long.set_defaults(_handler=_cmd_data_align_long)

    data_clip_outliers = data_sub.add_parser(
        "clip-outliers", help="Clip per-series outliers in a long-format CSV"
    )
    data_clip_outliers.add_argument("--path", required=True, help=_LONG_CSV_PATH_HELP)
    data_clip_outliers.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse ds with pandas.to_datetime before clipping",
    )
    data_clip_outliers.add_argument(
        "--method",
        type=str,
        default="iqr",
        help="Clipping method: iqr or zscore",
    )
    data_clip_outliers.add_argument(
        "--columns",
        type=str,
        default="y",
        help="Comma-separated numeric columns to clip (default: y)",
    )
    data_clip_outliers.add_argument(
        "--iqr-k",
        type=float,
        default=1.5,
        help="IQR multiplier for method=iqr",
    )
    data_clip_outliers.add_argument(
        "--zmax",
        type=float,
        default=3.0,
        help="Max z-score for method=zscore",
    )
    data_clip_outliers.add_argument("--output", type=str, default="", help=_OUTPUT_PATH_HELP)
    data_clip_outliers.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help=_OUTPUT_CSV_FORMAT_HELP,
    )
    data_clip_outliers.set_defaults(_handler=_cmd_data_clip_outliers)

    data_calendar_features = data_sub.add_parser(
        "calendar-features", help="Append calendar/time features to a long-format CSV"
    )
    data_calendar_features.add_argument("--path", required=True, help=_LONG_CSV_PATH_HELP)
    data_calendar_features.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse ds with pandas.to_datetime before generating features",
    )
    data_calendar_features.add_argument(
        "--prefix",
        type=str,
        default="cal_",
        help="Prefix for generated feature columns",
    )
    data_calendar_features.add_argument("--output", type=str, default="", help=_OUTPUT_PATH_HELP)
    data_calendar_features.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help=_OUTPUT_CSV_FORMAT_HELP,
    )
    data_calendar_features.set_defaults(_handler=_cmd_data_calendar_features)

    data_make_supervised = data_sub.add_parser(
        "make-supervised", help="Build a supervised training frame from long or wide CSV data"
    )
    data_make_supervised.add_argument("--path", required=True, help=_CSV_PATH_HELP)
    data_make_supervised.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse ds/ds_col with pandas.to_datetime before building the frame",
    )
    data_make_supervised.add_argument(
        "--input-format",
        choices=["auto", "long", "wide"],
        default="auto",
        help="Input format hint (default: auto-detect)",
    )
    data_make_supervised.add_argument(
        "--ds-col",
        type=str,
        default="ds",
        help="Timestamp column for wide input",
    )
    data_make_supervised.add_argument(
        "--target-cols",
        type=str,
        default="",
        help="Optional comma-separated target columns for wide input",
    )
    data_make_supervised.add_argument(
        "--lags", type=str, default="5", help="Lag window or lag list"
    )
    data_make_supervised.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    data_make_supervised.add_argument(
        "--x-cols",
        type=str,
        default="",
        help="Optional comma-separated numeric feature columns for long input",
    )
    data_make_supervised.add_argument(
        "--roll-windows",
        type=str,
        default="",
        help="Optional comma-separated rolling windows for lag-derived features",
    )
    data_make_supervised.add_argument(
        "--roll-stats",
        type=str,
        default="",
        help="Optional comma-separated lag-derived stats",
    )
    data_make_supervised.add_argument(
        "--diff-lags",
        type=str,
        default="",
        help="Optional comma-separated diff lag specs",
    )
    data_make_supervised.add_argument(
        "--seasonal-lags",
        type=str,
        default="",
        help="Optional comma-separated seasonal lag periods",
    )
    data_make_supervised.add_argument(
        "--seasonal-diff-lags",
        type=str,
        default="",
        help="Optional comma-separated seasonal diff lag periods",
    )
    data_make_supervised.add_argument(
        "--fourier-periods",
        type=str,
        default="",
        help="Optional comma-separated Fourier periods",
    )
    data_make_supervised.add_argument(
        "--fourier-orders",
        type=str,
        default="2",
        help="Fourier order or comma-separated order list",
    )
    data_make_supervised.add_argument(
        "--add-time-features",
        action="store_true",
        help="Append dependency-free time features from ds",
    )
    data_make_supervised.add_argument("--output", type=str, default="", help=_OUTPUT_PATH_HELP)
    data_make_supervised.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help=_OUTPUT_CSV_FORMAT_HELP,
    )
    data_make_supervised.set_defaults(_handler=_cmd_data_make_supervised)

    data_infer_freq = data_sub.add_parser(
        "infer-freq", help="Infer a regular datetime frequency from timestamps"
    )
    data_infer_freq.add_argument("--path", required=True, help=_CSV_PATH_HELP)
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
                path = resolve_dataset_path(
                    key,
                    data_dir=_cli_shared._string_arg_value(args, "data_dir"),
                )
                path_s = path.as_posix()
            except FileNotFoundError:
                path_s = get_dataset_spec(key).rel_path.as_posix()
            print(f"{key}\t{path_s}\t{describe_dataset(key)}")
        else:
            print(f"{key}\t{describe_dataset(key)}")
    return 0


def _cmd_datasets_preview(args: argparse.Namespace) -> int:
    from .datasets.loaders import load_dataset

    df = load_dataset(
        _cli_shared._string_arg_value(args, "key"),
        nrows=int(args.nrows),
        data_dir=_cli_shared._string_arg_value(args, "data_dir"),
    )
    print(df.head(int(args.nrows)).to_string(index=False))
    return 0


def _cmd_datasets_path(args: argparse.Namespace) -> int:
    from .datasets.registry import resolve_dataset_path

    path = resolve_dataset_path(
        _cli_shared._string_arg_value(args, "key"),
        data_dir=_cli_shared._string_arg_value(args, "data_dir"),
    )
    print(path.as_posix())
    return 0


def _validate_dataset_frame(df: pd.DataFrame, *, expected_columns: set[str]) -> None:
    if len(df) <= 0:
        raise ValueError("loaded 0 rows")
    missing = sorted(expected_columns.difference(df.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _validate_dataset_parse_dates(df: pd.DataFrame, *, parse_dates: tuple[str, ...]) -> None:
    for col in parse_dates:
        if col not in df.columns:
            raise ValueError(f"missing parse_dates column: {col!r}")
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise ValueError(f"parse_dates column is not datetime: {col!r}")
        if df[col].isna().any():
            raise ValueError(f"parse_dates column contains NaT/NA values: {col!r}")


def _validate_dataset_time_contracts(df: pd.DataFrame, *, spec: Any) -> None:
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


def _cmd_datasets_validate(args: argparse.Namespace) -> int:
    from .datasets.loaders import load_dataset
    from .datasets.registry import get_dataset_spec, list_datasets

    nrows = int(args.nrows)
    dataset = _cli_shared._string_arg_value(args, "dataset")
    keys = [dataset] if dataset else list_datasets()
    failures = 0
    for key in keys:
        try:
            spec = get_dataset_spec(key)
            df = load_dataset(
                key,
                nrows=nrows,
                data_dir=_cli_shared._string_arg_value(args, "data_dir"),
            )
            _validate_dataset_frame(df, expected_columns=set(spec.expected_columns))
            _validate_dataset_parse_dates(df, parse_dates=tuple(spec.parse_dates))
            if bool(args.check_time):
                _validate_dataset_time_contracts(df, spec=spec)
            print(f"OK {key} rows={len(df)} cols={len(df.columns)}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {key}: {type(e).__name__}: {e}", file=sys.stderr)

    return 1 if failures else 0


def _cmd_data_to_long(args: argparse.Namespace) -> int:
    from .data.format import to_long
    from .io import ensure_datetime, load_csv

    df = load_csv(_cli_shared._string_arg_value(args, "path"))
    if bool(args.parse_dates):
        ensure_datetime(df, _cli_shared._string_arg_value(args, "time_col"))

    id_cols = _cli_shared._parse_id_cols_arg(args)
    x_cols = _cli_shared._parse_cols_arg(args, "x_cols")
    historic_x_cols = _cli_shared._parse_cols_arg(args, "historic_x_cols")
    future_x_cols = _cli_shared._parse_cols_arg(args, "future_x_cols")

    freq = _cli_shared._optional_stripped_arg_value(args, "freq")
    historic_x_missing = _cli_shared._optional_stripped_arg_value(args, "historic_x_missing")
    future_x_missing = _cli_shared._optional_stripped_arg_value(args, "future_x_missing")

    out = to_long(
        df,
        time_col=_cli_shared._string_arg_value(args, "time_col"),
        y_col=_cli_shared._string_arg_value(args, "y_col"),
        id_cols=id_cols,
        x_cols=x_cols,
        historic_x_cols=historic_x_cols,
        future_x_cols=future_x_cols,
        dropna=not _cli_shared._bool_arg_value(args, "keepna"),
        prepare=_cli_shared._bool_arg_value(args, "prepare"),
        freq=freq,
        strict_freq=_cli_shared._bool_arg_value(args, "strict_freq"),
        y_missing=_cli_shared._string_arg_value(args, "y_missing", default="error"),
        x_missing=_cli_shared._string_arg_value(args, "x_missing", default="error"),
        historic_x_missing=historic_x_missing,
        future_x_missing=future_x_missing,
    )

    _cli_shared._emit_dataframe(
        out,
        output=_cli_shared._output_arg_value(args),
        fmt=_cli_shared._format_arg_value(args),
    )
    return 0


def _cmd_data_prepare_long(args: argparse.Namespace) -> int:
    from .data.prep import prepare_long_df
    from .io import ensure_datetime, load_csv

    df = load_csv(_cli_shared._string_arg_value(args, "path"))
    if bool(args.parse_dates):
        ensure_datetime(df, "ds")

    freq = _cli_shared._optional_stripped_arg_value(args, "freq")
    historic_x_missing = _cli_shared._optional_stripped_arg_value(args, "historic_x_missing")
    future_x_missing = _cli_shared._optional_stripped_arg_value(args, "future_x_missing")

    historic_x_cols = _cli_shared._parse_cols_arg(args, "historic_x_cols")
    future_x_cols = _cli_shared._parse_cols_arg(args, "future_x_cols")

    out = prepare_long_df(
        df,
        freq=freq,
        strict_freq=_cli_shared._bool_arg_value(args, "strict_freq"),
        y_missing=_cli_shared._string_arg_value(args, "y_missing", default="error"),
        x_missing=_cli_shared._string_arg_value(args, "x_missing", default="error"),
        historic_x_cols=tuple(historic_x_cols),
        future_x_cols=tuple(future_x_cols),
        historic_x_missing=historic_x_missing,
        future_x_missing=future_x_missing,
    )

    _cli_shared._emit_dataframe(
        out,
        output=_cli_shared._output_arg_value(args),
        fmt=_cli_shared._format_arg_value(args),
    )
    return 0


def _cmd_data_align_long(args: argparse.Namespace) -> int:
    from .data.workflows import align_long_df
    from .io import ensure_datetime, load_csv

    df = load_csv(_cli_shared._string_arg_value(args, "path"))
    if bool(args.parse_dates):
        ensure_datetime(df, "ds")

    freq = _cli_shared._optional_stripped_arg_value(args, "freq")
    columns = _cli_shared._parse_cols_arg(args, "columns") or None
    out = align_long_df(
        df,
        freq=freq,
        agg=_cli_shared._string_arg_value(args, "agg", default="last"),
        columns=columns,
        strict_freq=_cli_shared._bool_arg_value(args, "strict_freq"),
    )
    _cli_shared._emit_dataframe(
        out,
        output=_cli_shared._output_arg_value(args),
        fmt=_cli_shared._format_arg_value(args),
    )
    return 0


def _cmd_data_clip_outliers(args: argparse.Namespace) -> int:
    from .data.workflows import clip_long_df_outliers
    from .io import ensure_datetime, load_csv

    df = load_csv(_cli_shared._string_arg_value(args, "path"))
    if bool(args.parse_dates):
        ensure_datetime(df, "ds")

    out = clip_long_df_outliers(
        df,
        method=_cli_shared._string_arg_value(args, "method", default="iqr"),
        columns=_cli_shared._parse_cols_arg(args, "columns", default="y"),
        iqr_k=_cli_shared._float_arg_value(args, "iqr_k", default=1.5),
        zmax=_cli_shared._float_arg_value(args, "zmax", default=3.0),
    )
    _cli_shared._emit_dataframe(
        out,
        output=_cli_shared._output_arg_value(args),
        fmt=_cli_shared._format_arg_value(args),
    )
    return 0


def _cmd_data_calendar_features(args: argparse.Namespace) -> int:
    from .data.workflows import enrich_long_df_calendar
    from .io import ensure_datetime, load_csv

    df = load_csv(_cli_shared._string_arg_value(args, "path"))
    if bool(args.parse_dates):
        ensure_datetime(df, "ds")

    out = enrich_long_df_calendar(
        df,
        prefix=_cli_shared._string_arg_value(args, "prefix", default="cal_"),
    )
    _cli_shared._emit_dataframe(
        out,
        output=_cli_shared._output_arg_value(args),
        fmt=_cli_shared._format_arg_value(args),
    )
    return 0


def _cmd_data_make_supervised(args: argparse.Namespace) -> int:
    from .data.workflows import make_supervised_frame
    from .io import ensure_datetime, load_csv

    df = load_csv(_cli_shared._string_arg_value(args, "path"))
    ds_col = _cli_shared._string_arg_value(args, "ds_col", default="ds")
    if bool(args.parse_dates):
        if "ds" in df.columns:
            ensure_datetime(df, "ds")
        elif ds_col in df.columns:
            ensure_datetime(df, ds_col)

    out = make_supervised_frame(
        df,
        input_format=_cli_shared._string_arg_value(args, "input_format", default="auto"),
        ds_col=ds_col,
        target_cols=_cli_shared._parse_cols_arg(args, "target_cols"),
        lags=_cli_shared._string_arg_value(args, "lags", default="5"),
        horizon=_cli_shared._int_arg_value(args, "horizon", default=1),
        x_cols=_cli_shared._parse_cols_arg(args, "x_cols"),
        roll_windows=_cli_shared._string_arg_value(args, "roll_windows"),
        roll_stats=_cli_shared._string_arg_value(args, "roll_stats"),
        diff_lags=_cli_shared._string_arg_value(args, "diff_lags"),
        seasonal_lags=_cli_shared._string_arg_value(args, "seasonal_lags"),
        seasonal_diff_lags=_cli_shared._string_arg_value(args, "seasonal_diff_lags"),
        fourier_periods=_cli_shared._string_arg_value(args, "fourier_periods"),
        fourier_orders=_cli_shared._string_arg_value(args, "fourier_orders", default="2"),
        add_time_features=_cli_shared._bool_arg_value(args, "add_time_features"),
    )
    _cli_shared._emit_dataframe(
        out,
        output=_cli_shared._output_arg_value(args),
        fmt=_cli_shared._format_arg_value(args),
    )
    return 0


def _cmd_data_infer_freq(args: argparse.Namespace) -> int:
    from .data.prep import infer_series_frequency
    from .io import ensure_datetime, load_csv

    df = load_csv(_cli_shared._string_arg_value(args, "path"))
    time_col = _cli_shared._string_arg_value(args, "time_col")
    if bool(args.parse_dates):
        ensure_datetime(df, time_col)

    id_cols = _cli_shared._parse_id_cols_arg(args)

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

    strict = _cli_shared._bool_arg_value(args, "strict")
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
    _cli_shared._emit_dataframe(
        out,
        output=_cli_shared._output_arg_value(args),
        fmt=_cli_shared._format_arg_value(args),
    )
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
    _cli_shared._emit_dataframe(
        out,
        output=_cli_shared._output_arg_value(args),
        fmt=_cli_shared._format_arg_value(args),
    )
    return 0
