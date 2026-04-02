from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

_CORE_REQUIRES_ALIASES = {"core", "none", "empty", "no", "norequires", "no-requires"}


def _leaderboard_columns() -> list[str]:
    # Stable output makes diffs and automation easier.
    return [
        "model",
        "task_group",
        "backend_family",
        "status",
        "skip_reason",
        "error_type",
        "error_message",
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


def _sanitize_tsv_cell(value: object) -> str:
    """
    TSV output is used for CLI piping. Keep it single-line and tab-safe.
    """

    s = str(value) if value is not None else ""
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def _parse_requires_filter(raw: str) -> tuple[set[str], bool]:
    """
    Parse `--requires` / `--exclude-requires` values.

    - Tokens are comma-separated.
    - Special tokens like "core"/"none" refer to models with no optional requires.
    Returns: (requires_set, include_core_flag)
    """

    items = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    tokens = set(items)
    include_core = bool(tokens.intersection(_CORE_REQUIRES_ALIASES))
    tokens.difference_update(_CORE_REQUIRES_ALIASES)
    return (tokens, include_core)


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


def _parse_param_assignment(
    item: object,
    *,
    option: str,
    value_spec: str,
) -> tuple[str, str]:
    raw = str(item)
    message = f"{option} must be {value_spec}, got: {item!r}"
    if "=" not in raw:
        raise ValueError(message)
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(message)
    return key, value


def _parse_model_params(items: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        key, value = _parse_param_assignment(
            item,
            option="--model-param",
            value_spec="key=value",
        )
        params[key] = _coerce_model_param_value(value)
    return params


def _parse_grid_params(items: list[str]) -> dict[str, tuple[Any, ...]]:
    params: dict[str, tuple[Any, ...]] = {}
    for item in items:
        key, value = _parse_param_assignment(
            item,
            option="--grid-param",
            value_spec="key=v1,v2,...",
        )
        parsed = _coerce_model_param_value(value)
        if isinstance(parsed, tuple):
            values = parsed
        else:
            values = (parsed,)
        if not values:
            raise ValueError(f"--grid-param must include at least one value, got: {item!r}")
        params[key] = tuple(values)
    return params


def _write_output(text: str, *, output: str) -> None:
    if not output:
        return
    out_path = Path(output).expanduser().resolve(strict=False)
    if out_path.exists() and out_path.is_dir():
        raise ValueError("Output path must be a file, got directory")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "" if not text else text.rstrip("\n") + "\n"
    # CLI callers explicitly choose the output path.
    out_path.write_text(payload, encoding="utf-8")  # NOSONAR


def _write_lines(lines: list[str], *, output: str) -> str:
    return _write_rendered(lambda: _lines_text(lines), output=output)


def _lines_text(lines: list[str]) -> str:
    return "\n".join(str(line) for line in lines)


def _write_rendered(render: Callable[[], str], *, output: str) -> str:
    text = render()
    _write_output(text, output=output)
    return text


def _print_and_write(text: str, *, output: str) -> str:
    print(text)
    _write_output(text, output=output)
    return text


def _emit_rendered(render: Callable[[], str], *, output: str) -> None:
    _print_and_write(render(), output=output)


def _emit_text(text: str, *, output: str) -> None:
    _emit_rendered(lambda: text, output=output)


def _emit_dataframe(df: Any, *, output: str, fmt: str) -> None:
    _emit_text(_dataframe_text(df, fmt=fmt), output=output)


def _dataframe_text(df: Any, *, fmt: str) -> str:
    if fmt == "csv":
        return df.to_csv(index=False).rstrip("\n")
    if fmt == "json":
        return df.to_json(orient="records", date_format="iso")
    raise ValueError(f"Unknown dataframe format: {fmt!r}")


def _emit(payload: object, *, output: str, fmt: str) -> None:
    _emit_rendered(lambda: _format_payload(payload, fmt=fmt), output=output)


def _coerce_row_payload(payload: object, *, fmt: str) -> list[dict]:
    rows: list[dict]
    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise TypeError(f"{fmt} format expects a dict row or list of dict rows")
    return rows


def _format_payload(payload: object, *, fmt: str) -> str:
    if fmt == "json":
        return _json_text(payload)
    return _format_rows(_coerce_row_payload(payload, fmt=fmt), fmt=fmt)


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _resolved_columns(columns: list[str] | None) -> list[str]:
    return list(columns) if columns is not None else _leaderboard_columns()


def _format_csv(rows: list[dict], *, columns: list[str] | None = None) -> str:
    cols = _resolved_columns(columns)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols)
    for row in rows:
        writer.writerow(_row_values(row, columns=cols))
    return buf.getvalue().rstrip("\n")


def _format_markdown(rows: list[dict], *, columns: list[str] | None = None) -> str:
    cols = _resolved_columns(columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(_markdown_cell_text(value) for value in _row_values(row, columns=cols)) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def _row_values(row: dict[str, Any], *, columns: list[str]) -> list[object]:
    return [row.get(column, "") for column in columns]


def _markdown_cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _emit_table(rows: list[dict[str, Any]], *, columns: list[str], output: str, fmt: str) -> None:
    _emit_rendered(lambda: _table_text(rows, columns=columns, fmt=fmt), output=output)


def _write_table(rows: list[dict[str, Any]], *, columns: list[str], output: str, fmt: str) -> str:
    return _write_rendered(
        lambda: _table_text(rows, columns=columns, fmt=fmt),
        output=output,
    )


def _table_text(rows: list[dict[str, Any]], *, columns: list[str], fmt: str) -> str:
    return _format_table(rows, columns=columns, fmt=fmt)


def _format_table(rows: list[dict[str, Any]], *, columns: list[str], fmt: str) -> str:
    return _format_rows(rows, columns=columns, fmt=fmt)


def _format_rows(rows: list[dict], *, columns: list[str] | None = None, fmt: str) -> str:
    if fmt == "json":
        return _json_text(rows)
    if fmt == "csv":
        return _format_csv(rows, columns=columns)
    if fmt == "md":
        return _format_markdown(rows, columns=columns)
    raise ValueError(f"Unknown format: {fmt!r}")
