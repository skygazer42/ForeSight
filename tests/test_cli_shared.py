from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import csv
import pytest
from foresight import cli_shared as _cli_shared


def test_write_output_rejects_directory_target(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(ValueError, match="Output path must be a file"):
        _cli_shared._write_output("payload", output=str(out_dir))


def test_write_output_writes_nested_text_file(tmp_path: Path) -> None:
    out_file = tmp_path / "nested" / "result.json"

    _cli_shared._write_output('{"ok": true}', output=str(out_file))

    assert out_file.read_text(encoding="utf-8") == '{"ok": true}\n'


def test_write_rendered_writes_formatted_text_and_returns_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    written: list[tuple[str, str]] = []

    monkeypatch.setattr(
        _cli_shared,
        "_write_output",
        lambda text, *, output: written.append((text, output)),
    )

    text = _cli_shared._write_rendered(lambda: "joined", output="out.txt")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert text == "joined"
    assert written == [("joined", "out.txt")]


def test_emit_rendered_formats_once_then_prints_and_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[tuple[str, str]] = []
    render_calls = 0

    def _render() -> str:
        nonlocal render_calls
        render_calls += 1
        return "payload"

    monkeypatch.setattr(
        _cli_shared,
        "_print_and_write",
        lambda text, *, output: emitted.append((text, output)) or text,
    )

    _cli_shared._emit_rendered(_render, output="report.txt")

    assert render_calls == 1
    assert emitted == [("payload", "report.txt")]


def test_dataframe_text_formats_csv_without_trailing_newline() -> None:
    calls: list[tuple[str, object]] = []

    class _FakeFrame:
        def to_csv(self, *, index: bool) -> str:
            calls.append(("csv", index))
            return "a\n1\n"

    text = _cli_shared._dataframe_text(_FakeFrame(), fmt="csv")

    assert text == "a\n1"
    assert calls == [("csv", False)]


def test_dataframe_text_formats_json_records_with_iso_dates() -> None:
    calls: list[tuple[str, str, str]] = []

    class _FakeFrame:
        def to_json(self, *, orient: str, date_format: str) -> str:
            calls.append(("json", orient, date_format))
            return '[{"a":1}]'

    text = _cli_shared._dataframe_text(_FakeFrame(), fmt="json")

    assert text == '[{"a":1}]'
    assert calls == [("json", "records", "iso")]


def test_emit_dataframe_uses_dataframe_text_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[tuple[str, str]] = []

    monkeypatch.setattr(
        _cli_shared,
        "_dataframe_text",
        lambda df, *, fmt: '{"ok": true}',
    )
    monkeypatch.setattr(
        _cli_shared,
        "_emit_text",
        lambda text, *, output: emitted.append((text, output)),
    )

    _cli_shared._emit_dataframe(object(), output="frame.json", fmt="json")

    assert emitted == [('{"ok": true}', "frame.json")]


def test_format_rows_formats_json_row_lists() -> None:
    text = _cli_shared._format_rows(
        [{"b": 2, "a": 1}],
        columns=["a", "b"],
        fmt="json",
    )

    assert text == '[{"a": 1, "b": 2}]'


def test_format_rows_dispatches_csv_with_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[dict], list[str] | None]] = []

    def _fake_format_csv(rows: list[dict], *, columns: list[str] | None = None) -> str:
        calls.append((rows, columns))
        return "a,b\n1,2"

    monkeypatch.setattr(_cli_shared, "_format_csv", _fake_format_csv)

    text = _cli_shared._format_rows(
        [{"a": 1, "b": 2}],
        columns=["a", "b"],
        fmt="csv",
    )

    assert text == "a,b\n1,2"
    assert calls == [([{"a": 1, "b": 2}], ["a", "b"])]


def test_format_rows_dispatches_markdown_with_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[dict], list[str] | None]] = []

    def _fake_format_markdown(rows: list[dict], *, columns: list[str] | None = None) -> str:
        calls.append((rows, columns))
        return "| a |"

    monkeypatch.setattr(_cli_shared, "_format_markdown", _fake_format_markdown)

    text = _cli_shared._format_rows(
        [{"a": 1}],
        columns=["a"],
        fmt="md",
    )

    assert text == "| a |"
    assert calls == [([{"a": 1}], ["a"])]


def test_resolved_columns_returns_copy_of_explicit_columns() -> None:
    columns = ["a", "b"]

    resolved = _cli_shared._resolved_columns(columns)

    assert resolved == ["a", "b"]
    assert resolved is not columns


def test_resolved_columns_defaults_to_leaderboard_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_cli_shared, "_leaderboard_columns", lambda: ["x", "y"])

    resolved = _cli_shared._resolved_columns(None)

    assert resolved == ["x", "y"]


def test_parse_param_assignment_splits_item_and_strips_key() -> None:
    key, value = _cli_shared._parse_param_assignment(
        " season_length = 12 ",
        option="--model-param",
        value_spec="key=value",
    )

    assert key == "season_length"
    assert value == " 12 "


def test_parse_param_assignment_reuses_model_param_error_message() -> None:
    with pytest.raises(ValueError, match=r"--model-param must be key=value"):
        _cli_shared._parse_param_assignment(
            "season_length",
            option="--model-param",
            value_spec="key=value",
        )


def test_parse_param_assignment_reuses_grid_param_error_message() -> None:
    with pytest.raises(ValueError, match=r"--grid-param must be key=v1,v2,\.\.\."):
        _cli_shared._parse_param_assignment(
            "=1,2",
            option="--grid-param",
            value_spec="key=v1,v2,...",
        )


def test_json_text_sorts_keys_and_preserves_unicode() -> None:
    text = _cli_shared._json_text({"b": 1, "a": "中"})

    assert text == '{"a": "中", "b": 1}'


def test_markdown_cell_text_formats_supported_value_types() -> None:
    assert _cli_shared._markdown_cell_text(None) == ""
    assert _cli_shared._markdown_cell_text(12.3456789) == "12.3457"
    assert _cli_shared._markdown_cell_text(7) == "7"


def test_row_values_preserves_column_order_and_fills_missing_values() -> None:
    values = _cli_shared._row_values(
        {"b": 2, "a": 1, "extra": 99},
        columns=["a", "missing", "b"],
    )

    assert values == [1, "", 2]


def test_split_csv_items_strips_and_discards_empty_parts() -> None:
    items = _cli_shared._split_csv_items(" a, ,b ,, c ")

    assert items == ["a", "b", "c"]


def test_output_arg_value_coerces_output_attr_to_string() -> None:
    value = _cli_shared._output_arg_value(SimpleNamespace(output=Path("nested") / "report.json"))

    assert value == "nested/report.json"


def test_output_arg_value_defaults_to_empty_string_when_missing() -> None:
    value = _cli_shared._output_arg_value(SimpleNamespace())

    assert value == ""


def test_stripped_arg_value_trims_named_string_argument() -> None:
    value = _cli_shared._stripped_arg_value(
        SimpleNamespace(path_prefix=" /tracking "),
        "path_prefix",
    )

    assert value == "/tracking"


def test_optional_stripped_arg_value_returns_none_for_blank_input() -> None:
    value = _cli_shared._optional_stripped_arg_value(
        SimpleNamespace(path_prefix="   "),
        "path_prefix",
    )

    assert value is None


def test_optional_stripped_arg_value_uses_default_then_trims_it() -> None:
    value = _cli_shared._optional_stripped_arg_value(
        SimpleNamespace(),
        "summary_format",
        default=" json ",
    )

    assert value == "json"


def test_string_arg_value_coerces_named_argument_to_string() -> None:
    value = _cli_shared._string_arg_value(
        SimpleNamespace(horizon=12),
        "horizon",
    )

    assert value == "12"


def test_string_arg_value_uses_default_when_missing() -> None:
    value = _cli_shared._string_arg_value(
        SimpleNamespace(),
        "lags",
        default="5",
    )

    assert value == "5"


def test_parse_cols_arg_uses_named_argument_value() -> None:
    value = _cli_shared._parse_cols_arg(
        SimpleNamespace(columns=" ds , promo "),
        "columns",
    )

    assert value == ("ds", "promo")


def test_parse_cols_arg_uses_default_when_missing() -> None:
    value = _cli_shared._parse_cols_arg(
        SimpleNamespace(),
        "columns",
        default="y",
    )

    assert value == ("y",)


def test_parse_id_cols_arg_uses_named_argument_value() -> None:
    value = _cli_shared._parse_id_cols_arg(
        SimpleNamespace(id_cols="store, dept"),
    )

    assert value == ("store", "dept")


def test_parse_requires_arg_uses_named_argument_value() -> None:
    value = _cli_shared._parse_requires_arg(
        SimpleNamespace(requires="torch,core"),
    )

    assert value == ({"torch"}, True)


def test_list_arg_values_returns_list_copy_for_existing_sequence() -> None:
    raw = ["alpha", "beta"]

    value = _cli_shared._list_arg_values(
        SimpleNamespace(model_param=raw),
        "model_param",
    )

    assert value == ["alpha", "beta"]
    assert value is not raw


def test_list_arg_values_wraps_scalar_values() -> None:
    value = _cli_shared._list_arg_values(
        SimpleNamespace(model_param="lags=5"),
        "model_param",
    )

    assert value == ["lags=5"]


def test_list_arg_values_defaults_to_empty_list_when_missing() -> None:
    value = _cli_shared._list_arg_values(
        SimpleNamespace(),
        "model_param",
    )

    assert value == []


def test_int_arg_value_coerces_named_argument_to_int() -> None:
    value = _cli_shared._int_arg_value(
        SimpleNamespace(horizon="12"),
        "horizon",
    )

    assert value == 12


def test_float_arg_value_coerces_named_argument_to_float() -> None:
    value = _cli_shared._float_arg_value(
        SimpleNamespace(iqr_k="1.5"),
        "iqr_k",
    )

    assert value == 1.5


def test_bool_arg_value_coerces_named_argument_to_bool() -> None:
    value = _cli_shared._bool_arg_value(
        SimpleNamespace(parse_dates=1),
        "parse_dates",
    )

    assert value is True


def test_format_arg_value_coerces_format_attr_to_string() -> None:
    value = _cli_shared._format_arg_value(SimpleNamespace(format="json"))

    assert value == "json"


def test_format_arg_value_normalizes_markdown_alias_when_requested() -> None:
    value = _cli_shared._format_arg_value(
        SimpleNamespace(format="markdown"),
        markdown_alias=True,
    )

    assert value == "md"


def test_format_csv_preserves_column_order_without_dictwriter(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbid_dict_writer(*args: object, **kwargs: object) -> object:
        raise AssertionError("csv.DictWriter should not be used")

    monkeypatch.setattr(csv, "DictWriter", _forbid_dict_writer)

    text = _cli_shared._format_csv(
        [{"b": 2, "a": 1, "extra": 99}, {"a": 3}],
        columns=["a", "b"],
    )

    assert text.splitlines() == [
        "a,b",
        "1,2",
        "3,",
    ]


def test_write_table_formats_and_writes_without_printing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "table.json"

    monkeypatch.setattr(
        _cli_shared,
        "_format_table",
        lambda rows, *, columns, fmt: '{"ok": true}',
    )

    text = _cli_shared._write_table(
        [{"a": 1}],
        columns=["a"],
        output=str(out),
        fmt="json",
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert text == '{"ok": true}'
    assert out.read_text(encoding="utf-8") == '{"ok": true}\n'


def test_emit_table_formats_prints_and_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "table.md"

    monkeypatch.setattr(
        _cli_shared,
        "_format_table",
        lambda rows, *, columns, fmt: "| ok |",
    )

    _cli_shared._emit_table(
        [{"a": 1}],
        columns=["a"],
        output=str(out),
        fmt="md",
    )

    captured = capsys.readouterr()
    assert captured.out == "| ok |\n"
    assert out.read_text(encoding="utf-8") == "| ok |\n"


def test_write_lines_joins_with_trailing_newline(tmp_path: Path) -> None:
    out = tmp_path / "lines.txt"

    text = _cli_shared._write_lines(
        ["first", "second"],
        output=str(out),
    )

    assert text == "first\nsecond"
    assert out.read_text(encoding="utf-8") == "first\nsecond\n"
