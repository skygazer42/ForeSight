from __future__ import annotations

from pathlib import Path

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
