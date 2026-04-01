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
