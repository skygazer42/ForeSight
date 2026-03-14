from __future__ import annotations

from pathlib import Path

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
