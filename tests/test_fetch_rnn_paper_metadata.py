from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest


def _load_tool():
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(root / "tools" / "fetch_rnn_paper_metadata.py")


def test_resolve_output_path_anchors_safe_filenames_under_docs_dir() -> None:
    mod = _load_tool()
    resolve_output_path = mod["_resolve_output_path"]
    repo_root = Path(__file__).resolve().parents[1]

    resolved = resolve_output_path("custom-metadata.json")

    assert resolved.path == repo_root / "docs" / "custom-metadata.json"


def test_resolve_output_path_rejects_nested_paths() -> None:
    mod = _load_tool()
    resolve_output_path = mod["_resolve_output_path"]

    with pytest.raises(ValueError, match="filename inside docs"):
        resolve_output_path("docs/custom-metadata.json")


def test_resolve_output_path_rejects_repo_escape() -> None:
    mod = _load_tool()
    resolve_output_path = mod["_resolve_output_path"]

    with pytest.raises(ValueError, match="filename inside docs"):
        resolve_output_path("../escaped.json")


def test_main_resolves_output_path_before_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_tool()
    repo_root = Path(__file__).resolve().parents[1]
    captured: dict[str, object] = {}

    def _fake_fetch_all(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setitem(mod["main"].__globals__, "fetch_all", _fake_fetch_all)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_rnn_paper_metadata.py", "--output", "cli-metadata.json"],
    )

    assert mod["main"]() == 0
    assert captured["output_path"].path == repo_root / "docs" / "cli-metadata.json"


def test_write_json_requires_resolved_docs_json_path(tmp_path: Path) -> None:
    mod = _load_tool()
    write_json = mod["_write_json"]

    with pytest.raises(TypeError, match="resolved docs JSON path"):
        write_json(tmp_path / "unsafe.json", {})


def test_fetch_tool_avoids_nested_conditionals_for_metadata_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools" / "fetch_rnn_paper_metadata.py").read_text(encoding="utf-8")

    assert "if isinstance(issued[0][0], int):" not in source
    assert (
        'else ("arxiv" if arxiv0 else ("override" if paper_id in url_overrides else ""))'
        not in source
    )
    assert (
        '"arxiv" if arxiv else ("hint" if hint_title else ("crossref" if crossref else ""))'
        not in source
    )
    assert (
        '"arxiv" if (arxiv and arxiv.doi) else ("crossref" if (crossref and crossref.doi) else "")'
        not in source
    )
