from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest


def _load_tool():
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(root / "tools" / "fetch_rnn_paper_metadata.py")


def test_resolve_output_path_keeps_repo_relative_paths_under_repo_root() -> None:
    mod = _load_tool()
    resolve_output_path = mod["_resolve_output_path"]
    repo_root = Path(__file__).resolve().parents[1]

    resolved = resolve_output_path("docs/custom-metadata.json")

    assert resolved == repo_root / "docs" / "custom-metadata.json"


def test_resolve_output_path_rejects_repo_escape() -> None:
    mod = _load_tool()
    resolve_output_path = mod["_resolve_output_path"]

    with pytest.raises(ValueError, match="inside the repository root"):
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
        ["fetch_rnn_paper_metadata.py", "--output", "docs/cli-metadata.json"],
    )

    assert mod["main"]() == 0
    assert captured["output_path"] == repo_root / "docs" / "cli-metadata.json"
