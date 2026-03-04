from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    assert src_dir.exists()

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(src_dir), env.get("PYTHONPATH", "")]).strip(os.pathsep)

    proc = subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout is not None
    return proc.stdout.strip()


def test_cli_papers_list_contains_elman_srn() -> None:
    out = _run_cli("papers", "list")
    assert "elman-srn" in out


def test_cli_papers_info_outputs_json() -> None:
    out = _run_cli("papers", "info", "elman-srn")
    payload = json.loads(out)
    assert payload["paper_id"] == "elman-srn"
    assert payload["year"] == 1990
    assert str(payload.get("title", "")).strip()
    assert str(payload.get("url", "")).startswith("https://")


def test_cli_models_search_finds_rnnpaper_elman() -> None:
    out = _run_cli("models", "search", "elman srn", "--format", "json", "--limit", "10")
    rows = json.loads(out)
    assert isinstance(rows, list)
    keys = {r.get("key") for r in rows if isinstance(r, dict)}
    assert "torch-rnnpaper-elman-srn-direct" in keys


def test_cli_papers_models_includes_rnnpaper_base_match() -> None:
    out = _run_cli("papers", "models", "elman-srn", "--format", "json", "--role", "base")
    rows = json.loads(out)
    assert isinstance(rows, list)
    pairs = {(r.get("key"), r.get("role")) for r in rows if isinstance(r, dict)}
    assert ("torch-rnnpaper-elman-srn-direct", "base") in pairs


def test_cli_papers_models_includes_rnnzoo_wrapper_match() -> None:
    out = _run_cli(
        "papers",
        "models",
        "bahdanau-attention",
        "--format",
        "json",
        "--role",
        "wrapper",
    )
    rows = json.loads(out)
    assert isinstance(rows, list)
    pairs = {(r.get("key"), r.get("role")) for r in rows if isinstance(r, dict)}
    assert ("torch-rnnzoo-clockwork-attn-direct", "wrapper") in pairs


def test_cli_models_search_requires_filter_is_applied() -> None:
    out = _run_cli(
        "models", "search", "arima", "--format", "json", "--requires", "stats", "--limit", "20"
    )
    rows = json.loads(out)
    assert isinstance(rows, list)
    assert rows
    assert all("stats" in str(r.get("requires", "")) for r in rows if isinstance(r, dict))
