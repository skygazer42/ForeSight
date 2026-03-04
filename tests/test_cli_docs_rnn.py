from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )


def test_cli_docs_rnn_writes_docs(tmp_path: Path) -> None:
    out_dir = tmp_path / "docs_out"
    proc = _run_cli("docs", "rnn", "--output-dir", str(out_dir))
    assert proc.returncode == 0

    paper = out_dir / "rnn_paper_zoo.md"
    zoo = out_dir / "rnn_zoo.md"
    assert paper.exists()
    assert zoo.exists()

    assert "# RNN Paper Zoo (100)" in paper.read_text(encoding="utf-8")
    assert "# RNN Zoo (100)" in zoo.read_text(encoding="utf-8")
