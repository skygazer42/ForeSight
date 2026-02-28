from __future__ import annotations

import runpy
from pathlib import Path


def _load_tool():
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(root / "tools" / "check_no_ipynb.py")


def test_check_no_ipynb_ignores_worktrees(tmp_path: Path):
    (tmp_path / ".worktrees" / "w1").mkdir(parents=True)
    (tmp_path / ".worktrees" / "w1" / "nb.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "real.ipynb").write_text("{}", encoding="utf-8")

    mod = _load_tool()
    find_notebooks = mod["find_notebooks"]
    hits = find_notebooks(tmp_path)
    assert hits == [Path("real.ipynb")]

