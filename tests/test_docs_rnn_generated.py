from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_generate_rnn_docs_module(repo_root: Path):
    path = repo_root / "tools" / "generate_rnn_docs.py"
    spec = importlib.util.spec_from_file_location("generate_rnn_docs", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rnn_docs_are_up_to_date() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"

    mod = _load_generate_rnn_docs_module(repo_root)

    expected_paper = mod._render_rnn_paper_zoo_doc()  # type: ignore[attr-defined]
    expected_zoo = mod._render_rnn_zoo_doc()  # type: ignore[attr-defined]

    actual_paper = (docs_dir / "rnn_paper_zoo.md").read_text(encoding="utf-8")
    actual_zoo = (docs_dir / "rnn_zoo.md").read_text(encoding="utf-8")

    assert actual_paper == expected_paper
    assert actual_zoo == expected_zoo
