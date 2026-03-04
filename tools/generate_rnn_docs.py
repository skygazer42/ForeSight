#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    # tools/generate_rnn_docs.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path(root: Path) -> None:
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _render_rnn_paper_zoo_doc() -> str:
    root = _repo_root()
    _ensure_src_on_path(root)
    from foresight.docsgen.rnn import render_rnn_paper_zoo_doc

    return render_rnn_paper_zoo_doc()


def _render_rnn_zoo_doc() -> str:
    root = _repo_root()
    _ensure_src_on_path(root)
    from foresight.docsgen.rnn import render_rnn_zoo_doc

    return render_rnn_zoo_doc()


def main() -> int:
    root = _repo_root()
    _ensure_src_on_path(root)
    from foresight.docsgen.rnn import write_rnn_docs

    docs_dir = root / "docs"
    write_rnn_docs(output_dir=docs_dir)

    print("Wrote:")
    print(" - docs/rnn_paper_zoo.md")
    print(" - docs/rnn_zoo.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
