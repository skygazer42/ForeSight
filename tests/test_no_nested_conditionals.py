from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _nested_ifexp_lines(path: str) -> list[int]:
    source = (_repo_root() / path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    return sorted(
        {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.IfExp)
            and (isinstance(node.body, ast.IfExp) or isinstance(node.orelse, ast.IfExp))
        }
    )


def test_rnn_docsgen_has_no_nested_conditional_expressions() -> None:
    assert _nested_ifexp_lines("src/foresight/docsgen/rnn.py") == []


def test_statsmodels_wrap_has_no_nested_conditional_expressions() -> None:
    assert _nested_ifexp_lines("src/foresight/models/statsmodels_wrap.py") == []


def test_torch_rnn_paper_zoo_has_no_nested_conditional_expressions() -> None:
    assert _nested_ifexp_lines("src/foresight/models/torch_rnn_paper_zoo.py") == []
