from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tuple_list_comprehension_lines(path: str) -> list[int]:
    source = (_repo_root() / path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    return sorted(
        {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "tuple"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.ListComp)
        }
    )


def test_global_regression_has_no_tuple_wrapped_list_comprehensions() -> None:
    assert _tuple_list_comprehension_lines("src/foresight/models/global_regression.py") == []


def test_cv_has_no_tuple_wrapped_list_comprehensions() -> None:
    assert _tuple_list_comprehension_lines("src/foresight/cv.py") == []


def test_torch_global_has_no_tuple_wrapped_list_comprehensions() -> None:
    assert _tuple_list_comprehension_lines("src/foresight/models/torch_global.py") == []


def test_io_has_no_tuple_wrapped_list_comprehensions() -> None:
    assert _tuple_list_comprehension_lines("src/foresight/io.py") == []
