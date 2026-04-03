from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _self_source() -> str:
    return Path(__file__).read_text(encoding="utf-8")


def _function_def_nodes(tree: ast.Module) -> list[ast.FunctionDef]:
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)]


def _find_named_function(tree: ast.Module, func_name: str) -> ast.FunctionDef:
    for node in _function_def_nodes(tree):
        if node.name == func_name:
            return node
    raise AssertionError(f"Function {func_name!r} not found in source")


def _is_literal_zero_return(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 0
    )


def _count_literal_zero_returns(node: ast.FunctionDef) -> int:
    count = 0
    for child in ast.walk(node):
        if _is_literal_zero_return(child):
            count += 1
    return count


def _literal_zero_return_count(path: str, func_name: str) -> int:
    source = (_repo_root() / path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    node = _find_named_function(tree, func_name)
    return _count_literal_zero_returns(node)


def test_cli_tuning_command_has_single_zero_exit_path() -> None:
    assert _literal_zero_return_count("src/foresight/cli.py", "_cmd_tuning_run") == 1


def test_release_check_main_has_single_zero_exit_path() -> None:
    assert _literal_zero_return_count("tools/release_check.py", "main") == 1


def test_convert_ipynb_main_has_single_zero_exit_path() -> None:
    assert _literal_zero_return_count("tools/convert_ipynb_to_py.py", "main") == 1


def test_source_extracts_zero_return_helpers() -> None:
    tree = ast.parse(_self_source(), filename=__file__)
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    assert "_function_def_nodes" in function_names
    assert "_is_literal_zero_return" in function_names
