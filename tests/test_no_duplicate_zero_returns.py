from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _literal_zero_return_count(path: str, func_name: str) -> int:
    source = (_repo_root() / path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            count = 0
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Constant):
                    if child.value.value == 0:
                        count += 1
            return count
    raise AssertionError(f"Function {func_name!r} not found in {path}")


def test_cli_tuning_command_has_single_zero_exit_path() -> None:
    assert _literal_zero_return_count("src/foresight/cli.py", "_cmd_tuning_run") == 1


def test_release_check_main_has_single_zero_exit_path() -> None:
    assert _literal_zero_return_count("tools/release_check.py", "main") == 1


def test_convert_ipynb_main_has_single_zero_exit_path() -> None:
    assert _literal_zero_return_count("tools/convert_ipynb_to_py.py", "main") == 1
