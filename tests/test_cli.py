import ast
import os
import subprocess
import sys
from pathlib import Path


def _top_level_bound_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
    names.discard("__all__")
    return names


def test_cli_help_exits_zero():
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    proc = subprocess.run(
        [sys.executable, "-m", "foresight", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert "ForeSight" in (proc.stdout + proc.stderr)


def test_public_facade_modules_only_bind_supported_entrypoints() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    forecast_names = _top_level_bound_names(repo_root / "src" / "foresight" / "forecast.py")
    assert forecast_names == {"forecast_model", "forecast_model_long_df"}

    eval_names = _top_level_bound_names(repo_root / "src" / "foresight" / "eval_forecast.py")
    assert eval_names == {
        "eval_hierarchical_forecast_df",
        "eval_model",
        "eval_model_long_df",
        "eval_multivariate_model_df",
    }
