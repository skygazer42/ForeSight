import ast
import os
import subprocess
import sys
from pathlib import Path


def _self_source() -> str:
    return Path(__file__).read_text(encoding="utf-8")


def _bound_names_from_import_aliases(node: ast.Import) -> set[str]:
    return {alias.asname or alias.name.split(".")[0] for alias in node.names}


def _bound_names_from_import_from_aliases(node: ast.ImportFrom) -> set[str]:
    if node.module == "__future__":
        return set()
    return {alias.asname or alias.name for alias in node.names if alias.name != "*"}


def _import_bound_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return _bound_names_from_import_aliases(node)
    if isinstance(node, ast.ImportFrom):
        return _bound_names_from_import_from_aliases(node)
    return set()


def _assignment_bound_name(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Assign):
        names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _definition_bound_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name
    return None


def _bound_names_for_node(node: ast.AST) -> set[str]:
    names = _import_bound_names(node)
    if names:
        return names
    names = _assignment_bound_name(node)
    if names:
        return names
    definition_name = _definition_bound_name(node)
    return {definition_name} if definition_name is not None else set()


def _top_level_bound_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        names.update(_bound_names_for_node(node))
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


def test_cli_facade_does_not_bind_shared_cli_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cli_names = _top_level_bound_names(repo_root / "src" / "foresight" / "cli.py")

    shared_helpers = {
        "_coerce_model_param_value",
        "_parse_model_params",
        "_parse_grid_params",
        "_emit_text",
        "_emit_dataframe",
        "_emit",
        "_emit_table",
        "_format_payload",
        "_format_csv",
        "_format_markdown",
        "_format_table",
    }

    assert cli_names.isdisjoint(shared_helpers)


def test_cli_facade_does_not_bind_catalog_cli_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cli_names = _top_level_bound_names(repo_root / "src" / "foresight" / "cli.py")

    catalog_helpers = {
        "_load_rnn_paper_metadata",
        "_cmd_models_list",
        "_cmd_models_info",
        "_cmd_models_search",
        "_cmd_papers_list",
        "_cmd_papers_info",
        "_cmd_papers_models",
        "_cmd_docs_rnn",
    }

    assert cli_names.isdisjoint(catalog_helpers)


def test_cli_facade_does_not_bind_data_cli_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cli_names = _top_level_bound_names(repo_root / "src" / "foresight" / "cli.py")

    data_helpers = {
        "_cmd_datasets_list",
        "_cmd_datasets_preview",
        "_cmd_datasets_path",
        "_cmd_datasets_validate",
        "_cmd_data_to_long",
        "_cmd_data_prepare_long",
        "_cmd_data_infer_freq",
        "_cmd_data_splits_rolling_origin",
    }

    assert cli_names.isdisjoint(data_helpers)


def test_cli_facade_does_not_bind_leaderboard_cli_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cli_names = _top_level_bound_names(repo_root / "src" / "foresight" / "cli.py")

    leaderboard_helpers = {
        "_cmd_leaderboard_naive",
        "_cmd_leaderboard_models",
        "_leaderboard_sweep_worker",
        "_cmd_leaderboard_sweep",
        "_cmd_leaderboard_summarize",
        "_summarize_leaderboard_rows",
        "_leaderboard_summary_columns",
        "_run_parallel_tasks",
        "_SWEEP_LONG_DF_CACHE",
    }

    assert cli_names.isdisjoint(leaderboard_helpers)


def test_source_extracts_top_level_binding_helpers() -> None:
    tree = ast.parse(_self_source(), filename=__file__)
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "_bound_names_from_import_aliases" in function_names
    assert "_bound_names_from_import_from_aliases" in function_names
