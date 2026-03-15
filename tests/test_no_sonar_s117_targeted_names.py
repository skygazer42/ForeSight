from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _self_source() -> str:
    return Path(__file__).read_text(encoding="utf-8")


TARGETED_BAD_NAMES: dict[str, set[str]] = {
    "src/foresight/models/ar.py": {"P"},
    "src/foresight/features/tabular.py": {"X_lags"},
    "src/foresight/models/analog.py": {"X_work"},
    "src/foresight/models/fourier.py": {"Xf"},
    "src/foresight/models/global_regression.py": {
        "C",
        "C_f",
        "X_all",
        "X_base",
        "X_core",
        "X_long",
        "X_pred",
        "X_rep",
        "Xs",
    },
    "src/foresight/models/regression.py": {
        "X_aug",
        "X_base",
        "X_base_aug",
        "X_long",
        "X_rep",
        "X_step",
        "Xj",
        "feat_1xL",
    },
    "src/foresight/models/spectral.py": {"Xf"},
    "src/foresight/models/ssa.py": {"U", "Ur", "Vt", "Vtr", "Xr"},
    "src/foresight/models/statsmodels_wrap.py": {
        "X_cols",
        "Xf",
        "Xf_cols",
        "max_D",
        "max_P",
        "max_Q",
    },
    "src/foresight/models/torch_global.py": {
        "B_inv",
        "Bmat",
        "Fp",
        "Hh",
        "L_pad",
        "X_chunks",
        "X_pred",
        "X_pred_arr",
        "X_series",
        "X_t",
        "X_train",
        "X_tr",
        "X_va",
        "X_val",
        "Xp",
        "Y_chunks",
        "Y_series",
        "Y_t",
        "Y_train",
        "Y_tr",
        "Y_va",
        "Y_val",
        "pred_X",
    },
    "src/foresight/models/catalog/classical.py": {"ModelSpec"},
    "src/foresight/models/catalog/foundation.py": {"ModelSpec"},
    "src/foresight/models/catalog/ml.py": {"ModelSpec"},
    "src/foresight/models/catalog/multivariate.py": {"ModelSpec"},
    "src/foresight/models/catalog/stats.py": {"ModelSpec"},
    "src/foresight/models/catalog/torch_global.py": {"ModelSpec"},
    "src/foresight/models/catalog/torch_local.py": {"ModelSpec"},
    "src/foresight/models/kalman.py": {"P_pred"},
    "src/foresight/models/torch_ct_rnn.py": {"X_seq"},
    "src/foresight/models/torch_nn.py": {
        "X_grid",
        "X_patch",
        "X_seg",
        "X_seq",
        "X_t",
        "X_val",
        "Y_next",
        "Y_t",
        "Y_val",
    },
    "src/foresight/models/torch_probabilistic.py": {"X_seq"},
    "src/foresight/models/torch_rnn_paper_zoo.py": {
        "Hh",
        "Ww",
        "X_seq",
        "X_t",
        "X_val",
        "Y_next",
        "Y_t",
        "Y_val",
    },
    "src/foresight/models/torch_rnn_zoo.py": {"X_seq"},
    "src/foresight/models/torch_seq2seq.py": {"X_seq", "X_t", "X_val", "Y_t", "Y_val"},
    "src/foresight/models/torch_ssm.py": {"X_seq"},
    "src/foresight/models/torch_xformer.py": {
        "B_inv",
        "Bmat",
        "Fp",
        "Hh",
        "L_pad",
        "X_pad",
        "X_seq",
    },
    "src/foresight/models/trend.py": {"Xf"},
    "tests/test_forecaster_api.py": {"C"},
}

TARGETED_RUNTIME_FUNCTION_BAD_NAMES: dict[str, set[str]] = {
    "_factory_sar_ols": {"P", "P_int"},
    "_factory_svr_lag": {"C", "C_f"},
    "_factory_linear_svr_lag": {"C", "C_f"},
    "_factory_auto_arima": {
        "max_P",
        "max_D",
        "max_Q",
        "max_P_int",
        "max_D_int",
        "max_Q_int",
    },
}


def _function_arg_names(node: ast.FunctionDef) -> list[str]:
    arg_nodes = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    if node.args.vararg is not None:
        arg_nodes.append(node.args.vararg)
    if node.args.kwarg is not None:
        arg_nodes.append(node.args.kwarg)
    return [arg.arg for arg in arg_nodes]


def _assignment_target_nodes_from_stmt(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    if isinstance(node, ast.For):
        return [node.target]
    if isinstance(node, ast.With):
        return [item.optional_vars for item in node.items if item.optional_vars is not None]
    return []


def _assignment_target_nodes(node: ast.FunctionDef) -> list[ast.AST]:
    targets: list[ast.AST] = []
    for sub in ast.walk(node):
        targets.extend(_assignment_target_nodes_from_stmt(sub))
    return targets


def _leaf_assigned_names(target: ast.AST) -> list[str]:
    return [leaf.id for leaf in ast.walk(target) if isinstance(leaf, ast.Name)]


def _assigned_names_from_function_node(node: ast.FunctionDef) -> list[str]:
    names = list(_function_arg_names(node))
    for target in _assignment_target_nodes(node):
        names.extend(_leaf_assigned_names(target))
    return names


def _parse_tree_for_path(path: str) -> ast.Module:
    source = (_repo_root() / path).read_text(encoding="utf-8")
    return ast.parse(source, filename=path)


def _function_def_nodes(tree: ast.AST) -> list[ast.FunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def _assigned_names_in_functions(path: str) -> list[str]:
    tree = _parse_tree_for_path(path)
    names: list[str] = []
    for node in _function_def_nodes(tree):
        names.extend(_assigned_names_from_function_node(node))
    return names


def _named_function_node(tree: ast.AST, func_name: str) -> ast.FunctionDef:
    for node in _function_def_nodes(tree):
        if node.name == func_name:
            return node
    raise AssertionError(f"Function {func_name!r} not found in parsed module")


def _assigned_names_in_function(path: str, func_name: str) -> list[str]:
    tree = _parse_tree_for_path(path)
    return _assigned_names_from_function_node(_named_function_node(tree, func_name))


def test_targeted_files_do_not_keep_s117_local_names() -> None:
    for path, bad_names in TARGETED_BAD_NAMES.items():
        names = set(_assigned_names_in_functions(path))
        present = sorted(names.intersection(bad_names))
        assert present == [], f"{path} still contains Sonar S117 names: {present}"


def test_runtime_targeted_factories_do_not_keep_s117_local_names() -> None:
    path = "src/foresight/models/runtime.py"
    for func_name, bad_names in TARGETED_RUNTIME_FUNCTION_BAD_NAMES.items():
        names = set(_assigned_names_in_function(path, func_name))
        present = sorted(names.intersection(bad_names))
        assert present == [], f"{path}:{func_name} still contains Sonar S117 names: {present}"


def test_source_extracts_ast_name_collection_helpers() -> None:
    tree = ast.parse(_self_source(), filename=__file__)
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "_parse_tree_for_path" in function_names
    assert "_function_def_nodes" in function_names
    assert "_assignment_target_nodes_from_stmt" in function_names
    assert "_named_function_node" in function_names
