from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


TARGETED_BAD_NAMES: dict[str, set[str]] = {
    "src/foresight/features/tabular.py": {"X_lags"},
    "src/foresight/models/analog.py": {"X_work"},
    "src/foresight/models/fourier.py": {"Xf"},
    "src/foresight/models/global_regression.py": {
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
    "src/foresight/models/statsmodels_wrap.py": {"X_cols", "Xf", "Xf_cols"},
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
}


def _assigned_names_in_functions(path: str) -> list[str]:
    source = (_repo_root() / path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    names: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            arg_nodes = [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
            if node.args.vararg is not None:
                arg_nodes.append(node.args.vararg)
            if node.args.kwarg is not None:
                arg_nodes.append(node.args.kwarg)
            names.extend(arg.arg for arg in arg_nodes)
            for sub in ast.walk(node):
                targets: list[ast.AST] = []
                if isinstance(sub, ast.Assign):
                    targets.extend(sub.targets)
                elif isinstance(sub, ast.AnnAssign):
                    targets.append(sub.target)
                elif isinstance(sub, ast.For):
                    targets.append(sub.target)
                elif isinstance(sub, ast.With):
                    for item in sub.items:
                        if item.optional_vars is not None:
                            targets.append(item.optional_vars)
                for target in targets:
                    for leaf in ast.walk(target):
                        if isinstance(leaf, ast.Name):
                            names.append(leaf.id)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return names


def test_targeted_files_do_not_keep_s117_local_names() -> None:
    for path, bad_names in TARGETED_BAD_NAMES.items():
        names = set(_assigned_names_in_functions(path))
        present = sorted(names.intersection(bad_names))
        assert present == [], f"{path} still contains Sonar S117 names: {present}"
