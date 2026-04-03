from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mergeable_nested_if_lines(path: str, func_name: str) -> list[int]:
    source = (_repo_root() / path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return sorted(
                {
                    child.lineno
                    for child in ast.walk(node)
                    if isinstance(child, ast.If)
                    and not child.orelse
                    and len(child.body) == 1
                    and isinstance(child.body[0], ast.If)
                    and not child.body[0].orelse
                }
            )

    raise AssertionError(f"Function {func_name!r} not found in {path}")


def _function_source(path: str, func_name: str) -> str:
    source = (_repo_root() / path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(source, node) or ""

    raise AssertionError(f"Function {func_name!r} not found in {path}")


def test_augment_lag_feat_row_has_no_mergeable_nested_ifs() -> None:
    assert (
        _mergeable_nested_if_lines("src/foresight/models/regression.py", "_augment_lag_feat_row")
        == []
    )


def test_xgb_lag_direct_forecast_has_no_mergeable_nested_ifs() -> None:
    assert (
        _mergeable_nested_if_lines("src/foresight/models/regression.py", "_xgb_lag_direct_forecast")
        == []
    )


def test_xgb_lag_recursive_forecast_has_no_mergeable_nested_ifs() -> None:
    assert (
        _mergeable_nested_if_lines(
            "src/foresight/models/regression.py", "_xgb_lag_recursive_forecast"
        )
        == []
    )


def test_xgb_objective_label_validation_has_no_mergeable_nested_ifs() -> None:
    assert (
        _mergeable_nested_if_lines(
            "src/foresight/models/regression.py", "_xgb_validate_objective_label_constraints"
        )
        == []
    )


def test_xgb_common_regressor_param_validation_has_no_mergeable_nested_ifs() -> None:
    assert (
        _mergeable_nested_if_lines(
            "src/foresight/models/regression.py", "_xgb_validate_common_regressor_params"
        )
        == []
    )


def test_xgb_common_regressor_param_validation_avoids_nested_bound_checks() -> None:
    source = _function_source(
        "src/foresight/models/regression.py", "_xgb_validate_common_regressor_params"
    )

    assert 'if "subsample" in params and params["subsample"] is not None:' not in source
    assert (
        'if "colsample_bytree" in params and params["colsample_bytree"] is not None:' not in source
    )


def test_lgbm_common_regressor_param_validation_avoids_nested_bound_checks() -> None:
    source = _function_source(
        "src/foresight/models/regression.py", "_lgbm_validate_common_regressor_params"
    )

    assert 'if "subsample" in params and params["subsample"] is not None:' not in source
    assert (
        'if "colsample_bytree" in params and params["colsample_bytree"] is not None:' not in source
    )
