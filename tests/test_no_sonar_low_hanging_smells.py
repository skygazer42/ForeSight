from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_repo_file(path: str) -> str:
    return (_repo_root() / path).read_text(encoding="utf-8")


def _parse_repo_file(path: str) -> ast.AST:
    return ast.parse(_read_repo_file(path), filename=path)


def _function_uses_name(path: str, func_name: str, name: str) -> bool:
    tree = _parse_repo_file(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id == name:
                    return True
            return False
    raise AssertionError(f"Function {func_name!r} not found in {path}")


def _call_lines_missing_keyword(path: str, *, method_name: str, keyword: str) -> list[int]:
    tree = _parse_repo_file(path)
    out: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != method_name:
            continue
        if any(kw.arg == keyword for kw in node.keywords):
            continue
        out.append(node.lineno)
    return sorted(out)


def _dict_comp_update_lines(path: str) -> list[int]:
    tree = _parse_repo_file(path)
    out: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "update" or len(node.args) != 1:
            continue
        if isinstance(node.args[0], ast.DictComp):
            out.append(node.lineno)
    return sorted(out)


def _set_generator_lines(path: str) -> list[int]:
    tree = _parse_repo_file(path)
    out: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "set":
            continue
        if len(node.args) != 1:
            continue
        if isinstance(node.args[0], ast.GeneratorExp):
            out.append(node.lineno)
    return sorted(out)


def _unused_shape_dims(path: str) -> list[tuple[int, str]]:
    tree = _parse_repo_file(path)
    findings: list[tuple[int, str]] = []

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            loaded = {
                sub.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
            }
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Assign):
                    continue
                if not isinstance(sub.value, ast.Attribute) or sub.value.attr != "shape":
                    continue
                for target in sub.targets:
                    if not isinstance(target, ast.Tuple):
                        continue
                    for elt in target.elts:
                        if not isinstance(elt, ast.Name) or elt.id not in {"B", "T"}:
                            continue
                        if elt.id not in loaded:
                            findings.append((sub.lineno, elt.id))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return sorted(findings)


def test_hf_time_series_forecast_uses_epochs_parameter() -> None:
    assert _function_uses_name(
        "src/foresight/models/hf_time_series.py",
        "hf_timeseries_transformer_direct_forecast",
        "epochs",
    )


def test_hierarchical_merge_calls_specify_validate_keyword() -> None:
    assert (
        _call_lines_missing_keyword(
            "src/foresight/hierarchical.py",
            method_name="merge",
            keyword="validate",
        )
        == []
    )


def test_tuning_avoids_dict_comprehension_in_update_calls() -> None:
    assert _dict_comp_update_lines("src/foresight/tuning.py") == []


def test_tabular_avoids_set_constructor_with_generator() -> None:
    assert _set_generator_lines("src/foresight/features/tabular.py") == []


def test_features_tabular_source_extracts_complexity_helpers() -> None:
    source = _read_repo_file("src/foresight/features/tabular.py")

    assert "def _prepare_lag_feature_inputs(" in source
    assert "def _resolve_column_names(" in source
    assert "def _append_roll_stat_features(" in source
    assert "def _append_diff_features(" in source
    assert _function_uses_name(
        "src/foresight/features/tabular.py",
        "build_column_lag_features",
        "_prepare_lag_feature_inputs",
    )
    assert _function_uses_name(
        "src/foresight/features/tabular.py",
        "build_column_lag_features",
        "_resolve_column_names",
    )
    assert _function_uses_name(
        "src/foresight/features/tabular.py",
        "build_lag_derived_features",
        "_append_roll_stat_features",
    )
    assert _function_uses_name(
        "src/foresight/features/tabular.py",
        "build_lag_derived_features",
        "_append_diff_features",
    )


def test_features_time_source_extracts_complexity_helpers() -> None:
    source = _read_repo_file("src/foresight/features/time.py")

    assert "def _coerce_datetime_series(" in source
    assert "def _append_cyclical_feature_pair(" in source
    assert "def _normalize_fourier_periods(" in source
    assert "def _normalize_fourier_orders(" in source
    assert "def _normalize_string_fourier_orders(" in source
    assert "def _normalize_sequence_fourier_orders(" in source
    assert _function_uses_name(
        "src/foresight/features/time.py",
        "build_time_features",
        "_coerce_datetime_series",
    )
    assert _function_uses_name(
        "src/foresight/features/time.py",
        "build_time_features",
        "_append_datetime_cyclical_feature_pairs",
    )
    assert _function_uses_name(
        "src/foresight/features/time.py",
        "build_fourier_features",
        "_normalize_fourier_periods",
    )
    assert _function_uses_name(
        "src/foresight/features/time.py",
        "build_fourier_features",
        "_normalize_fourier_orders",
    )
    assert _function_uses_name(
        "src/foresight/features/time.py",
        "_normalize_fourier_orders",
        "_normalize_string_fourier_orders",
    )
    assert _function_uses_name(
        "src/foresight/features/time.py",
        "_normalize_fourier_orders",
        "_normalize_sequence_fourier_orders",
    )


def test_cli_data_source_extracts_dataset_validate_helpers() -> None:
    source = _read_repo_file("src/foresight/cli_data.py")

    assert "def _validate_dataset_frame(" in source
    assert "def _validate_dataset_parse_dates(" in source
    assert "def _validate_dataset_time_contracts(" in source
    assert _function_uses_name(
        "src/foresight/cli_data.py",
        "_cmd_datasets_validate",
        "_validate_dataset_frame",
    )
    assert _function_uses_name(
        "src/foresight/cli_data.py",
        "_cmd_datasets_validate",
        "_validate_dataset_parse_dates",
    )
    assert _function_uses_name(
        "src/foresight/cli_data.py",
        "_cmd_datasets_validate",
        "_validate_dataset_time_contracts",
    )


def test_cli_workflows_source_extracts_forecast_helpers() -> None:
    source = _read_repo_file("src/foresight/services/cli_workflows.py")

    assert "def _resolve_forecast_covariates(" in source
    assert "def _build_forecast_long_frames(" in source
    assert "def _save_local_forecast_artifact(" in source
    assert "def _save_global_forecast_artifact(" in source
    assert "def _forecast_local_artifact(" in source
    assert "def _forecast_global_artifact(" in source
    assert _function_uses_name(
        "src/foresight/services/cli_workflows.py",
        "forecast_csv_workflow",
        "_build_forecast_long_frames",
    )
    assert _function_uses_name(
        "src/foresight/services/cli_workflows.py",
        "forecast_csv_workflow",
        "_save_local_forecast_artifact",
    )
    assert _function_uses_name(
        "src/foresight/services/cli_workflows.py",
        "forecast_csv_workflow",
        "_save_global_forecast_artifact",
    )
    assert _function_uses_name(
        "src/foresight/services/cli_workflows.py",
        "forecast_artifact_workflow",
        "_forecast_local_artifact",
    )
    assert _function_uses_name(
        "src/foresight/services/cli_workflows.py",
        "forecast_artifact_workflow",
        "_forecast_global_artifact",
    )


def test_evaluation_source_extracts_eval_model_long_df_helpers() -> None:
    source = _read_repo_file("src/foresight/services/evaluation.py")

    assert "def _eval_global_model_long_df(" in source
    assert "def _eval_local_xreg_model_long_df(" in source
    assert "def _eval_local_univariate_model_long_df(" in source
    assert "def _summarize_eval_model_long_df_results(" in source
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "eval_model_long_df",
        "_eval_global_model_long_df",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "eval_model_long_df",
        "_eval_local_xreg_model_long_df",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "eval_model_long_df",
        "_eval_local_univariate_model_long_df",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "eval_model_long_df",
        "_summarize_eval_model_long_df_results",
    )


def test_cli_catalog_source_extracts_reused_help_literals() -> None:
    source = _read_repo_file("src/foresight/cli_catalog.py")

    assert '_OUTPUT_FORMAT_DEFAULT_TSV_HELP = "Output format (default: tsv)"' in source
    assert '_OPTIONAL_OUTPUT_PATH_HELP = "Optional path to write output"' in source
    assert '_RNN_PAPER_METADATA_FILENAME = "rnn_paper_metadata.json"' in source
    assert source.count('"Output format (default: tsv)"') == 1
    assert source.count('"Optional path to write output"') == 1
    assert source.count('"rnn_paper_metadata.json"') == 1


def test_kalman_positive_guard_uses_direct_non_positive_check() -> None:
    source = _read_repo_file("src/foresight/models/kalman.py")
    assert "if not (vf > 0.0):" not in source


def test_m4_summary_avoids_float_equality_guards() -> None:
    source = _read_repo_file("transformer time series/Time-Series/utils/m4_summary.py")

    assert "denom[denom == 0.0] = 1.0" not in source
    assert source.count("np.isclose(denom, 0.0)") >= 2


def test_global_regression_uses_lowercase_private_predict_helper_name() -> None:
    source = _read_repo_file("src/foresight/models/global_regression.py")
    assert "def _panel_step_lag_predict_X(" not in source
    assert "def _panel_step_lag_predict_x(" in source


def test_torch_files_do_not_leave_unused_shape_dimensions() -> None:
    assert _unused_shape_dims("src/foresight/models/torch_nn.py") == []
    assert _unused_shape_dims("src/foresight/models/torch_rnn_paper_zoo.py") == []


def test_torch_rnn_paper_zoo_avoids_sonar_flagged_source_patterns() -> None:
    source = _read_repo_file("src/foresight/models/torch_rnn_paper_zoo.py")

    assert "_PAPER_DESC = {paper_id: desc for paper_id, desc in _PAPER_DEFS}" not in source
    assert "w = torch.zeros((int(B), self.M), device=xb.device, dtype=xb.dtype)" not in source
    assert "return self.head(last)  # (B,2) = (mu, raw_sigma)" not in source
