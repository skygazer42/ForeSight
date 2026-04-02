from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_repo_file(path: str) -> str:
    return (_repo_root() / path).read_text(encoding="utf-8-sig")


def _read_repo_lines(path: str) -> list[str]:
    return _read_repo_file(path).splitlines()


def _line_at(path: str, lineno: int) -> str:
    return _read_repo_lines(path)[lineno - 1]


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


def _function_uses_attr(path: str, func_name: str, owner_name: str, attr_name: str) -> bool:
    tree = _parse_repo_file(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == owner_name
                    and sub.attr == attr_name
                ):
                    return True
            return False
    raise AssertionError(f"Function {func_name!r} not found in {path}")


def _call_matches_method_without_keyword(
    node: ast.AST,
    *,
    method_name: str,
    keyword: str,
) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method_name
        and not any(kw.arg == keyword for kw in node.keywords)
    )


def _call_lines_missing_keyword(path: str, *, method_name: str, keyword: str) -> list[int]:
    tree = _parse_repo_file(path)
    out: list[int] = []
    for node in ast.walk(tree):
        if _call_matches_method_without_keyword(
            node,
            method_name=method_name,
            keyword=keyword,
        ):
            out.append(node.lineno)
    return sorted(out)


def _call_lines_with_keyword_alias(
    path: str,
    *,
    owner_name: str,
    method_name: str,
    keyword: str,
) -> list[int]:
    tree = _parse_repo_file(path)
    out: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == owner_name
            and node.func.attr == method_name
            and any(kw.arg == keyword for kw in node.keywords)
        ):
            out.append(node.lineno)
    return sorted(out)


def _is_single_arg_dict_comp_update_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.DictComp)
    )


def _dict_comp_update_lines(path: str) -> list[int]:
    tree = _parse_repo_file(path)
    out: list[int] = []
    for node in ast.walk(tree):
        if _is_single_arg_dict_comp_update_call(node):
            out.append(node.lineno)
    return sorted(out)


def _is_single_arg_set_generator_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.GeneratorExp)
    )


def _set_generator_lines(path: str) -> list[int]:
    tree = _parse_repo_file(path)
    out: list[int] = []
    for node in ast.walk(tree):
        if _is_single_arg_set_generator_call(node):
            out.append(node.lineno)
    return sorted(out)


def _shape_dim_load_names(node: ast.AST) -> set[str]:
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
    }


def _shape_tuple_targets_for_assign(node: ast.AST) -> list[ast.Tuple]:
    if not isinstance(node, ast.Assign):
        return []
    if not isinstance(node.value, ast.Attribute) or node.value.attr != "shape":
        return []
    return [target for target in node.targets if isinstance(target, ast.Tuple)]


def _shape_dim_assignment_targets(node: ast.AST) -> list[tuple[int, ast.Tuple]]:
    targets: list[tuple[int, ast.Tuple]] = []
    for sub in ast.walk(node):
        targets.extend((sub.lineno, target) for target in _shape_tuple_targets_for_assign(sub))
    return targets


def _class_method_names(path: str, class_name: str) -> list[str]:
    tree = _parse_repo_file(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                sub.name
                for sub in node.body
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
    raise AssertionError(f"Class {class_name!r} not found in {path}")


def _required_method_param_names(path: str, class_name: str, method_name: str) -> list[str]:
    tree = _parse_repo_file(path)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for sub in node.body:
            if not isinstance(sub, ast.FunctionDef) or sub.name != method_name:
                continue
            positional = list(sub.args.posonlyargs) + list(sub.args.args)
            non_self = positional[1:]
            required_count = max(0, len(non_self) - len(sub.args.defaults))
            return [arg.arg for arg in non_self[:required_count]]
        raise AssertionError(f"Method {method_name!r} not found on {class_name!r} in {path}")
    raise AssertionError(f"Class {class_name!r} not found in {path}")


def _top_level_class_names(path: str) -> set[str]:
    tree = _parse_repo_file(path)
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def _top_level_name_aliases(path: str) -> set[tuple[str, str]]:
    tree = _parse_repo_file(path)
    aliases: set[tuple[str, str]] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Name):
            continue
        aliases.add((target.id, node.value.id))
    return aliases


def _declared_names(path: str) -> set[str]:
    tree = _parse_repo_file(path)
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    names.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    return names


def _collect_unused_shape_dims(node: ast.AST) -> list[tuple[int, str]]:
    loaded = _shape_dim_load_names(node)
    findings: list[tuple[int, str]] = []
    for lineno, target in _shape_dim_assignment_targets(node):
        for elt in target.elts:
            if not isinstance(elt, ast.Name) or elt.id not in {"B", "T"}:
                continue
            if elt.id not in loaded:
                findings.append((lineno, elt.id))
    return findings


def _unused_shape_dims(path: str) -> list[tuple[int, str]]:
    tree = _parse_repo_file(path)
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            findings.extend(_collect_unused_shape_dims(node))
    return sorted(findings)


_CODE_LIKE_COMMENT_PATTERNS = (
    re.compile(
        r"^\s*#\s*(print\(|exit\(|try:|except\b|with\b|if\b|elif\b|else:|for\b|while\b|return\b|"
        r"self\.|torch\.|plt\.|parser\.|trainer\.|download_[A-Za-z_]+\(|config\s*=|x_forward\s*=|"
        r"context_forward\s*=|fill\s*=|zero\s*=|flatten\s*=|ipdb\.|train_percent_check|"
        r"val_percent_check|test_percent_check|fast_dev_run|distributed_backend|on_gpu\s*=|!pwd)"
    ),
    re.compile(r"^\s*#\s*[\w.\[\]'\"()]+\s*="),
)

_NON_CODE_COMMENT_PREFIXES = (
    "# -*-",
    "# %%",
    "# ##",
    "# ###",
    "# Converted from:",
    "# Data ",
    "# Relevant ",
    "# Network ",
    "# Serialisation ",
    "# Extra ",
    "# Build ",
    "# Static ",
    "# Target ",
    "# Observed ",
    "# A priori ",
    "# OPTIONAL",
    "# REQUIRED",
    "# can return",
    "# example ",
)


def _code_like_comment_lines(path: str) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    for lineno, line in enumerate(_read_repo_file(path).splitlines(), start=1):
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            continue
        if stripped.startswith(_NON_CODE_COMMENT_PREFIXES):
            continue
        if any(pattern.search(line) for pattern in _CODE_LIKE_COMMENT_PATTERNS):
            findings.append((lineno, stripped))
    return findings


def _pyflakes_messages(paths: list[str]) -> list[str]:
    repo = _repo_root()
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", *paths],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _has_side_effect_value(node: ast.AST) -> bool:
    return isinstance(node, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom))


def _bare_no_effect_expr_lines(path: str) -> list[int]:
    tree = _parse_repo_file(path)
    out: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if _has_side_effect_value(node.value):
            continue
        out.append(node.lineno)
    return sorted(out)


def test_source_extracts_ast_scan_helpers() -> None:
    tree = _parse_repo_file("tests/test_no_sonar_low_hanging_smells.py")
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "_call_matches_method_without_keyword" in function_names
    assert "_is_single_arg_dict_comp_update_call" in function_names
    assert "_is_single_arg_set_generator_call" in function_names
    assert "_shape_tuple_targets_for_assign" in function_names


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
    assert "def _validated_eval_long_df_request(" in source
    assert "def _local_eval_model_long_df_results(" in source
    assert "def _global_eval_metrics_payload(" in source
    assert "def _update_global_eval_quantile_payload(" in source
    assert "def _update_global_eval_conformal_payload(" in source
    assert "def _local_xreg_eval_arrays(" in source
    assert "def _append_eval_window_results(" in source
    assert "def _eval_local_xreg_model_long_df(" in source
    assert "def _eval_local_univariate_model_long_df(" in source
    assert "def _summarize_eval_model_long_df_results(" in source
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "eval_model_long_df",
        "_validated_eval_long_df_request",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "eval_model_long_df",
        "_local_eval_model_long_df_results",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "_local_eval_model_long_df_results",
        "_eval_local_xreg_model_long_df",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "_local_eval_model_long_df_results",
        "_eval_local_univariate_model_long_df",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "eval_model_long_df",
        "_eval_global_model_long_df",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "_eval_global_model_long_df",
        "_global_eval_metrics_payload",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "_eval_global_model_long_df",
        "_update_global_eval_quantile_payload",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "_eval_global_model_long_df",
        "_update_global_eval_conformal_payload",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "_local_eval_model_long_df_results",
        "_eval_local_xreg_model_long_df",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "_eval_local_xreg_model_long_df",
        "_local_xreg_eval_arrays",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "_eval_local_xreg_model_long_df",
        "_append_eval_window_results",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "_local_eval_model_long_df_results",
        "_eval_local_univariate_model_long_df",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "_eval_local_univariate_model_long_df",
        "_append_eval_window_results",
    )
    assert _function_uses_name(
        "src/foresight/services/evaluation.py",
        "eval_model_long_df",
        "_summarize_eval_model_long_df_results",
    )


def test_forecasting_source_extracts_forecast_model_long_df_helpers() -> None:
    source = _read_repo_file("src/foresight/services/forecasting.py")

    assert "def _global_forecast_group_cutoff_and_future(" in source
    assert "def _global_forecast_group_future_frame(" in source
    assert "def _append_future_rows_for_group(" in source
    assert "def _validated_global_forecast_cutoff(" in source
    assert "def _validate_global_forecast_group_x_cols(" in source
    assert "def _forecast_result_row(" in source
    assert "def _forecast_local_xreg_long_df(" in source
    assert "def _forecast_local_univariate_long_df(" in source
    assert "def _forecast_global_long_df(" in source
    assert _function_uses_name(
        "src/foresight/services/forecasting.py",
        "_prepare_global_forecast_input",
        "_global_forecast_group_cutoff_and_future",
    )
    assert _function_uses_name(
        "src/foresight/services/forecasting.py",
        "_prepare_global_forecast_input",
        "_validated_global_forecast_cutoff",
    )
    assert _function_uses_name(
        "src/foresight/services/forecasting.py",
        "_prepare_global_forecast_input",
        "_append_future_rows_for_group",
    )
    assert _function_uses_name(
        "src/foresight/services/forecasting.py",
        "_global_forecast_group_future_frame",
        "_global_forecast_group_cutoff_and_future",
    )
    assert _function_uses_name(
        "src/foresight/services/forecasting.py",
        "_global_forecast_group_future_frame",
        "_validate_global_forecast_group_x_cols",
    )
    assert _function_uses_name(
        "src/foresight/services/forecasting.py",
        "forecast_model_long_df",
        "_forecast_local_xreg_long_df",
    )
    assert _function_uses_name(
        "src/foresight/services/forecasting.py",
        "forecast_model_long_df",
        "_forecast_local_univariate_long_df",
    )
    assert _function_uses_name(
        "src/foresight/services/forecasting.py",
        "forecast_model_long_df",
        "_forecast_global_long_df",
    )


def test_cli_catalog_source_extracts_reused_help_literals() -> None:
    source = _read_repo_file("src/foresight/cli_catalog.py")

    assert '_OUTPUT_FORMAT_DEFAULT_TSV_HELP = "Output format (default: tsv)"' in source
    assert '_OPTIONAL_OUTPUT_PATH_HELP = "Optional path to write output"' in source
    assert '_RNN_PAPER_METADATA_FILENAME = "rnn_paper_metadata.json"' in source
    assert source.count('"Output format (default: tsv)"') == 1
    assert source.count('"Optional path to write output"') == 1
    assert source.count('"rnn_paper_metadata.json"') == 1


def test_runtime_source_extracts_reused_literal_constants() -> None:
    source = _read_repo_file("src/foresight/models/runtime.py")

    assert '_XGB_REG_SQUAREDERROR_OBJECTIVE = "reg:squarederror"' in source
    assert '_MEMBERS_NON_EMPTY_ERROR = "members must be non-empty"' in source
    assert '_ORDER_TUPLE_ERROR = "order must be a 3-tuple like (p, d, q)"' in source
    assert (
        '_SEASONAL_ORDER_TUPLE_ERROR = "seasonal_order must be a 4-tuple like (P, D, Q, s)"'
        in source
    )
    assert '_LOCAL_LEVEL_LITERAL = "local level"' in source
    assert source.count('"reg:squarederror"') == 1
    assert source.count('"members must be non-empty"') == 1
    assert source.count('"order must be a 3-tuple like (p, d, q)"') == 1
    assert source.count('"seasonal_order must be a 4-tuple like (P, D, Q, s)"') == 1
    assert source.count('"local level"') == 1


def test_tft_sources_avoid_code_like_comments() -> None:
    for path in (
        "transformer time series/tft/training_tft.py",
        "transformer time series/tft/run.py",
        "transformer time series/tft/models.py",
        "transformer time series/Time-Series/layers/FourierCorrelation.py",
        "transformer time series/Time-Series/run.py",
        "transformer time series/tft/quantile_loss.py",
        "transformer time series/Time-Series/utils/timefeatures.py",
        "transformer time series/SelfAttention_Family.py",
        "transformer time series/tft/data/data_download.py",
        "transformer time series/Time-Series/layers/Transformer_EncDec.py",
        "transformer time series/Time-Series/layers/SelfAttention_Family.py",
        "transformer time series/Time-Series/layers/Embed.py",
        "transformer time series/Time-Series/exp/exp_short_term_forecasting.py",
        "transformer time series/tft/data_download.py",
        "transformer time series/Time-Series/layers/MultiWaveletCorrelation.py",
        "transformer time series/Time-Series/models/Crossformer.py",
        "transformer time series/Embed.py",
        "transformer time series/Time-Series/utils/tools.py",
    ):
        assert _code_like_comment_lines(path) == []


def test_transformer_sources_avoid_selected_inline_code_like_comment_fragments() -> None:
    custom_dataset_source = _read_repo_file("transformer time series/tft/data/custom_dataset.py")
    training_tft_source = _read_repo_file("transformer time series/tft/training_tft.py")
    tft_run_source = _read_repo_file("transformer time series/tft/run.py")
    embed_source = _read_repo_file("transformer time series/Time-Series/layers/Embed.py")
    embed_legacy_source = _read_repo_file("transformer time series/Embed.py")
    transformer_encdec_source = _read_repo_file(
        "transformer time series/Time-Series/layers/Transformer_EncDec.py"
    )
    autoformer_source = _read_repo_file("transformer time series/Time-Series/models/Autoformer.py")
    crossformer_source = _read_repo_file("transformer time series/Time-Series/models/Crossformer.py")
    dlinear_source = _read_repo_file("transformer time series/Time-Series/models/DLinear.py")
    etsformer_source = _read_repo_file("transformer time series/Time-Series/models/ETSformer.py")
    fedformer_source = _read_repo_file("transformer time series/Time-Series/models/FEDformer.py")
    informer_source = _read_repo_file("transformer time series/Time-Series/models/Informer.py")
    lightts_source = _read_repo_file("transformer time series/Time-Series/models/LightTS.py")
    masking_source = _read_repo_file("transformer time series/Time-Series/utils/masking.py")
    micn_source = _read_repo_file("transformer time series/Time-Series/models/MICN.py")
    multi_wavelet_source = _read_repo_file(
        "transformer time series/Time-Series/layers/MultiWaveletCorrelation.py"
    )
    nonstationary_source = _read_repo_file(
        "transformer time series/Time-Series/models/Nonstationary_Transformer.py"
    )
    patchtst_source = _read_repo_file("transformer time series/Time-Series/models/PatchTST.py")
    pyraformer_source = _read_repo_file("transformer time series/Time-Series/models/Pyraformer.py")
    reformer_source = _read_repo_file("transformer time series/Time-Series/models/Reformer.py")
    self_attention_source = _read_repo_file(
        "transformer time series/Time-Series/layers/SelfAttention_Family.py"
    )
    self_attention_legacy_source = _read_repo_file("transformer time series/SelfAttention_Family.py")
    timesnet_source = _read_repo_file("transformer time series/Time-Series/models/TimesNet.py")
    transformer_source = _read_repo_file("transformer time series/Time-Series/models/Transformer.py")

    assert "#int(params['" not in training_tft_source
    assert "#float(params['" not in training_tft_source
    assert "#json.loads(str(params['" not in training_tft_source
    assert "torch.Size(" not in custom_dataset_source
    assert "torch.Size(" not in embed_source
    assert "torch.Size(" not in embed_legacy_source
    assert "torch.Size(" not in tft_run_source
    assert "torch.Size(" not in transformer_encdec_source
    assert "torch.Size(" not in multi_wavelet_source
    assert "torch.Size(" not in self_attention_source
    assert "torch.Size(" not in self_attention_legacy_source
    assert "torch.Size(" not in masking_source
    assert "# [B, L, D]" not in autoformer_source
    assert "# [B, L, D]" not in crossformer_source
    assert "# [B, L, D]" not in dlinear_source
    assert "# [B, L, D]" not in etsformer_source
    assert "# [B, L, D]" not in fedformer_source
    assert "# [B, L, D]" not in informer_source
    assert "# [B, L, D]" not in lightts_source
    assert "# [B, L, D]" not in micn_source
    assert "# [B, L, D]" not in nonstationary_source
    assert "# [B, L, D]" not in patchtst_source
    assert "# [B, L, D]" not in pyraformer_source
    assert "# [B, L, D]" not in reformer_source
    assert "# [B, L, D]" not in timesnet_source
    assert "# [B, L, D]" not in transformer_source
    assert "# [B,T,C]" not in reformer_source
    assert "# [B,T,C]" not in timesnet_source
    assert "# [B, N]" not in autoformer_source
    assert "# [B, N]" not in crossformer_source
    assert "# [B, N]" not in dlinear_source
    assert "# [B, N]" not in etsformer_source
    assert "# [B, N]" not in fedformer_source
    assert "# [B, N]" not in informer_source
    assert "# [B, N]" not in lightts_source
    assert "# [B, N]" not in micn_source
    assert "# [B, N]" not in patchtst_source
    assert "# [B, N]" not in pyraformer_source
    assert "# [B, N]" not in reformer_source
    assert "# [B, N]" not in timesnet_source
    assert "# [B, N]" not in transformer_source


def test_fetch_rnn_paper_metadata_source_extracts_complexity_helpers() -> None:
    source = _read_repo_file("tools/fetch_rnn_paper_metadata.py")

    assert "def _iter_arxiv_queries(" in source
    assert "def _search_best_arxiv_hit(" in source
    assert "def _iter_crossref_queries(" in source
    assert "def _search_best_crossref_hit(" in source
    assert "def _crossref_hits_for_query(" in source
    assert "def _crossref_hit_score(" in source
    assert "def _ensure_metadata_entry(" in source
    assert "def _build_metadata_record(" in source
    assert "def _print_metadata_coverage(" in source
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_best_arxiv_match",
        "_iter_arxiv_queries",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_best_arxiv_match",
        "_search_best_arxiv_hit",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_best_crossref_match",
        "_iter_crossref_queries",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_best_crossref_match",
        "_search_best_crossref_hit",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_search_best_crossref_hit",
        "_crossref_hits_for_query",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_search_best_crossref_hit",
        "_crossref_hit_score",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "fetch_all",
        "_ensure_metadata_entry",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "fetch_all",
        "_refresh_metadata_record",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "fetch_all",
        "_print_metadata_coverage",
    )


def test_fetch_rnn_paper_metadata_source_extracts_fetch_all_helpers() -> None:
    source = _read_repo_file("tools/fetch_rnn_paper_metadata.py")

    assert "def _resolve_wanted_paper_defs(" in source
    assert "def _should_rebuild_metadata_record(" in source
    assert "def _print_metadata_progress_row(" in source
    assert "def _refresh_metadata_record(" in source
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "fetch_all",
        "_resolve_wanted_paper_defs",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "fetch_all",
        "_should_rebuild_metadata_record",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "fetch_all",
        "_refresh_metadata_record",
    )


def test_fetch_rnn_paper_metadata_source_extracts_entry_resolution_helpers() -> None:
    source = _read_repo_file("tools/fetch_rnn_paper_metadata.py")

    assert "def _resolve_entry_url_fields(" in source
    assert "def _expected_title_for_crossref(" in source
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_ensure_metadata_entry",
        "_resolve_entry_url_fields",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_build_metadata_record",
        "_expected_title_for_crossref",
    )


def test_fetch_rnn_paper_metadata_source_extracts_scoring_helpers() -> None:
    source = _read_repo_file("tools/fetch_rnn_paper_metadata.py")

    assert "def _score_year_delta(" in source
    assert "def _count_author_last_name_matches(" in source
    assert "def _arxiv_hint_title_score(" in source
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_score_arxiv_hit",
        "_score_year_delta",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_score_arxiv_hit",
        "_count_author_last_name_matches",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_score_arxiv_hit",
        "_arxiv_hint_title_score",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_score_crossref_hit",
        "_score_year_delta",
    )
    assert _function_uses_name(
        "tools/fetch_rnn_paper_metadata.py",
        "_score_crossref_hit",
        "_count_author_last_name_matches",
    )


def test_convert_ipynb_source_extracts_complexity_helpers() -> None:
    source = _read_repo_file("tools/convert_ipynb_to_py.py")

    assert "def _normalized_cell_source(" in source
    assert "def _normalized_cell_text(" in source
    assert "def _append_markdown_cell(" in source
    assert "def _append_code_cell(" in source
    assert "def _append_unknown_cell(" in source
    assert "def _append_cell(" in source
    assert _function_uses_name(
        "tools/convert_ipynb_to_py.py",
        "convert_one",
        "_normalized_cell_source",
    )
    assert _function_uses_name(
        "tools/convert_ipynb_to_py.py",
        "convert_one",
        "_append_cell",
    )
    assert _function_uses_name(
        "tools/convert_ipynb_to_py.py",
        "_append_cell",
        "_append_markdown_cell",
    )
    assert _function_uses_name(
        "tools/convert_ipynb_to_py.py",
        "_append_cell",
        "_append_code_cell",
    )
    assert _function_uses_name(
        "tools/convert_ipynb_to_py.py",
        "_append_cell",
        "_append_unknown_cell",
    )


def test_transformer_tools_source_extracts_adjustment_helpers() -> None:
    source = _read_repo_file("transformer time series/Time-Series/utils/tools.py")

    assert "def _starts_detected_anomaly_run(" in source
    assert "def _mark_anomaly_run(" in source
    assert _function_uses_name(
        "transformer time series/Time-Series/utils/tools.py",
        "adjustment",
        "_starts_detected_anomaly_run",
    )
    assert _function_uses_name(
        "transformer time series/Time-Series/utils/tools.py",
        "adjustment",
        "_mark_anomaly_run",
    )


def test_multiwavelet_correlation_source_uses_sonar_safe_names() -> None:
    source = _read_repo_file("transformer time series/Time-Series/layers/MultiWaveletCorrelation.py")

    assert "self.nCZ" not in source
    assert "self.MWT_CZ" not in source
    assert "B, seq_len, _, _ = queries.shape" not in source
    assert "_, source_steps, _, D = values.shape" not in source
    assert "V = self.Lk0(values).view(B, seq_len, self.c, -1)" not in source
    assert "V = self.Lk1(V.view(B, seq_len, -1))" not in source
    assert "ud = torch.jit.annotate(List[Tensor], [])" not in source
    assert "us = torch.jit.annotate(List[Tensor], [])" not in source
    assert "self.n_cz" in source
    assert "self.mwt_cz" in source


def test_multiwavelet_correlation_declared_names_avoid_selected_s117_patterns() -> None:
    declared = _declared_names("transformer time series/Time-Series/layers/MultiWaveletCorrelation.py")
    forbidden = {"B", "N", "S", "E", "H", "H0", "H1", "G0", "G1", "PHI0", "PHI1"}

    assert forbidden.isdisjoint(declared)


def test_self_attention_family_declared_names_avoid_selected_s117_patterns() -> None:
    forbidden = {"A", "B", "E", "H", "L", "S", "V"}
    for path in (
        "transformer time series/Time-Series/layers/SelfAttention_Family.py",
        "transformer time series/SelfAttention_Family.py",
    ):
        declared = _declared_names(path)
        assert forbidden.isdisjoint(declared)


def test_tft_models_declared_names_avoid_selected_s117_patterns() -> None:
    declared = _declared_names("transformer time series/tft/models.py")
    assert {"B", "T"}.isdisjoint(declared)


def test_autocorrelation_declared_names_avoid_selected_s117_patterns() -> None:
    declared = _declared_names("transformer time series/Time-Series/layers/AutoCorrelation.py")
    assert {"B", "H", "L", "S", "V"}.isdisjoint(declared)


def test_etsformer_encdec_declared_names_avoid_selected_s117_patterns() -> None:
    declared = _declared_names("transformer time series/Time-Series/layers/ETSformer_EncDec.py")
    assert {"M", "N"}.isdisjoint(declared)


def test_timeseries_run_declared_names_avoid_selected_s117_patterns() -> None:
    declared = _declared_names("transformer time series/Time-Series/run.py")
    assert {"Exp"}.isdisjoint(declared)


def test_lightts_declared_names_avoid_selected_s117_patterns() -> None:
    declared = _declared_names("transformer time series/Time-Series/models/LightTS.py")
    assert {"B", "N"}.isdisjoint(declared)


def test_timesnet_declared_names_avoid_selected_s117_patterns() -> None:
    declared = _declared_names("transformer time series/Time-Series/models/TimesNet.py")
    assert {"B", "N", "T"}.isdisjoint(declared)


def test_fourier_correlation_declared_names_avoid_selected_s117_patterns() -> None:
    declared = _declared_names("transformer time series/Time-Series/layers/FourierCorrelation.py")
    assert {"B", "E", "H", "L"}.isdisjoint(declared)


def test_exp_imputation_declared_names_avoid_selected_s117_patterns() -> None:
    declared = _declared_names("transformer time series/Time-Series/exp/exp_imputation.py")
    assert {"B", "N", "T"}.isdisjoint(declared)


def test_exp_short_term_declared_names_avoid_selected_s117_patterns() -> None:
    declared = _declared_names(
        "transformer time series/Time-Series/exp/exp_short_term_forecasting.py"
    )
    assert {"B", "C"}.isdisjoint(declared)


def test_short_term_forecasting_source_avoids_seq_len_inline_formula_comment() -> None:
    source = _read_repo_file(
        "transformer time series/Time-Series/exp/exp_short_term_forecasting.py"
    )
    assert "self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len" not in source


def test_tft_data_download_source_avoids_zip_extraction_comment_block() -> None:
    lines = _read_repo_file("transformer time series/tft/data_download.py").splitlines()

    assert not lines[21].lstrip().startswith("#")
    assert not lines[22].lstrip().startswith("#")


def test_transformer_sources_replace_selected_unused_bindings_with_underscores() -> None:
    anomaly_detection_source = _read_repo_file(
        "transformer time series/Time-Series/exp/exp_anomaly_detection.py"
    )
    anomaly_detection_lines = anomaly_detection_source.splitlines()
    autoformer_source = _read_repo_file("transformer time series/Time-Series/models/Autoformer.py")
    autocorrelation_source = _read_repo_file(
        "transformer time series/Time-Series/layers/AutoCorrelation.py"
    )
    etsformer_encdec_source = _read_repo_file(
        "transformer time series/Time-Series/layers/ETSformer_EncDec.py"
    )
    etsformer_encdec_lines = etsformer_encdec_source.splitlines()
    exp_imputation_source = _read_repo_file(
        "transformer time series/Time-Series/exp/exp_imputation.py"
    )
    exp_long_term_source = _read_repo_file(
        "transformer time series/Time-Series/exp/exp_long_term_forecasting.py"
    )
    exp_short_term_source = _read_repo_file(
        "transformer time series/Time-Series/exp/exp_short_term_forecasting.py"
    )
    classification_source = _read_repo_file(
        "transformer time series/Time-Series/exp/exp_classification.py"
    )
    classification_lines = classification_source.splitlines()
    fedformer_source = _read_repo_file("transformer time series/Time-Series/models/FEDformer.py")
    fourier_correlation_source = _read_repo_file(
        "transformer time series/Time-Series/layers/FourierCorrelation.py"
    )
    informer_source = _read_repo_file("transformer time series/Time-Series/models/Informer.py")
    lightts_source = _read_repo_file("transformer time series/Time-Series/models/LightTS.py")
    micn_source = _read_repo_file("transformer time series/Time-Series/models/MICN.py")
    nonstationary_source = _read_repo_file(
        "transformer time series/Time-Series/models/Nonstationary_Transformer.py"
    )
    reformer_source = _read_repo_file("transformer time series/Time-Series/models/Reformer.py")
    transformer_source = _read_repo_file("transformer time series/Time-Series/models/Transformer.py")
    crossformer_encdec_source = _read_repo_file(
        "transformer time series/Time-Series/layers/Crossformer_EncDec.py"
    )
    patchtst_source = _read_repo_file("transformer time series/Time-Series/models/PatchTST.py")
    self_attention_source = _read_repo_file(
        "transformer time series/Time-Series/layers/SelfAttention_Family.py"
    )
    self_attention_legacy_source = _read_repo_file("transformer time series/SelfAttention_Family.py")
    multi_wavelet_source = _read_repo_file(
        "transformer time series/Time-Series/layers/MultiWaveletCorrelation.py"
    )
    crossformer_source = _read_repo_file(
        "transformer time series/Time-Series/models/Crossformer.py"
    )
    training_tft_source = _read_repo_file("transformer time series/tft/training_tft.py")

    assert "train_data, train_loader = self._get_data(flag='train')" not in anomaly_detection_source
    assert "train_data, train_loader = self._get_data(flag='train')" not in anomaly_detection_source
    assert "precision, recall, f_score, support = precision_recall_fscore_support(" not in anomaly_detection_source
    assert anomaly_detection_lines[128].lstrip().startswith("_, test_loader = self._get_data(flag='test')")
    assert anomaly_detection_lines[129].lstrip().startswith("_, train_loader = self._get_data(flag='train')")

    assert ") for l in range(configs.e_layers)" not in autoformer_source
    assert "for l in range(configs.d_layers)" not in autoformer_source
    assert "enc_out, attns = self.encoder(enc_out, attn_mask=None)" not in autoformer_source

    assert "B, L, H, E = queries.shape" not in autocorrelation_source
    assert "_, S, _, D = values.shape" not in autocorrelation_source

    assert "train_data, train_loader = self._get_data(flag='TRAIN')" not in classification_source
    assert classification_lines[145].lstrip().startswith("_, test_loader = self._get_data(flag='TEST')")

    assert "train_data, train_loader = self._get_data(flag='train')" not in exp_imputation_source
    assert "_, train_loader = self._get_data(flag='train')" in exp_imputation_source
    assert exp_imputation_source.count("test_data, test_loader = self._get_data(flag='test')") == 1
    assert exp_imputation_source.count("_, test_loader = self._get_data(flag='test')") == 1

    assert "train_data, train_loader = self._get_data(flag='train')" not in exp_long_term_source
    assert "_, train_loader = self._get_data(flag='train')" in exp_long_term_source
    assert exp_long_term_source.count("test_data, test_loader = self._get_data(flag='test')") == 1
    assert exp_long_term_source.count("_, test_loader = self._get_data(flag='test')") == 1

    assert "train_data, train_loader = self._get_data(flag='train')" not in exp_short_term_source
    assert "vali_data, vali_loader = self._get_data(flag='val')" not in exp_short_term_source
    assert "mse = nn.MSELoss()" not in exp_short_term_source
    assert "loss_sharpness =" not in exp_short_term_source
    assert "_, train_loader = self._get_data(flag='train')" in exp_short_term_source
    assert "_, vali_loader = self._get_data(flag='val')" in exp_short_term_source

    assert ") for l in range(configs.e_layers)" not in fedformer_source
    assert "for l in range(configs.d_layers)" not in fedformer_source
    assert "enc_out, attns = self.encoder(enc_out, attn_mask=None)" not in fedformer_source

    assert "b, t, h, d = values.shape" not in etsformer_encdec_source
    assert "b, t, d = inputs.shape" not in etsformer_encdec_source
    assert "values, indices = torch.topk(" not in etsformer_encdec_source
    assert not etsformer_encdec_lines[143].lstrip().startswith("b, t, d = x.shape")

    assert ") for l in range(configs.e_layers)" not in informer_source
    assert ") for l in range(configs.e_layers - 1)" not in informer_source
    assert "for l in range(configs.d_layers)" not in informer_source
    assert "enc_out, attns = self.encoder(enc_out, attn_mask=None)" not in informer_source

    assert "xv = v.permute(0, 2, 3, 1)" not in fourier_correlation_source

    assert "B, T, N = x.size()" not in lightts_source
    assert "B, _, N = x.size()" not in lightts_source
    assert "batch_size, _, num_nodes = x.size()" in lightts_source

    assert "batch, seq_len, channel = input.shape" not in micn_source
    assert "src_out, trend1 = self.decomp[i](src)" not in micn_source
    assert "for i in range(d_layers)]" not in micn_source

    assert ") for l in range(configs.e_layers)" not in nonstationary_source
    assert "for l in range(configs.d_layers)" not in nonstationary_source
    assert "enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)" not in nonstationary_source

    assert ") for l in range(configs.e_layers)" not in reformer_source
    assert "enc_out, attns = self.encoder(enc_out, attn_mask=None)" not in reformer_source
    assert "enc_out, attns = self.encoder(enc_out)" not in reformer_source

    assert ") for l in range(configs.e_layers)" not in transformer_source
    assert "for l in range(configs.d_layers)" not in transformer_source
    assert "enc_out, attns = self.encoder(enc_out, attn_mask=None)" not in transformer_source

    assert "batch_size, ts_d, seg_num, d_model = x.shape" not in crossformer_encdec_source
    assert "for i in range(depth):" not in crossformer_encdec_source
    assert "_, ts_dim, _, _ = x.shape" not in crossformer_encdec_source
    assert "x, attns = block(x)" not in crossformer_encdec_source
    assert "tmp, attn = self.cross_attention(" not in crossformer_encdec_source

    assert ") for l in range(configs.e_layers)" not in patchtst_source
    assert "enc_out, attns = self.encoder(enc_out)" not in patchtst_source

    assert "B, L, H, E = queries.shape" not in self_attention_source
    assert "_, S, _, D = values.shape" not in self_attention_source
    assert "dim_buffer, attn = self.dim_sender(" not in self_attention_source
    assert "dim_receive, attn = self.dim_receiver(" not in self_attention_source
    assert "B, N, C = queries.shape" not in self_attention_source
    assert "time_enc, attn = self.time_attention(" not in self_attention_source

    assert "B, L, H, E = queries.shape" not in self_attention_legacy_source
    assert "_, S, _, D = values.shape" not in self_attention_legacy_source
    assert "B, H, L_V, D = V.shape" not in self_attention_legacy_source
    assert "B, L_Q, H, D = queries.shape" not in self_attention_legacy_source
    assert "dim_buffer, attn = self.dim_sender(" not in self_attention_legacy_source
    assert "dim_receive, attn = self.dim_receiver(" not in self_attention_legacy_source
    assert "B, N, C = queries.shape" not in self_attention_legacy_source
    assert "time_enc, attn = self.time_attention(" not in self_attention_legacy_source

    assert "B, L, H, E = queries.shape" not in multi_wavelet_source
    assert "B, N, H, E = q.shape" not in multi_wavelet_source
    assert "B, N, c, k = x.shape  # (B, N, k)" not in multi_wavelet_source
    assert "_, S, _, D = values.shape" not in multi_wavelet_source
    assert multi_wavelet_source.count("for i in range(ns - self.L):") == 1

    assert ") for l in range(configs.e_layers)" not in crossformer_source
    assert "for l in range(configs.e_layers + 1)" not in crossformer_source
    assert "enc_out, attns = self.encoder(" not in crossformer_source

    assert "static_encoder, sparse_weights = self.static_vsn(static_inputs)" not in training_tft_source
    assert "historical_features, historical_flags \\" not in training_tft_source
    assert "future_features, future_flags \\" not in training_tft_source
    assert "x, self_att = self.self_attn_layer(enriched," not in training_tft_source


def test_transformer_sources_drop_selected_unused_imports() -> None:
    autocorrelation_source = _read_repo_file(
        "transformer time series/Time-Series/layers/AutoCorrelation.py"
    )
    embed_source = _read_repo_file("transformer time series/Time-Series/layers/Embed.py")
    embed_legacy_source = _read_repo_file("transformer time series/Embed.py")
    etsformer_encdec_source = _read_repo_file(
        "transformer time series/Time-Series/layers/ETSformer_EncDec.py"
    )
    training_tft_source = _read_repo_file("transformer time series/tft/training_tft.py")
    tft_run_source = _read_repo_file("transformer time series/tft/run.py")
    crossformer_source = _read_repo_file("transformer time series/Time-Series/models/Crossformer.py")
    crossformer_encdec_source = _read_repo_file(
        "transformer time series/Time-Series/layers/Crossformer_EncDec.py"
    )
    autoformer_source = _read_repo_file("transformer time series/Time-Series/models/Autoformer.py")
    transformer_source = _read_repo_file("transformer time series/Time-Series/models/Transformer.py")
    transformer_encdec_source = _read_repo_file(
        "transformer time series/Time-Series/layers/Transformer_EncDec.py"
    )

    assert "import torch.nn.functional as F" not in autocorrelation_source
    assert "import matplotlib.pyplot as plt" not in autocorrelation_source
    assert "import numpy as np" not in autocorrelation_source
    assert "from math import sqrt" not in autocorrelation_source
    assert "import os" not in autocorrelation_source
    assert "import torch.nn.functional as F" not in embed_source
    assert "from torch.nn.utils import weight_norm" not in embed_source
    assert "import torch.nn.functional as F" not in embed_legacy_source
    assert "from torch.nn.utils import weight_norm" not in embed_legacy_source

    assert "import math, random" not in etsformer_encdec_source

    assert "import numpy as np" not in training_tft_source
    assert "import pyunpack" not in training_tft_source
    assert "import math" not in training_tft_source
    assert "from models import GatedLinearUnit" not in training_tft_source
    assert "import torch.nn.functional as F" not in training_tft_source
    assert "from torch.utils.data import Dataset, DataLoader" not in training_tft_source

    assert "import numpy as np" not in tft_run_source
    assert "import pyunpack" not in tft_run_source
    assert "import math" not in tft_run_source
    assert "from data.data_download import Config, download_electricity" not in tft_run_source
    assert "from models import GatedLinearUnit" not in tft_run_source
    assert "from models import ScaledDotProductAttention" not in tft_run_source
    assert "import torch.nn.functional as F" not in tft_run_source
    assert "from torch.utils.data import Dataset, DataLoader" not in tft_run_source
    assert "import matplotlib.pyplot as plt" not in tft_run_source
    assert "import pickle" not in tft_run_source

    assert "import torch.nn.functional as F" not in crossformer_source
    assert "from einops import rearrange, repeat" not in crossformer_encdec_source
    assert "from layers.Embed import DataEmbedding, DataEmbedding_wo_pos" not in autoformer_source
    assert "import math" not in autoformer_source
    assert "import numpy as np" not in autoformer_source
    assert "from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer" not in transformer_source
    assert "import numpy as np" not in transformer_source
    assert transformer_encdec_source.count("\n    import torch\n") == 0


def test_selected_transformer_sources_pass_pyflakes() -> None:
    paths = [
        "transformer time series/Time-Series/exp/exp_anomaly_detection.py",
        "transformer time series/Time-Series/exp/exp_classification.py",
        "transformer time series/Time-Series/exp/exp_imputation.py",
        "transformer time series/Time-Series/exp/exp_long_term_forecasting.py",
        "transformer time series/Time-Series/exp/exp_short_term_forecasting.py",
        "transformer time series/Time-Series/layers/Embed.py",
        "transformer time series/Embed.py",
        "transformer time series/Time-Series/layers/FourierCorrelation.py",
        "transformer time series/Time-Series/layers/MultiWaveletCorrelation.py",
        "transformer time series/Time-Series/layers/Transformer_EncDec.py",
        "transformer time series/Time-Series/models/LightTS.py",
    ]
    assert _pyflakes_messages(paths) == []


def test_selected_examples_and_cli_sources_avoid_current_low_risk_sonar_patterns() -> None:
    run_benchmarks_source = _read_repo_file("benchmarks/run_benchmarks.py")
    rnn_paper_zoo_source = _read_repo_file("examples/rnn_paper_zoo.py")
    torch_global_models_source = _read_repo_file("examples/torch_global_models.py")
    cli_leaderboard_source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert 'list(config.get("conformal_levels", []))' not in run_benchmarks_source
    assert "common = dict(" not in rnn_paper_zoo_source
    assert torch_global_models_source.count('"0.1,0.5,0.9"') <= 1
    assert "if False else" not in cli_leaderboard_source


def test_ml_time_series_sources_make_merge_contracts_explicit() -> None:
    prophet_path = "ml time series/prophet.py"
    cashflow_path = "ml time series/现金流预测模型开发.py"

    assert _call_lines_missing_keyword(prophet_path, method_name="merge", keyword="on") == []
    assert _call_lines_missing_keyword(prophet_path, method_name="merge", keyword="validate") == []

    assert _call_lines_missing_keyword(cashflow_path, method_name="merge", keyword="how") == []
    assert _call_lines_missing_keyword(cashflow_path, method_name="merge", keyword="on") == []
    assert _call_lines_missing_keyword(cashflow_path, method_name="merge", keyword="validate") == []


def test_selected_transformer_sources_remove_current_public_commented_code() -> None:
    fourier_source = _read_repo_file("transformer time series/Time-Series/layers/FourierCorrelation.py")
    multiwavelet_source = _read_repo_file(
        "transformer time series/Time-Series/layers/MultiWaveletCorrelation.py"
    )
    run_source = _read_repo_file("transformer time series/Time-Series/run.py")

    assert "# size = [B, H, E, L]" not in fourier_source
    assert "# size = [B, H, E, L]" not in multiwavelet_source
    assert not _line_at(
        "transformer time series/Time-Series/layers/Transformer_EncDec.py",
        5,
    ).lstrip().startswith("#")
    assert not _line_at(
        "transformer time series/Time-Series/layers/Transformer_EncDec.py",
        6,
    ).lstrip().startswith("#")
    assert "# type1:" not in run_source
    assert "# type2:" not in run_source
    assert "# type3:" not in run_source
    assert "#M4" not in run_source
    assert "传感器读数" not in run_source
    assert not _line_at(
        "transformer time series/Time-Series/utils/timefeatures.py",
        27,
    ).lstrip().startswith("#")
    assert not _line_at(
        "transformer time series/Time-Series/utils/timefeatures.py",
        28,
    ).lstrip().startswith("#")
    assert not _line_at(
        "transformer time series/tft/data/data_download.py",
        22,
    ).lstrip().startswith("#")
    assert not _line_at(
        "transformer time series/tft/data/data_download.py",
        23,
    ).lstrip().startswith("#")


def test_selected_sources_avoid_current_public_empty_method_bodies() -> None:
    timefeatures_source = _read_repo_file("transformer time series/Time-Series/utils/timefeatures.py")
    exp_basic_source = _read_repo_file("transformer time series/Time-Series/exp/exp_basic.py")
    sonar_coverage_source = _read_repo_file("tests/test_sonar_coverage_recent_fixes.py")

    assert "def __init__(self):\n        pass" not in timefeatures_source
    assert "def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:\n        pass" not in timefeatures_source

    assert "def _get_data(self):\n        pass" not in exp_basic_source
    assert "def vali(self):\n        pass" not in exp_basic_source
    assert "def train(self):\n        pass" not in exp_basic_source
    assert "def test(self):\n        pass" not in exp_basic_source

    assert (
        sonar_coverage_source.count("def __init__(self, **_: object) -> None:\n            pass")
        == 0
    )


def test_selected_sources_extract_current_public_repeated_literals_to_constants() -> None:
    anomaly_source = _read_repo_file("statistics time series/Anomaly Detection.py")
    garch_source = _read_repo_file("statistics time series/GARCH Model.py")
    sarima_source = _read_repo_file("statistics time series/SARIMA Model.py")
    undo_stationary_source = _read_repo_file(
        "statistics time series/Undo Stationary Transformations.py"
    )
    var_source = _read_repo_file("statistics time series/VAR Model.py")
    data_loader_source = _read_repo_file(
        "transformer time series/Time-Series/data_provider/data_loader.py"
    )
    etsformer_encdec_source = _read_repo_file(
        "transformer time series/Time-Series/layers/ETSformer_EncDec.py"
    )
    crossformer_source = _read_repo_file("transformer time series/Time-Series/models/Crossformer.py")

    assert anomaly_source.count("Catfish Sales in 1000s of Pounds") <= 1
    assert garch_source.count("True Volatility") <= 1
    assert garch_source.count("Predicted Volatility") <= 1
    assert sarima_source.count("Catfish Sales in 1000s of Pounds") <= 1
    assert undo_stationary_source.count("Hours Since Published") <= 1
    assert var_source.count("'ice cream'") <= 1
    assert var_source.count("'Ice Cream'") <= 1
    assert var_source.count("'First Difference'") <= 1
    assert data_loader_source.count("ETTh1.csv") <= 1
    assert data_loader_source.count("test:") <= 1
    assert data_loader_source.count("train:") <= 1
    assert etsformer_encdec_source.count("b f d -> b f () d") <= 1
    assert crossformer_source.count("(b d) seg_num d_model -> b d seg_num d_model") <= 1


def test_sonar_coverage_recent_fixes_test_uses_precise_callable_annotations() -> None:
    source = _read_repo_file("tests/test_sonar_coverage_recent_fixes.py")

    assert "fit_model: object" not in source
    assert "installer: object" not in source
    assert "factory: object" not in source


def test_transformer_experiment_overrides_keep_added_parameters_optional() -> None:
    expectations = {
        (
            "transformer time series/Time-Series/exp/exp_anomaly_detection.py",
            "ExpAnomalyDetection",
            "_get_data",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_anomaly_detection.py",
            "ExpAnomalyDetection",
            "vali",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_anomaly_detection.py",
            "ExpAnomalyDetection",
            "train",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_anomaly_detection.py",
            "ExpAnomalyDetection",
            "test",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_classification.py",
            "ExpClassification",
            "_get_data",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_classification.py",
            "ExpClassification",
            "vali",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_classification.py",
            "ExpClassification",
            "train",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_classification.py",
            "ExpClassification",
            "test",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_imputation.py",
            "ExpImputation",
            "_get_data",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_imputation.py",
            "ExpImputation",
            "vali",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_imputation.py",
            "ExpImputation",
            "train",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_imputation.py",
            "ExpImputation",
            "test",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_long_term_forecasting.py",
            "ExpLongTermForecast",
            "_get_data",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_long_term_forecasting.py",
            "ExpLongTermForecast",
            "vali",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_long_term_forecasting.py",
            "ExpLongTermForecast",
            "train",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_long_term_forecasting.py",
            "ExpLongTermForecast",
            "test",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_short_term_forecasting.py",
            "ExpShortTermForecast",
            "_get_data",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_short_term_forecasting.py",
            "ExpShortTermForecast",
            "vali",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_short_term_forecasting.py",
            "ExpShortTermForecast",
            "train",
        ): [],
        (
            "transformer time series/Time-Series/exp/exp_short_term_forecasting.py",
            "ExpShortTermForecast",
            "test",
        ): [],
    }

    for key, expected in expectations.items():
        path, class_name, method_name = key
        assert _required_method_param_names(path, class_name, method_name) == expected


def test_selected_transformer_sources_use_current_lowercase_helper_names() -> None:
    multiwavelet_source = _read_repo_file(
        "transformer time series/Time-Series/layers/MultiWaveletCorrelation.py"
    )
    timesnet_source = _read_repo_file("transformer time series/Time-Series/models/TimesNet.py")
    metrics_path = "transformer time series/Time-Series/utils/metrics.py"
    metrics_source = _read_repo_file(metrics_path)

    assert "def legendreDer(" not in multiwavelet_source
    assert "def legendre_der(" in multiwavelet_source
    assert "legendreDer(" not in multiwavelet_source
    assert "legendre_der(" in multiwavelet_source

    assert "def FFT_for_Period(" not in timesnet_source
    assert "def fft_for_period(" in timesnet_source
    assert "FFT_for_Period(" not in timesnet_source
    assert "fft_for_period(" in timesnet_source

    for old_name, new_name in (
        ("RSE", "rse"),
        ("CORR", "corr"),
        ("MAE", "mean_absolute_error"),
        ("MSE", "mean_squared_error"),
        ("RMSE", "root_mean_squared_error"),
        ("MAPE", "mean_absolute_percentage_error"),
        ("MSPE", "mean_squared_percentage_error"),
    ):
        assert f"def {old_name}(" not in metrics_source
        assert f"def {new_name}(" in metrics_source

    assert _function_uses_name(metrics_path, "metric", "mean_absolute_error")
    assert _function_uses_name(metrics_path, "metric", "mean_squared_error")
    assert _function_uses_name(metrics_path, "metric", "root_mean_squared_error")
    assert _function_uses_name(metrics_path, "metric", "mean_absolute_percentage_error")
    assert _function_uses_name(metrics_path, "metric", "mean_squared_percentage_error")


def test_selected_transformer_sources_use_s101_safe_class_names_with_legacy_aliases() -> None:
    expectations = {
        "transformer time series/Embed.py": [("DataEmbedding_wo_pos", "DataEmbeddingWoPos")],
        "transformer time series/Time-Series/layers/Embed.py": [
            ("DataEmbedding_wo_pos", "DataEmbeddingWoPos")
        ],
        "transformer time series/Time-Series/data_provider/data_loader.py": [
            ("Dataset_ETT_hour", "DatasetEttHour"),
            ("Dataset_ETT_minute", "DatasetEttMinute"),
            ("Dataset_Custom", "DatasetCustom"),
            ("Dataset_M4", "DatasetM4"),
        ],
        "transformer time series/Time-Series/exp/exp_basic.py": [("Exp_Basic", "ExpBasic")],
        "transformer time series/Time-Series/exp/exp_anomaly_detection.py": [
            ("Exp_Anomaly_Detection", "ExpAnomalyDetection")
        ],
        "transformer time series/Time-Series/exp/exp_classification.py": [
            ("Exp_Classification", "ExpClassification")
        ],
        "transformer time series/Time-Series/exp/exp_imputation.py": [
            ("Exp_Imputation", "ExpImputation")
        ],
        "transformer time series/Time-Series/exp/exp_long_term_forecasting.py": [
            ("Exp_Long_Term_Forecast", "ExpLongTermForecast")
        ],
        "transformer time series/Time-Series/exp/exp_short_term_forecasting.py": [
            ("Exp_Short_Term_Forecast", "ExpShortTermForecast")
        ],
        "transformer time series/Time-Series/layers/Autoformer_EncDec.py": [
            ("my_Layernorm", "MyLayerNorm")
        ],
        "transformer time series/Time-Series/layers/Conv_Blocks.py": [
            ("Inception_Block_V1", "InceptionBlockV1"),
            ("Inception_Block_V2", "InceptionBlockV2"),
        ],
        "transformer time series/Time-Series/layers/MultiWaveletCorrelation.py": [
            ("sparseKernelFT1d", "SparseKernelFt1d"),
            ("MWT_CZ1d", "MwtCz1d"),
        ],
        "transformer time series/Time-Series/layers/Pyraformer_EncDec.py": [
            ("Bottleneck_Construct", "BottleneckConstruct")
        ],
    }

    for path, alias_pairs in expectations.items():
        class_names = _top_level_class_names(path)
        name_aliases = _top_level_name_aliases(path)
        for old_name, new_name in alias_pairs:
            assert new_name in class_names
            assert old_name not in class_names
            assert (old_name, new_name) in name_aliases


def test_selected_transformer_sources_avoid_public_dataframe_values_calls() -> None:
    data_loader_source = _read_repo_file(
        "transformer time series/Time-Series/data_provider/data_loader.py"
    )
    m4_summary_source = _read_repo_file("transformer time series/Time-Series/utils/m4_summary.py")

    assert "data = data.values[:, 1:]" not in data_loader_source
    assert "test_data = test_data.values[:, 1:]" not in data_loader_source
    assert "pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]" not in data_loader_source
    assert "labels = test_data.values[:, -1:]" not in data_loader_source
    assert "train_data = train_data.values[:, :-1]" not in data_loader_source
    assert "test_data = test_data.values[:, :-1]" not in data_loader_source

    assert "pd.read_csv(self.naive_path).values[:, 1:]" not in m4_summary_source
    assert "model_forecast = pd.read_csv(file_name).values" not in m4_summary_source


def test_selected_tft_sources_avoid_public_unused_local_names() -> None:
    volatility_source = _read_repo_file("transformer time series/tft/data_formatters/volatility.py")
    models_source = _read_repo_file("transformer time series/tft/models.py")
    run_source = _read_repo_file("transformer time series/tft/run.py")
    training_source = _read_repo_file("transformer time series/tft/training_tft.py")

    assert "fixed_params = {" not in volatility_source
    assert "for i in range(self.output_size)])" not in models_source
    assert "for i in range(self.num_regular_variables)])" not in run_source
    assert "for i in range(self.num_regular_variables)])" not in training_source
    assert "sparse_weights" not in run_source
    assert "historical_flags" not in run_source
    assert "future_flags" not in run_source
    assert "x, self_att = self.self_attn_layer(" not in run_source


def test_selected_example_scripts_avoid_bare_no_effect_expr_statements() -> None:
    for path in (
        "ml time series/prophet.py",
        "ml time series/现金流预测模型开发.py",
        "statistics time series/ARMA Model.py",
        "statistics time series/Anomaly Detection.py",
        "statistics time series/Model Selection.py",
        "statistics time series/SARIMA Model.py",
        "statistics time series/STL Decomposition.py",
        "statistics time series/Time Series Data Preprocessing.py",
        "statistics time series/Time Series Data.py",
        "statistics time series/VAR Model.py",
    ):
        assert _bare_no_effect_expr_lines(path) == []


def test_tft_sources_avoid_bare_no_effect_expr_statements() -> None:
    for path in (
        "transformer time series/tft/training_tft.py",
        "transformer time series/tft/run.py",
    ):
        assert _bare_no_effect_expr_lines(path) == []


def test_tft_sources_use_dim_keyword_for_torch_tensor_joins() -> None:
    for path in (
        "transformer time series/tft/training_tft.py",
        "transformer time series/tft/run.py",
        "transformer time series/tft/models.py",
    ):
        assert _call_lines_with_keyword_alias(
            path,
            owner_name="torch",
            method_name="stack",
            keyword="axis",
        ) == []
        assert _call_lines_with_keyword_alias(
            path,
            owner_name="torch",
            method_name="cat",
            keyword="axis",
        ) == []


def test_transformer_sources_remove_selected_static_bug_patterns() -> None:
    exp_basic_source = _read_repo_file("transformer time series/Time-Series/exp/exp_basic.py")
    self_attention_source = _read_repo_file(
        "transformer time series/Time-Series/layers/SelfAttention_Family.py"
    )
    self_attention_legacy_source = _read_repo_file("transformer time series/SelfAttention_Family.py")
    losses_source = _read_repo_file("transformer time series/Time-Series/utils/losses.py")

    assert "raise NotImplementedError\n        return None" not in exp_basic_source
    assert "attn_mask = ProbMask(" not in self_attention_source
    assert "attn_mask = ProbMask(" not in self_attention_legacy_source
    assert "result[result != result]" not in losses_source
    assert "t.isnan(result)" in losses_source


def test_volatility_formatter_exposes_fixed_params_as_class_method() -> None:
    assert "get_fixed_params" in _class_method_names(
        "transformer time series/tft/data_formatters/volatility.py",
        "VolatilityFormatter",
    )


def test_selected_transformer_sources_avoid_uppercase_s117_identifiers() -> None:
    expected_missing = {
        "transformer time series/Time-Series/layers/Embed.py": {"Embed"},
        "transformer time series/Embed.py": {"Embed"},
        "transformer time series/Time-Series/data_provider/data_factory.py": {"Data"},
        "transformer time series/Time-Series/layers/ETSformer_EncDec.py": {
            "F_f",
            "F_g",
            "F_fg",
            "T",
        },
        "transformer time series/Time-Series/layers/MultiWaveletCorrelation.py": {
            "kUse",
            "L",
            "nCZ",
            "H0r",
            "G0r",
            "H1r",
            "G1r",
            "Ud_q",
            "Ud_k",
            "Ud_v",
            "Us_q",
            "Us_k",
            "Us_v",
            "Ud",
            "Us",
        },
        "transformer time series/Time-Series/utils/masking.py": {"B", "H", "L"},
        "transformer time series/Time-Series/layers/SelfAttention_Family.py": {
            "K",
            "Q",
            "K_expand",
            "K_sample",
            "Q_K_sample",
            "M_top",
            "Q_reduce",
            "L_Q",
            "V_sum",
            "U_part",
        },
        "transformer time series/SelfAttention_Family.py": {
            "K",
            "Q",
            "K_expand",
            "K_sample",
            "Q_K_sample",
            "M_top",
            "Q_reduce",
            "L_Q",
            "V_sum",
            "U_part",
        },
    }

    for path, names in expected_missing.items():
        assert _declared_names(path).isdisjoint(names)


def test_check_architecture_imports_source_extracts_import_helpers() -> None:
    source = _read_repo_file("tools/check_architecture_imports.py")

    assert "def _append_import_entries(" in source
    assert "def _append_import_from_entries(" in source
    assert _function_uses_name(
        "tools/check_architecture_imports.py",
        "_imports_for",
        "_append_import_entries",
    )
    assert _function_uses_name(
        "tools/check_architecture_imports.py",
        "_imports_for",
        "_append_import_from_entries",
    )


def test_check_architecture_imports_source_extracts_main_helpers() -> None:
    source = _read_repo_file("tools/check_architecture_imports.py")

    assert "def _run_architecture_checks(" in source
    assert "def _print_architecture_violations(" in source
    assert _function_uses_name(
        "tools/check_architecture_imports.py",
        "main",
        "_run_architecture_checks",
    )
    assert _function_uses_name(
        "tools/check_architecture_imports.py",
        "main",
        "_print_architecture_violations",
    )


def test_torch_local_catalog_source_extracts_variant_registration_helpers() -> None:
    source = _read_repo_file("src/foresight/models/catalog/torch_local.py")

    assert "def _register_local_xformer_specs(" in source
    assert "def _register_global_xformer_specs(" in source
    assert _function_uses_name(
        "src/foresight/models/catalog/torch_local.py",
        "_make_torch_dl_variant_specs",
        "_register_local_xformer_specs",
    )
    assert _function_uses_name(
        "src/foresight/models/catalog/torch_local.py",
        "_make_torch_dl_variant_specs",
        "_register_global_xformer_specs",
    )


def test_run_benchmarks_source_extracts_benchmark_case_helpers() -> None:
    source = _read_repo_file("benchmarks/run_benchmarks.py")

    assert "def _benchmark_dataset_case_fields(" in source
    assert "def _benchmark_result_row(" in source
    assert _function_uses_name(
        "benchmarks/run_benchmarks.py",
        "_get_cli_shared_module",
        "get_cli_shared_module",
    )
    assert _function_uses_name(
        "benchmarks/run_benchmarks.py",
        "_get_batch_execution_module",
        "get_batch_execution_module",
    )
    assert _function_uses_name(
        "benchmarks/run_benchmarks.py",
        "_resolve_benchmark_chunk_size",
        "_get_batch_execution_module",
    )
    assert _function_uses_name(
        "benchmarks/run_benchmarks.py",
        "run_benchmark_suite",
        "_benchmark_dataset_case_fields",
    )
    assert _function_uses_name(
        "benchmarks/run_benchmarks.py",
        "run_benchmark_suite",
        "_benchmark_result_row",
    )


def test_exp_long_term_forecasting_source_extracts_sequence_helpers() -> None:
    path = "transformer time series/Time-Series/exp/exp_long_term_forecasting.py"
    source = _read_repo_file(path)

    assert "def _long_term_decoder_input(" in source
    assert "def _long_term_model_outputs(" in source
    assert "def _long_term_prediction_pair(" in source
    assert _function_uses_name(path, "_long_term_model_outputs", "_long_term_decoder_input")
    assert _function_uses_name(path, "vali", "_long_term_model_outputs")
    assert _function_uses_name(path, "vali", "_long_term_prediction_pair")
    assert _function_uses_name(path, "train", "_long_term_model_outputs")
    assert _function_uses_name(path, "train", "_long_term_prediction_pair")
    assert _function_uses_name(path, "test", "_long_term_model_outputs")
    assert _function_uses_name(path, "test", "_long_term_prediction_pair")


def test_docsgen_rnn_source_extracts_metadata_loading_helpers() -> None:
    path = "src/foresight/docsgen/rnn.py"
    source = _read_repo_file(path)

    assert "def _rnn_paper_metadata_candidate_paths(" in source
    assert "def _normalize_rnn_paper_metadata(" in source
    assert _function_uses_name(path, "_read_rnn_paper_metadata", "_rnn_paper_metadata_candidate_paths")
    assert _function_uses_name(path, "_read_rnn_paper_metadata", "_normalize_rnn_paper_metadata")


def test_docsgen_rnn_source_extracts_doc_row_helpers() -> None:
    path = "src/foresight/docsgen/rnn.py"
    source = _read_repo_file(path)

    assert "def _metadata_entry_fields(" in source
    assert "def _rnn_paper_zoo_index_row(" in source
    assert "def _rnnzoo_base_index_row(" in source
    assert "def _rnnzoo_variant_index_row(" in source
    assert "def _rnnzoo_model_index_row(" in source
    assert _function_uses_name(path, "render_rnn_paper_zoo_doc", "_rnn_paper_zoo_index_row")
    assert _function_uses_name(path, "render_rnn_zoo_doc", "_rnnzoo_base_index_row")
    assert _function_uses_name(path, "render_rnn_zoo_doc", "_rnnzoo_variant_index_row")
    assert _function_uses_name(path, "render_rnn_zoo_doc", "_rnnzoo_model_index_row")


def test_cli_leaderboard_summarize_source_extracts_input_parsing_helpers() -> None:
    source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert "def _read_leaderboard_summarize_input_text(" in source
    assert "def _detect_leaderboard_summarize_input_format(" in source
    assert "def _parse_leaderboard_summarize_rows(" in source
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_cmd_leaderboard_summarize",
        "_read_leaderboard_summarize_input_text",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_cmd_leaderboard_summarize",
        "_detect_leaderboard_summarize_input_format",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_cmd_leaderboard_summarize",
        "_parse_leaderboard_summarize_rows",
    )


def test_cli_catalog_source_extracts_papers_and_docs_helpers() -> None:
    source = _read_repo_file("src/foresight/cli_catalog.py")

    assert "def _paper_payload_for_paper_id(" in source
    assert "def _paper_model_matches_role(" in source
    assert "def _paper_model_rows_for_spec(" in source
    assert "def _paper_matching_rows_for_spec(" in source
    assert "def _rnn_doc_check_failures(" in source
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_papers_list",
        "_paper_payload_for_paper_id",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_papers_info",
        "_paper_payload_for_paper_id",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_papers_models",
        "_paper_matching_rows_for_spec",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_paper_matching_rows_for_spec",
        "_paper_model_matches_role",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_paper_matching_rows_for_spec",
        "_paper_model_rows_for_spec",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_docs_rnn",
        "_rnn_doc_check_failures",
    )


def test_cli_catalog_source_extracts_models_list_helpers() -> None:
    source = _read_repo_file("src/foresight/cli_catalog.py")

    assert "def _validated_model_interface_filter(" in source
    assert "def _matches_required_model_requires(" in source
    assert "def _matches_excluded_model_requires(" in source
    assert "def _model_spec_matches_filters(" in source
    assert "def _catalog_model_row_from_spec(" in source
    assert "def _sort_catalog_model_rows(" in source
    assert "def _models_list_tsv_lines(" in source
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_models_list",
        "_validated_model_interface_filter",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_models_list",
        "_model_spec_matches_filters",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_model_spec_matches_filters",
        "_matches_required_model_requires",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_model_spec_matches_filters",
        "_matches_excluded_model_requires",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_models_list",
        "_catalog_model_row_from_spec",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_models_list",
        "_sort_catalog_model_rows",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_models_list",
        "_models_list_tsv_lines",
    )


def test_cli_catalog_source_extracts_models_search_helpers() -> None:
    source = _read_repo_file("src/foresight/cli_catalog.py")

    assert "def _score_model_search_token(" in source
    assert "def _model_search_metadata(" in source
    assert "def _model_search_row(" in source
    assert "def _models_search_tsv_lines(" in source
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_models_search",
        "_score_model_search_token",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_models_search",
        "_model_search_metadata",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_models_search",
        "_model_search_row",
    )
    assert _function_uses_name(
        "src/foresight/cli_catalog.py",
        "_cmd_models_search",
        "_models_search_tsv_lines",
    )


def test_cli_leaderboard_source_extracts_parallel_task_helpers() -> None:
    source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert "def _get_batch_execution_module(" in source
    assert "def _parallel_task_label(" in source
    assert "def _record_parallel_task_errors(" in source
    assert "def _run_parallel_tasks_sequential(" in source
    assert "def _build_parallel_task_executor(" in source
    assert "def _resolve_parallel_task_result(" in source
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_get_batch_execution_module",
        "get_batch_execution_module",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_resolve_leaderboard_chunk_size",
        "_get_batch_execution_module",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_run_parallel_tasks",
        "_parallel_task_label",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_run_parallel_tasks",
        "_run_parallel_tasks_sequential",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_run_parallel_tasks",
        "_build_parallel_task_executor",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_run_parallel_tasks",
        "_resolve_parallel_task_result",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_build_parallel_task_executor",
        "_get_batch_execution_module",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_resolve_parallel_task_result",
        "_get_batch_execution_module",
    )


def test_cli_shared_source_extracts_table_text_helper() -> None:
    source = _read_repo_file("src/foresight/cli_shared.py")

    assert "def _table_text(" in source
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_emit_table",
        "_table_text",
    )
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_write_table",
        "_table_text",
    )


def test_cli_shared_source_extracts_rendered_output_helpers() -> None:
    source = _read_repo_file("src/foresight/cli_shared.py")

    assert "def _write_rendered(" in source
    assert "def _emit_rendered(" in source
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_write_lines",
        "_write_rendered",
    )
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_write_table",
        "_write_rendered",
    )
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_emit",
        "_emit_rendered",
    )
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_emit_table",
        "_emit_rendered",
    )


def test_cli_shared_source_extracts_dataframe_text_helper() -> None:
    source = _read_repo_file("src/foresight/cli_shared.py")

    assert "def _dataframe_text(" in source
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_emit_dataframe",
        "_dataframe_text",
    )


def test_cli_shared_source_extracts_format_rows_helper() -> None:
    source = _read_repo_file("src/foresight/cli_shared.py")

    assert "def _format_rows(" in source
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_format_payload",
        "_format_rows",
    )
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_format_table",
        "_format_rows",
    )


def test_cli_shared_source_extracts_resolved_columns_helper() -> None:
    source = _read_repo_file("src/foresight/cli_shared.py")

    assert "def _resolved_columns(" in source
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_format_csv",
        "_resolved_columns",
    )
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_format_markdown",
        "_resolved_columns",
    )


def test_cli_shared_source_extracts_param_assignment_helper() -> None:
    source = _read_repo_file("src/foresight/cli_shared.py")

    assert "def _parse_param_assignment(" in source
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_parse_model_params",
        "_parse_param_assignment",
    )
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_parse_grid_params",
        "_parse_param_assignment",
    )


def test_cli_shared_source_extracts_json_text_helper() -> None:
    source = _read_repo_file("src/foresight/cli_shared.py")

    assert "def _json_text(" in source
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_format_payload",
        "_json_text",
    )
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_format_rows",
        "_json_text",
    )


def test_cli_shared_source_extracts_markdown_cell_text_helper() -> None:
    source = _read_repo_file("src/foresight/cli_shared.py")

    assert "def _markdown_cell_text(" in source
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_format_markdown",
        "_markdown_cell_text",
    )


def test_cli_shared_source_extracts_row_values_helper() -> None:
    source = _read_repo_file("src/foresight/cli_shared.py")

    assert "def _row_values(" in source
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_format_csv",
        "_row_values",
    )
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_format_markdown",
        "_row_values",
    )


def test_cli_shared_source_extracts_split_csv_items_helper() -> None:
    source = _read_repo_file("src/foresight/cli_shared.py")

    assert "def _split_csv_items(" in source
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_parse_requires_filter",
        "_split_csv_items",
    )
    assert _function_uses_name(
        "src/foresight/cli_shared.py",
        "_coerce_model_param_value",
        "_split_csv_items",
    )


def test_cli_shared_source_extracts_arg_value_helpers() -> None:
    source = _read_repo_file("src/foresight/cli_shared.py")

    assert "def _string_arg_value(" in source
    assert "def _list_arg_values(" in source
    assert "def _int_arg_value(" in source
    assert "def _float_arg_value(" in source
    assert "def _bool_arg_value(" in source
    assert "def _output_arg_value(" in source
    assert "def _format_arg_value(" in source
    assert "def _stripped_arg_value(" in source
    assert "def _optional_stripped_arg_value(" in source
    assert "def _parse_cols_arg(" in source
    assert "def _parse_id_cols_arg(" in source
    assert "def _parse_requires_arg(" in source


def test_cli_modules_route_output_and_format_args_through_cli_shared_helpers() -> None:
    cli_source = _read_repo_file("src/foresight/cli.py")
    cli_catalog_source = _read_repo_file("src/foresight/cli_catalog.py")
    cli_data_source = _read_repo_file("src/foresight/cli_data.py")
    cli_leaderboard_source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert "_cli_shared._output_arg_value(" in cli_source
    assert "_cli_shared._format_arg_value(" in cli_source
    assert 'output=str(args.output)' not in cli_source
    assert "output=args.output" not in cli_source
    assert "str(args.format)" not in cli_source

    assert "_cli_shared._output_arg_value(" in cli_catalog_source
    assert "_cli_shared._format_arg_value(" in cli_catalog_source
    assert 'output=str(args.output)' not in cli_catalog_source
    assert "str(args.format)" not in cli_catalog_source

    assert "_cli_shared._output_arg_value(" in cli_data_source
    assert "_cli_shared._format_arg_value(" in cli_data_source
    assert 'str(getattr(args, "output", ""))' not in cli_data_source
    assert "str(args.format)" not in cli_data_source

    assert "_cli_shared._output_arg_value(" in cli_leaderboard_source
    assert "_cli_shared._format_arg_value(" in cli_leaderboard_source
    assert 'output=str(args.output)' not in cli_leaderboard_source
    assert "output=args.output" not in cli_leaderboard_source
    assert "str(args.format)" not in cli_leaderboard_source


def test_cli_modules_route_stripped_string_args_through_cli_shared_helpers() -> None:
    cli_source = _read_repo_file("src/foresight/cli.py")
    cli_catalog_source = _read_repo_file("src/foresight/cli_catalog.py")
    cli_data_source = _read_repo_file("src/foresight/cli_data.py")
    cli_leaderboard_source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert "_cli_shared._stripped_arg_value(" in cli_source
    assert "_cli_shared._optional_stripped_arg_value(" in cli_source
    assert 'str(getattr(args, "future_path", "")).strip()' not in cli_source
    assert 'str(getattr(args, "interval_levels", "")).strip()' not in cli_source
    assert 'str(getattr(args, "save_artifact", "")).strip()' not in cli_source
    assert 'str(getattr(args, "path_prefix", "")).strip()' not in cli_source

    assert "_cli_shared._stripped_arg_value(" in cli_catalog_source
    assert 'str(getattr(args, "stability", "any")).strip()' not in cli_catalog_source
    assert 'str(getattr(args, "columns", "")).strip()' not in cli_catalog_source

    assert "_cli_shared._optional_stripped_arg_value(" in cli_data_source
    assert 'str(getattr(args, "freq", "")).strip()' not in cli_data_source
    assert 'str(getattr(args, "historic_x_missing", "")).strip()' not in cli_data_source
    assert 'str(getattr(args, "future_x_missing", "")).strip()' not in cli_data_source

    assert "_cli_shared._stripped_arg_value(" in cli_leaderboard_source
    assert "_cli_shared._optional_stripped_arg_value(" in cli_leaderboard_source
    assert 'str(getattr(args, "summary_output", "")).strip()' not in cli_leaderboard_source
    assert 'str(getattr(args, "summary_format", "json")).strip()' not in cli_leaderboard_source
    assert 'str(getattr(args, "task_reports_output", "")).strip()' not in cli_leaderboard_source
    assert 'str(getattr(args, "task_reports_format", "json")).strip()' not in cli_leaderboard_source
    assert 'str(getattr(args, "resume", "")).strip()' not in cli_leaderboard_source
    assert 'str(getattr(args, "failures_output", "")).strip()' not in cli_leaderboard_source
    assert 'str(getattr(args, "input", "-")).strip()' not in cli_leaderboard_source
    assert 'str(getattr(args, "input_format", "auto")).strip()' not in cli_leaderboard_source


def test_cli_modules_route_direct_stripped_args_through_cli_shared_helpers() -> None:
    cli_source = _read_repo_file("src/foresight/cli.py")
    cli_catalog_source = _read_repo_file("src/foresight/cli_catalog.py")
    cli_leaderboard_source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert 'str(args.y_col).strip()' not in cli_source
    assert 'str(args.model).strip()' not in cli_source
    assert 'str(args.score_method).strip()' not in cli_source
    assert 'str(args.threshold_method).strip()' not in cli_source
    assert 'str(args.conformal_levels).strip()' not in cli_source

    assert 'str(args.prefix).strip()' not in cli_catalog_source
    assert 'str(args.sort).strip()' not in cli_catalog_source
    assert 'str(args.query).strip()' not in cli_catalog_source
    assert 'str(args.paper_id).strip()' not in cli_catalog_source
    assert 'str(args.role).strip()' not in cli_catalog_source

    assert 'str(args.y_col).strip()' not in cli_leaderboard_source
    assert 'str(args.models).strip()' not in cli_leaderboard_source
    assert 'str(args.data_dir).strip()' not in cli_leaderboard_source


def test_cli_modules_route_column_arg_parsing_through_cli_shared_helpers() -> None:
    cli_source = _read_repo_file("src/foresight/cli.py")
    cli_data_source = _read_repo_file("src/foresight/cli_data.py")

    assert "_cli_shared._parse_id_cols_arg(" in cli_source
    assert 'parse_id_cols(str(args.id_cols))' not in cli_source

    assert "_cli_shared._parse_cols_arg(" in cli_data_source
    assert "_cli_shared._parse_id_cols_arg(" in cli_data_source
    assert 'parse_id_cols(str(args.id_cols))' not in cli_data_source
    assert 'parse_id_cols(str(getattr(args, "id_cols", "")))' not in cli_data_source
    assert 'parse_cols(str(args.x_cols))' not in cli_data_source
    assert 'parse_cols(str(args.historic_x_cols))' not in cli_data_source
    assert 'parse_cols(str(args.future_x_cols))' not in cli_data_source
    assert 'parse_cols(str(getattr(args, "historic_x_cols", "")))' not in cli_data_source
    assert 'parse_cols(str(getattr(args, "future_x_cols", "")))' not in cli_data_source
    assert 'parse_cols(str(getattr(args, "columns", "")))' not in cli_data_source
    assert 'parse_cols(str(getattr(args, "columns", "y")))' not in cli_data_source
    assert 'parse_cols(str(getattr(args, "target_cols", "")))' not in cli_data_source
    assert 'parse_cols(str(getattr(args, "x_cols", "")))' not in cli_data_source


def test_cli_data_routes_default_string_args_through_cli_shared_helpers() -> None:
    cli_data_source = _read_repo_file("src/foresight/cli_data.py")

    assert "_cli_shared._string_arg_value(" in cli_data_source
    assert 'str(getattr(args, "y_missing", "error"))' not in cli_data_source
    assert 'str(getattr(args, "x_missing", "error"))' not in cli_data_source
    assert 'str(getattr(args, "agg", "last"))' not in cli_data_source
    assert 'str(getattr(args, "method", "iqr"))' not in cli_data_source
    assert 'str(getattr(args, "prefix", "cal_"))' not in cli_data_source
    assert 'str(getattr(args, "ds_col", "ds"))' not in cli_data_source
    assert 'str(getattr(args, "input_format", "auto"))' not in cli_data_source
    assert 'str(getattr(args, "lags", "5"))' not in cli_data_source
    assert 'str(getattr(args, "roll_windows", ""))' not in cli_data_source
    assert 'str(getattr(args, "roll_stats", ""))' not in cli_data_source
    assert 'str(getattr(args, "diff_lags", ""))' not in cli_data_source
    assert 'str(getattr(args, "seasonal_lags", ""))' not in cli_data_source
    assert 'str(getattr(args, "seasonal_diff_lags", ""))' not in cli_data_source
    assert 'str(getattr(args, "fourier_periods", ""))' not in cli_data_source
    assert 'str(getattr(args, "fourier_orders", "2"))' not in cli_data_source


def test_cli_data_routes_required_string_args_through_cli_shared_helpers() -> None:
    cli_data_source = _read_repo_file("src/foresight/cli_data.py")

    assert "_cli_shared._string_arg_value(" in cli_data_source
    assert 'str(args.data_dir)' not in cli_data_source
    assert 'str(args.key)' not in cli_data_source
    assert 'str(args.dataset)' not in cli_data_source
    assert 'str(args.path)' not in cli_data_source
    assert 'str(args.time_col)' not in cli_data_source
    assert 'str(args.y_col)' not in cli_data_source


def test_cli_command_handlers_route_required_string_args_through_cli_shared_helpers() -> None:
    path = "src/foresight/cli.py"

    assert _function_uses_attr(path, "_cmd_cv_run", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_cv_csv", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_forecast_csv", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_forecast_artifact", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_artifact_info", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_artifact_validate", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_artifact_diff", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_tuning_run", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_detect_run", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_detect_csv", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_eval_naive_last", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_eval_seasonal_naive", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_eval_run", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_eval_csv", "_cli_shared", "_string_arg_value")


def test_cli_routes_doctor_and_shortcut_string_args_through_cli_shared_helpers() -> None:
    path = "src/foresight/cli.py"

    assert _function_uses_attr(path, "_root_shortcut_handler", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_doctor", "_cli_shared", "_string_arg_value")


def test_cli_leaderboard_routes_string_args_through_cli_shared_helpers() -> None:
    path = "src/foresight/cli_leaderboard.py"
    source = _read_repo_file(path)

    assert _function_uses_attr(path, "_cmd_leaderboard_naive", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_leaderboard_models", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(
        path,
        "_resolve_leaderboard_sweep_model_keys",
        "_cli_shared",
        "_string_arg_value",
    )
    assert _function_uses_attr(
        path,
        "_write_leaderboard_sweep_task_reports",
        "_cli_shared",
        "_string_arg_value",
    )
    assert _function_uses_attr(path, "_cmd_leaderboard_sweep", "_cli_shared", "_string_arg_value")

    assert 'str(args.dataset)' not in source
    assert 'str(args.y_col)' not in source
    assert 'str(args.data_dir)' not in source
    assert 'str(args.datasets)' not in source
    assert 'str(args.backend)' not in source
    assert 'str(args.models)' not in source


def test_cli_leaderboard_routes_default_string_args_through_cli_shared_helpers() -> None:
    path = "src/foresight/cli_leaderboard.py"
    source = _read_repo_file(path)

    assert _function_uses_attr(path, "_cmd_leaderboard_models", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_write_leaderboard_sweep_summary", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_leaderboard_sweep", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_leaderboard_summarize", "_cli_shared", "_string_arg_value")

    assert 'str(getattr(args, "task_group", "all"))' not in source
    assert 'str(getattr(args, "summary_sort", "mae_rank_mean"))' not in source
    assert 'str(getattr(args, "chunk_size", "1"))' not in source
    assert 'str(getattr(args, "sort", "mae_mean"))' not in source


def test_cli_catalog_routes_string_and_requires_args_through_cli_shared_helpers() -> None:
    path = "src/foresight/cli_catalog.py"
    source = _read_repo_file(path)

    assert _function_uses_attr(path, "_cmd_models_list", "_cli_shared", "_parse_requires_arg")
    assert _function_uses_attr(path, "_cmd_models_info", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_cmd_models_search", "_cli_shared", "_parse_requires_arg")
    assert _function_uses_attr(path, "_cmd_docs_rnn", "_cli_shared", "_string_arg_value")

    assert 'str(getattr(args, "requires", ""))' not in source
    assert 'str(getattr(args, "exclude_requires", ""))' not in source
    assert 'str(args.key)' not in source
    assert 'str(args.output_dir)' not in source


def test_cli_catalog_and_leaderboard_route_default_filter_args_through_cli_shared_helpers() -> None:
    cli_catalog_source = _read_repo_file("src/foresight/cli_catalog.py")
    cli_leaderboard_source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert 'getattr(args, "interface", "any")' not in cli_catalog_source
    assert 'getattr(args, "task_group", "all")' not in cli_leaderboard_source


def test_cli_catalog_and_leaderboard_route_csv_splitting_through_cli_shared_helper() -> None:
    cli_catalog_source = _read_repo_file("src/foresight/cli_catalog.py")
    cli_leaderboard_source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert "_cli_shared._split_csv_items(" in cli_catalog_source
    assert "_cli_shared._split_csv_items(" in cli_leaderboard_source

    assert 'columns_raw.split(",")' not in cli_catalog_source
    assert 'raw_models.split(",")' not in cli_leaderboard_source
    assert 'raw.split(",")' not in cli_leaderboard_source
    assert 'datasets_raw.split(",")' not in cli_leaderboard_source


def test_cli_runtime_routes_config_args_through_cli_shared_helpers() -> None:
    path = "src/foresight/cli_runtime.py"

    assert _function_uses_attr(path, "_config_from_args", "_cli_shared", "_string_arg_value")
    assert _function_uses_attr(path, "_config_from_args", "_cli_shared", "_bool_arg_value")


def test_cli_modules_route_list_args_through_cli_shared_helpers() -> None:
    cli_source = _read_repo_file("src/foresight/cli.py")
    cli_catalog_source = _read_repo_file("src/foresight/cli_catalog.py")
    cli_leaderboard_source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert "_cli_shared._list_arg_values(" in cli_source
    assert 'list(getattr(args, "require_extra", []))' not in cli_source
    assert "list(args.model_param)" not in cli_source
    assert "list(args.grid_param)" not in cli_source

    assert "_cli_shared._list_arg_values(" in cli_catalog_source
    assert 'list(getattr(args, "capability", []))' not in cli_catalog_source

    assert "_cli_shared._list_arg_values(" in cli_leaderboard_source
    assert 'list(getattr(args, "model_param", []))' not in cli_leaderboard_source


def test_cli_modules_route_numeric_and_bool_args_through_cli_shared_helpers() -> None:
    cli_source = _read_repo_file("src/foresight/cli.py")
    cli_data_source = _read_repo_file("src/foresight/cli_data.py")
    cli_catalog_source = _read_repo_file("src/foresight/cli_catalog.py")
    cli_leaderboard_source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert "_cli_shared._bool_arg_value(" in cli_source
    assert "_cli_shared._int_arg_value(" in cli_source
    assert 'bool(getattr(args, "strict", False))' not in cli_source
    assert 'int(getattr(args, "interval_samples", 1000))' not in cli_source
    assert 'bool(getattr(args, "parse_dates", False))' not in cli_source

    assert "_cli_shared._bool_arg_value(" in cli_data_source
    assert "_cli_shared._int_arg_value(" in cli_data_source
    assert "_cli_shared._float_arg_value(" in cli_data_source
    assert 'bool(getattr(args, "keepna", False))' not in cli_data_source
    assert 'bool(getattr(args, "prepare", False))' not in cli_data_source
    assert 'bool(getattr(args, "strict_freq", False))' not in cli_data_source
    assert 'float(getattr(args, "iqr_k", 1.5))' not in cli_data_source
    assert 'float(getattr(args, "zmax", 3.0))' not in cli_data_source
    assert 'int(getattr(args, "horizon", 1))' not in cli_data_source
    assert 'bool(getattr(args, "add_time_features", False))' not in cli_data_source
    assert 'bool(getattr(args, "strict", False))' not in cli_data_source

    assert "_cli_shared._bool_arg_value(" in cli_catalog_source
    assert 'bool(getattr(args, "desc", False))' not in cli_catalog_source
    assert 'bool(getattr(args, "header", False))' not in cli_catalog_source

    assert "_cli_shared._bool_arg_value(" in cli_leaderboard_source
    assert "_cli_shared._int_arg_value(" in cli_leaderboard_source
    assert 'int(getattr(args, "summary_limit", 0))' not in cli_leaderboard_source
    assert 'int(getattr(args, "summary_min_datasets", 0))' not in cli_leaderboard_source
    assert 'bool(getattr(args, "strict", False))' not in cli_leaderboard_source
    assert 'int(getattr(args, "limit", 0))' not in cli_leaderboard_source
    assert 'int(getattr(args, "min_datasets", 0))' not in cli_leaderboard_source


def test_batch_execution_source_extracts_timed_task_helper() -> None:
    source = _read_repo_file("src/foresight/batch_execution.py")

    assert "def _run_timed_task(" in source
    assert _function_uses_name(
        "src/foresight/batch_execution.py",
        "run_batch_tasks_sequential",
        "_run_timed_task",
    )


def test_cli_leaderboard_source_extracts_sweep_helpers() -> None:
    source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert "def _leaderboard_sweep_row_key(" in source
    assert "def _resolve_leaderboard_sweep_model_keys(" in source
    assert "def _load_leaderboard_sweep_resume_state(" in source
    assert "def _build_leaderboard_sweep_tasks(" in source
    assert "def _merge_leaderboard_sweep_rows(" in source
    assert "def _write_leaderboard_sweep_summary(" in source
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_merge_leaderboard_sweep_rows",
        "_leaderboard_sweep_row_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_cmd_leaderboard_sweep",
        "_resolve_leaderboard_sweep_model_keys",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_cmd_leaderboard_sweep",
        "_load_leaderboard_sweep_resume_state",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_cmd_leaderboard_sweep",
        "_build_leaderboard_sweep_tasks",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_cmd_leaderboard_sweep",
        "_merge_leaderboard_sweep_rows",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_cmd_leaderboard_sweep",
        "_write_leaderboard_sweep_summary",
    )


def test_cli_leaderboard_source_extracts_summary_helpers() -> None:
    source = _read_repo_file("src/foresight/cli_leaderboard.py")

    assert "def _leaderboard_summary_group_key(" in source
    assert "def _leaderboard_summary_model_key(" in source
    assert "def _leaderboard_summary_group_model_key(" in source
    assert "def _leaderboard_summary_metric_key(" in source
    assert "def _leaderboard_summary_metric_model_key(" in source
    assert "def _clean_leaderboard_summary_rows(" in source
    assert "def _leaderboard_summary_best_by_dataset_metric(" in source
    assert "def _leaderboard_summary_rank_by_dataset_metric_model(" in source
    assert "def _leaderboard_model_summary_row(" in source
    assert "def _sort_leaderboard_summary_rows(" in source
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_leaderboard_summary_best_by_dataset_metric",
        "_leaderboard_summary_group_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_leaderboard_summary_rank_by_dataset_metric_model",
        "_leaderboard_summary_group_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_leaderboard_summary_metric_relative_values",
        "_leaderboard_summary_metric_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_leaderboard_summary_metric_relative_pairs",
        "_leaderboard_summary_metric_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_leaderboard_summary_metric_ranks",
        "_leaderboard_summary_metric_model_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_leaderboard_summary_metric_rank_pairs",
        "_leaderboard_summary_metric_model_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_leaderboard_summary_rank_by_dataset_metric_model",
        "_leaderboard_summary_model_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_build_leaderboard_metric_contexts",
        "_leaderboard_summary_group_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_build_leaderboard_metric_contexts",
        "_leaderboard_summary_metric_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_build_leaderboard_metric_contexts",
        "_leaderboard_summary_metric_model_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_summarize_leaderboard_rows",
        "_leaderboard_summary_group_model_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_leaderboard_model_summary_row",
        "_leaderboard_summary_group_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_summarize_leaderboard_rows",
        "_leaderboard_summary_group_key",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_summarize_leaderboard_rows",
        "_clean_leaderboard_summary_rows",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_summarize_leaderboard_rows",
        "_leaderboard_summary_best_by_dataset_metric",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_summarize_leaderboard_rows",
        "_leaderboard_summary_rank_by_dataset_metric_model",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_summarize_leaderboard_rows",
        "_leaderboard_model_summary_row",
    )
    assert _function_uses_name(
        "src/foresight/cli_leaderboard.py",
        "_summarize_leaderboard_rows",
        "_sort_leaderboard_summary_rows",
    )


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


def test_global_regression_source_extracts_panel_step_lag_helpers() -> None:
    source = _read_repo_file("src/foresight/models/global_regression.py")

    assert "def _panel_step_lag_required_start(" in source
    assert "def _panel_step_lag_extra_parts(" in source
    assert "def _panel_step_lag_feature_matrix(" in source
    assert _function_uses_name(
        "src/foresight/models/global_regression.py",
        "_panel_step_lag_train_xy",
        "_panel_step_lag_required_start",
    )
    assert _function_uses_name(
        "src/foresight/models/global_regression.py",
        "_panel_step_lag_train_xy",
        "_panel_step_lag_extra_parts",
    )
    assert _function_uses_name(
        "src/foresight/models/global_regression.py",
        "_panel_step_lag_train_xy",
        "_panel_step_lag_feature_matrix",
    )
    assert _function_uses_name(
        "src/foresight/models/global_regression.py",
        "_panel_step_lag_predict_x",
        "_panel_step_lag_required_start",
    )
    assert _function_uses_name(
        "src/foresight/models/global_regression.py",
        "_panel_step_lag_predict_x",
        "_panel_step_lag_extra_parts",
    )
    assert _function_uses_name(
        "src/foresight/models/global_regression.py",
        "_panel_step_lag_predict_x",
        "_panel_step_lag_feature_matrix",
    )


def test_global_regression_source_extracts_forecaster_prediction_helpers() -> None:
    path = "src/foresight/models/global_regression.py"
    source = _read_repo_file(path)

    assert "def _normalize_step_lag_quantiles(" in source
    assert "def _step_lag_point_quantile(" in source
    assert "def _step_lag_training_payload(" in source
    assert "def _iter_step_lag_prediction_inputs(" in source
    assert "def _validated_step_lag_prediction(" in source
    assert "def _step_lag_point_rows(" in source
    assert "def _step_lag_quantile_rows(" in source
    assert _function_uses_name(path, "hgb_step_lag_global_forecaster", "_step_lag_training_payload")
    assert _function_uses_name(path, "hgb_step_lag_global_forecaster", "_iter_step_lag_prediction_inputs")
    assert _function_uses_name(path, "hgb_step_lag_global_forecaster", "_step_lag_point_rows")
    assert _function_uses_name(path, "_xgb_step_lag_global_forecaster_impl", "_normalize_step_lag_quantiles")
    assert _function_uses_name(path, "_xgb_step_lag_global_forecaster_impl", "_step_lag_point_quantile")
    assert _function_uses_name(path, "_xgb_step_lag_global_forecaster_impl", "_step_lag_training_payload")
    assert _function_uses_name(path, "_xgb_step_lag_global_forecaster_impl", "_iter_step_lag_prediction_inputs")
    assert _function_uses_name(path, "_xgb_step_lag_global_forecaster_impl", "_step_lag_quantile_rows")
    assert _function_uses_name(path, "_xgb_step_lag_global_forecaster_impl", "_step_lag_point_rows")
    assert _function_uses_name(path, "lgbm_step_lag_global_forecaster", "_normalize_step_lag_quantiles")
    assert _function_uses_name(path, "lgbm_step_lag_global_forecaster", "_step_lag_point_quantile")
    assert _function_uses_name(path, "lgbm_step_lag_global_forecaster", "_step_lag_training_payload")
    assert _function_uses_name(path, "lgbm_step_lag_global_forecaster", "_iter_step_lag_prediction_inputs")
    assert _function_uses_name(path, "lgbm_step_lag_global_forecaster", "_step_lag_quantile_rows")
    assert _function_uses_name(path, "lgbm_step_lag_global_forecaster", "_step_lag_point_rows")
    assert _function_uses_name(path, "catboost_step_lag_global_forecaster", "_normalize_step_lag_quantiles")
    assert _function_uses_name(path, "catboost_step_lag_global_forecaster", "_step_lag_point_quantile")
    assert _function_uses_name(path, "catboost_step_lag_global_forecaster", "_step_lag_training_payload")
    assert _function_uses_name(path, "catboost_step_lag_global_forecaster", "_iter_step_lag_prediction_inputs")
    assert _function_uses_name(path, "catboost_step_lag_global_forecaster", "_step_lag_quantile_rows")
    assert _function_uses_name(path, "catboost_step_lag_global_forecaster", "_step_lag_point_rows")


def test_regression_source_extracts_lag_matrix_feature_helpers() -> None:
    source = _read_repo_file("src/foresight/models/regression.py")

    assert "def _validated_lag_matrix_time_index(" in source
    assert "def _append_matrix_seasonal_features(" in source
    assert "def _append_matrix_fourier_features(" in source
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_augment_lag_matrix",
        "_validated_lag_matrix_time_index",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_augment_lag_matrix",
        "_append_matrix_seasonal_features",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_augment_lag_matrix",
        "_append_matrix_fourier_features",
    )


def test_regression_source_extracts_shared_regressor_validator_helpers() -> None:
    source = _read_repo_file("src/foresight/models/regression.py")

    assert "def _require_optional_int_min_param(" in source
    assert "def _require_optional_positive_float_param(" in source
    assert "def _require_optional_fraction_param(" in source
    assert "def _require_optional_non_negative_params(" in source
    assert "def _require_optional_non_zero_int_param(" in source
    assert "def _validate_lgbm_max_depth_param(" in source
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_xgb_validate_common_regressor_params",
        "_require_optional_int_min_param",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_xgb_validate_common_regressor_params",
        "_require_optional_positive_float_param",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_xgb_validate_common_regressor_params",
        "_require_optional_fraction_param",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_xgb_validate_common_regressor_params",
        "_require_optional_non_negative_params",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_xgb_validate_common_regressor_params",
        "_require_optional_non_zero_int_param",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_lgbm_validate_common_regressor_params",
        "_validate_lgbm_max_depth_param",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_lgbm_validate_common_regressor_params",
        "_require_optional_fraction_param",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_lgbm_validate_common_regressor_params",
        "_require_optional_non_negative_params",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_catboost_validate_common_regressor_params",
        "_require_optional_int_min_param",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_catboost_validate_common_regressor_params",
        "_require_optional_positive_float_param",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_catboost_validate_common_regressor_params",
        "_require_optional_non_negative_params",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_catboost_validate_common_regressor_params",
        "_require_optional_non_zero_int_param",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_xgb_lag_direct_forecast",
        "_xgb_validate_common_regressor_params",
    )
    assert _function_uses_name(
        "src/foresight/models/regression.py",
        "_xgb_lag_recursive_forecast",
        "_xgb_validate_common_regressor_params",
    )


def test_regression_source_extracts_dirrec_feature_helpers() -> None:
    source = _read_repo_file("src/foresight/models/regression.py")

    assert "def _dirrec_step_training_features(" in source
    assert "def _dirrec_step_prediction_features(" in source
    for func_name in (
        "_xgb_lag_dirrec_forecast_kwargs",
        "_lgbm_lag_dirrec_forecast_kwargs",
        "_catboost_lag_dirrec_forecast_kwargs",
    ):
        assert _function_uses_name(
            "src/foresight/models/regression.py",
            func_name,
            "_dirrec_step_training_features",
        )
        assert _function_uses_name(
            "src/foresight/models/regression.py",
            func_name,
            "_dirrec_step_prediction_features",
        )


def test_regression_source_extracts_xgbrf_direct_helpers() -> None:
    path = "src/foresight/models/regression.py"
    source = _read_repo_file(path)

    assert "def _direct_forecast_step_features(" in source
    assert "def _build_xgbrf_regressor(" in source
    assert _function_uses_name(path, "xgbrf_lag_direct_forecast", "_xgb_validate_common_regressor_params")
    assert _function_uses_name(path, "xgbrf_lag_direct_forecast", "_direct_forecast_step_features")
    assert _function_uses_name(path, "xgbrf_lag_direct_forecast", "_build_xgbrf_regressor")
    assert _function_uses_name(path, "xgbrf_lag_recursive_forecast", "_xgb_validate_common_regressor_params")
    assert _function_uses_name(path, "xgbrf_lag_recursive_forecast", "_build_xgbrf_regressor")


def test_torch_probabilistic_source_extracts_validation_and_loss_helpers() -> None:
    path = "src/foresight/models/torch_probabilistic.py"
    source = _read_repo_file(path)

    assert "def _validate_probabilistic_direct_config(" in source
    assert "def _make_probabilistic_loss(" in source
    assert _function_uses_name(
        path,
        "torch_probabilistic_direct_forecast",
        "_validate_probabilistic_direct_config",
    )
    assert _function_uses_name(path, "torch_probabilistic_direct_forecast", "_make_probabilistic_loss")


def test_torch_seq2seq_source_extracts_training_and_forecast_helpers() -> None:
    path = "src/foresight/models/torch_seq2seq.py"
    source = _read_repo_file(path)

    assert "def _validate_seq2seq_training_config(" in source
    assert "def _validate_seq2seq_teacher_forcing(" in source
    assert "def _split_seq2seq_train_validation(" in source
    assert "def _make_seq2seq_optimizer(" in source
    assert "def _make_seq2seq_scheduler(" in source
    assert "def _seq2seq_teacher_forcing_ratio(" in source
    assert "def _validate_seq2seq_direct_config(" in source
    assert "def _build_torch_train_config(" in source
    assert "def _prepare_univariate_direct_payload(" in source
    assert "def _predict_direct_torch_model(" in source
    assert "def _maybe_denormalize_forecast(" in source
    assert _function_uses_name(path, "_train_seq2seq", "_validate_seq2seq_training_config")
    assert _function_uses_name(path, "_train_seq2seq", "_validate_seq2seq_teacher_forcing")
    assert _function_uses_name(path, "_train_seq2seq", "_split_seq2seq_train_validation")
    assert _function_uses_name(path, "_train_seq2seq", "_make_seq2seq_optimizer")
    assert _function_uses_name(path, "_train_seq2seq", "_make_seq2seq_scheduler")
    assert _function_uses_name(path, "_train_seq2seq", "_seq2seq_teacher_forcing_ratio")
    assert _function_uses_name(path, "torch_seq2seq_direct_forecast", "_validate_seq2seq_direct_config")
    assert _function_uses_name(path, "torch_seq2seq_direct_forecast", "_build_torch_train_config")
    assert _function_uses_name(path, "torch_seq2seq_direct_forecast", "_prepare_univariate_direct_payload")
    assert _function_uses_name(path, "torch_seq2seq_direct_forecast", "_predict_direct_torch_model")
    assert _function_uses_name(path, "torch_seq2seq_direct_forecast", "_maybe_denormalize_forecast")
    assert _function_uses_name(path, "torch_lstnet_direct_forecast", "_build_torch_train_config")
    assert _function_uses_name(path, "torch_lstnet_direct_forecast", "_prepare_univariate_direct_payload")
    assert _function_uses_name(path, "torch_lstnet_direct_forecast", "_predict_direct_torch_model")
    assert _function_uses_name(path, "torch_lstnet_direct_forecast", "_maybe_denormalize_forecast")


def test_torch_rnnzoo_source_extracts_validation_and_dispatch_helpers() -> None:
    path = "src/foresight/models/torch_rnn_zoo.py"
    source = _read_repo_file(path)

    assert "def _validate_rnnzoo_direct_config(" in source
    assert "def _normalize_clock_periods(" in source
    assert "def _build_rnnzoo_train_config(" in source
    assert "def _prepare_rnnzoo_payload(" in source
    assert "def _predict_rnnzoo_direct_model(" in source
    assert "def _maybe_denormalize_rnnzoo_forecast(" in source
    assert _function_uses_name(path, "torch_rnnzoo_direct_forecast", "_validate_rnnzoo_direct_config")
    assert _function_uses_name(path, "torch_rnnzoo_direct_forecast", "_build_rnnzoo_train_config")
    assert _function_uses_name(path, "torch_rnnzoo_direct_forecast", "_prepare_rnnzoo_payload")
    assert _function_uses_name(path, "torch_rnnzoo_direct_forecast", "_predict_rnnzoo_direct_model")
    assert _function_uses_name(path, "torch_rnnzoo_direct_forecast", "_maybe_denormalize_rnnzoo_forecast")
    assert _function_uses_name(path, "_make_base_encoder", "_normalize_clock_periods")


def test_torch_files_do_not_leave_unused_shape_dimensions() -> None:
    assert _unused_shape_dims("src/foresight/models/torch_nn.py") == []
    assert _unused_shape_dims("src/foresight/models/torch_rnn_paper_zoo.py") == []


def test_torch_rnn_paper_zoo_avoids_sonar_flagged_source_patterns() -> None:
    source = _read_repo_file("src/foresight/models/torch_rnn_paper_zoo.py")

    assert "_PAPER_DESC = {paper_id: desc for paper_id, desc in _PAPER_DEFS}" not in source
    assert "w = torch.zeros((int(B), self.M), device=xb.device, dtype=xb.dtype)" not in source
    assert "return self.head(last)  # (B,2) = (mu, raw_sigma)" not in source


def test_smells_test_source_extracts_unused_shape_dim_helpers() -> None:
    source = _read_repo_file("tests/test_no_sonar_low_hanging_smells.py")

    assert "def _shape_dim_load_names(" in source
    assert "def _shape_dim_assignment_targets(" in source
    assert "def _collect_unused_shape_dims(" in source
    assert _function_uses_name(
        "tests/test_no_sonar_low_hanging_smells.py",
        "_unused_shape_dims",
        "_collect_unused_shape_dims",
    )


def test_s117_targeted_names_test_source_extracts_assignment_helpers() -> None:
    source = _read_repo_file("tests/test_no_sonar_s117_targeted_names.py")

    assert "def _function_arg_names(" in source
    assert "def _assignment_target_nodes(" in source
    assert "def _leaf_assigned_names(" in source
    assert "def _assigned_names_from_function_node(" in source
    assert _function_uses_name(
        "tests/test_no_sonar_s117_targeted_names.py",
        "_assigned_names_in_functions",
        "_assigned_names_from_function_node",
    )
    assert _function_uses_name(
        "tests/test_no_sonar_s117_targeted_names.py",
        "_assigned_names_in_function",
        "_assigned_names_from_function_node",
    )


def test_zero_return_test_source_extracts_counting_helpers() -> None:
    source = _read_repo_file("tests/test_no_duplicate_zero_returns.py")

    assert "def _find_named_function(" in source
    assert "def _count_literal_zero_returns(" in source
    assert _function_uses_name(
        "tests/test_no_duplicate_zero_returns.py",
        "_literal_zero_return_count",
        "_find_named_function",
    )
    assert _function_uses_name(
        "tests/test_no_duplicate_zero_returns.py",
        "_literal_zero_return_count",
        "_count_literal_zero_returns",
    )


def test_cli_test_source_extracts_bound_name_helpers() -> None:
    source = _read_repo_file("tests/test_cli.py")

    assert "def _import_bound_names(" in source
    assert "def _assignment_bound_name(" in source
    assert "def _definition_bound_name(" in source
    assert "def _bound_names_for_node(" in source
    assert _function_uses_name(
        "tests/test_cli.py",
        "_top_level_bound_names",
        "_bound_names_for_node",
    )


def test_data_prep_source_extracts_regularization_helpers() -> None:
    source = _read_repo_file("src/foresight/data/prep.py")

    assert "def _resolve_long_covariate_columns(" in source
    assert "def _resolve_long_missing_policies(" in source
    assert "def _frequency_by_unique_id(" in source
    assert "def _prepare_long_group_frame(" in source
    assert "def _coerce_wide_frame_ds_column(" in source
    assert "def _resolve_wide_target_columns(" in source
    assert "def _regularize_wide_frame(" in source
    assert _function_uses_name(
        "src/foresight/data/prep.py",
        "prepare_long_df",
        "_resolve_long_covariate_columns",
    )
    assert _function_uses_name(
        "src/foresight/data/prep.py",
        "prepare_long_df",
        "_resolve_long_missing_policies",
    )
    assert _function_uses_name(
        "src/foresight/data/prep.py",
        "prepare_long_df",
        "_frequency_by_unique_id",
    )
    assert _function_uses_name(
        "src/foresight/data/prep.py",
        "prepare_long_df",
        "_prepare_long_group_frame",
    )
    assert _function_uses_name(
        "src/foresight/data/prep.py",
        "prepare_wide_df",
        "_coerce_wide_frame_ds_column",
    )
    assert _function_uses_name(
        "src/foresight/data/prep.py",
        "prepare_wide_df",
        "_resolve_wide_target_columns",
    )
    assert _function_uses_name(
        "src/foresight/data/prep.py",
        "prepare_wide_df",
        "_regularize_wide_frame",
    )


def test_features_lag_source_extracts_seasonal_feature_helpers() -> None:
    source = _read_repo_file("src/foresight/features/lag.py")

    assert "def _lagged_feature_matrix(" in source
    assert "def _validated_lag_steps_and_start_index(" in source
    assert "def _validated_target_indices(" in source
    assert "def _seasonal_lag_feature_columns(" in source
    assert "def _seasonal_diff_feature_columns(" in source
    assert _function_uses_name(
        "src/foresight/features/lag.py",
        "make_lagged_xy",
        "_validated_lag_steps_and_start_index",
    )
    assert _function_uses_name(
        "src/foresight/features/lag.py",
        "make_lagged_xy_multi",
        "_validated_lag_steps_and_start_index",
    )
    assert _function_uses_name(
        "src/foresight/features/lag.py",
        "make_lagged_xy",
        "_lagged_feature_matrix",
    )
    assert _function_uses_name(
        "src/foresight/features/lag.py",
        "make_lagged_xy_multi",
        "_lagged_feature_matrix",
    )
    assert _function_uses_name(
        "src/foresight/features/lag.py",
        "build_seasonal_lag_features",
        "_validated_target_indices",
    )
    assert _function_uses_name(
        "src/foresight/features/lag.py",
        "build_seasonal_lag_features",
        "_seasonal_lag_feature_columns",
    )
    assert _function_uses_name(
        "src/foresight/features/lag.py",
        "build_seasonal_lag_features",
        "_seasonal_diff_feature_columns",
    )


def test_cv_source_extracts_prediction_helpers() -> None:
    source = _read_repo_file("src/foresight/cv.py")

    assert "def _normalize_cv_x_cols(" in source
    assert "def _trim_cv_splits(" in source
    assert "def _local_cv_prediction_rows(" in source
    assert "def _global_cv_prediction_table(" in source
    assert _function_uses_name(
        "src/foresight/cv.py",
        "cross_validation_predictions",
        "_normalize_cv_x_cols",
    )
    assert _function_uses_name(
        "src/foresight/cv.py",
        "cross_validation_predictions_long_df",
        "_local_cv_prediction_rows",
    )
    assert _function_uses_name(
        "src/foresight/cv.py",
        "cross_validation_predictions_long_df",
        "_global_cv_prediction_table",
    )
    assert _function_uses_name(
        "src/foresight/cv.py",
        "_local_cv_prediction_rows",
        "_trim_cv_splits",
    )
    assert _function_uses_name(
        "src/foresight/cv.py",
        "_global_cv_cutoffs",
        "_trim_cv_splits",
    )


def test_hierarchical_source_extracts_reconciliation_helpers() -> None:
    source = _read_repo_file("src/foresight/hierarchical.py")

    assert "def _missing_leaf_nodes_from_pivot(" in source
    assert "def _extra_node_series_from_pivot(" in source
    assert "def _frame_from_series_map(" in source
    assert "def _bottom_up_series_map(" in source
    assert "def _top_down_allocated_series(" in source
    assert _function_uses_name(
        "src/foresight/hierarchical.py",
        "_reconcile_exog_bottom_up",
        "_bottom_up_reconciled_frame",
    )
    assert _function_uses_name(
        "src/foresight/hierarchical.py",
        "_bottom_up_reconciled_frame",
        "_bottom_up_series_map",
    )
    assert _function_uses_name(
        "src/foresight/hierarchical.py",
        "_bottom_up_reconciled_frame",
        "_frame_from_series_map",
    )
    assert _function_uses_name(
        "src/foresight/hierarchical.py",
        "reconcile_hierarchical_forecasts",
        "_missing_leaf_nodes_from_pivot",
    )
    assert _function_uses_name(
        "src/foresight/hierarchical.py",
        "reconcile_hierarchical_forecasts",
        "_top_down_reconciled_frame",
    )
    assert _function_uses_name(
        "src/foresight/hierarchical.py",
        "_top_down_reconciled_frame",
        "_top_down_allocated_series",
    )


def test_analog_source_extracts_neighbor_selection_helpers() -> None:
    source = _read_repo_file("src/foresight/models/analog.py")

    assert "def _validate_analog_forecast_inputs(" in source
    assert "def _maybe_normalize_window_bank(" in source
    assert "def _analog_neighbor_prediction(" in source
    assert _function_uses_name(
        "src/foresight/models/analog.py",
        "analog_knn_forecast",
        "_validate_analog_forecast_inputs",
    )
    assert _function_uses_name(
        "src/foresight/models/analog.py",
        "analog_knn_forecast",
        "_maybe_normalize_window_bank",
    )
    assert _function_uses_name(
        "src/foresight/models/analog.py",
        "analog_knn_forecast",
        "_analog_neighbor_prediction",
    )


def test_fourier_source_extracts_order_normalization_helpers() -> None:
    source = _read_repo_file("src/foresight/models/fourier.py")

    assert "def _fourier_design_matrix(" in source
    assert "def _linear_design_forecast(" in source
    assert "def _default_order_tuple(" in source
    assert "def _coerce_order_parts(" in source
    assert "def _normalize_order_parts(" in source
    assert _function_uses_name(
        "src/foresight/models/fourier.py",
        "fourier_regression_forecast",
        "_fourier_design_matrix",
    )
    assert _function_uses_name(
        "src/foresight/models/fourier.py",
        "fourier_regression_forecast",
        "_linear_design_forecast",
    )
    assert _function_uses_name(
        "src/foresight/models/fourier.py",
        "fourier_multi_regression_forecast",
        "_fourier_design_matrix",
    )
    assert _function_uses_name(
        "src/foresight/models/fourier.py",
        "fourier_multi_regression_forecast",
        "_linear_design_forecast",
    )
    assert _function_uses_name(
        "src/foresight/models/fourier.py",
        "_normalize_orders",
        "_default_order_tuple",
    )
    assert _function_uses_name(
        "src/foresight/models/fourier.py",
        "_normalize_orders",
        "_coerce_order_parts",
    )
    assert _function_uses_name(
        "src/foresight/models/fourier.py",
        "_normalize_orders",
        "_normalize_order_parts",
    )


def test_eval_predictions_source_extracts_quantile_interval_helpers() -> None:
    source = _read_repo_file("src/foresight/eval_predictions.py")

    assert "def _step_group_inverse_counts(" in source
    assert "def _validated_interval_arrays(" in source
    assert "def _mean_by_step_from_inverse(" in source
    assert "def _interval_score_vector(" in source
    assert "def _quantile_column_map(" in source
    assert "def _symmetric_interval_levels(" in source
    assert "def _per_step_interval_metrics(" in source
    assert "def _weighted_interval_score_by_step(" in source
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "_vectorized_interval_metrics",
        "_mean_by_step_from_inverse",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "_weighted_interval_score_by_step",
        "_mean_by_step_from_inverse",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "_vectorized_point_metrics",
        "_mean_by_step_from_inverse",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "_vectorized_interval_metrics",
        "_validated_interval_arrays",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "_weighted_interval_score_by_step",
        "_validated_interval_arrays",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "_vectorized_interval_metrics",
        "_interval_score_vector",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "_weighted_interval_score_by_step",
        "_interval_score_vector",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "_vectorized_interval_metrics",
        "_step_group_inverse_counts",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "_weighted_interval_score_by_step",
        "_step_group_inverse_counts",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "_vectorized_point_metrics",
        "_step_group_inverse_counts",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "evaluate_quantile_predictions",
        "_quantile_column_map",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "evaluate_quantile_predictions",
        "_symmetric_interval_levels",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "evaluate_quantile_predictions",
        "_per_step_interval_metrics",
    )
    assert _function_uses_name(
        "src/foresight/eval_predictions.py",
        "evaluate_quantile_predictions",
        "_weighted_interval_score_by_step",
    )


def test_intermittent_source_extracts_les_helpers() -> None:
    path = "src/foresight/models/intermittent.py"
    source = _read_repo_file(path)

    assert "def _zero_forecast(" in source
    assert "def _validated_intermittent_input(" in source
    assert "def _validated_alpha_beta(" in source
    assert "def _first_nonzero_index(" in source
    assert "def _croston_initial_state(" in source
    assert "def _les_decay_value(" in source
    assert "def _les_update_state(" in source
    assert _function_uses_name(path, "croston_classic_forecast", "_zero_forecast")
    assert _function_uses_name(path, "croston_optimized_forecast", "_zero_forecast")
    assert _function_uses_name(path, "les_forecast", "_zero_forecast")
    assert _function_uses_name(path, "tsb_forecast", "_zero_forecast")
    assert _function_uses_name(path, "adida_forecast", "_zero_forecast")
    assert _function_uses_name(path, "croston_classic_forecast", "_validated_intermittent_input")
    assert _function_uses_name(path, "croston_optimized_forecast", "_validated_intermittent_input")
    assert _function_uses_name(path, "les_forecast", "_validated_intermittent_input")
    assert _function_uses_name(path, "tsb_forecast", "_validated_intermittent_input")
    assert _function_uses_name(path, "adida_forecast", "_validated_intermittent_input")
    assert _function_uses_name(path, "les_forecast", "_validated_alpha_beta")
    assert _function_uses_name(path, "tsb_forecast", "_validated_alpha_beta")
    assert _function_uses_name(path, "_croston_initial_state", "_first_nonzero_index")
    assert _function_uses_name(path, "croston_classic_forecast", "_croston_initial_state")
    assert _function_uses_name(path, "_croston_sse", "_croston_initial_state")
    assert _function_uses_name(path, "les_forecast", "_first_nonzero_index")
    assert _function_uses_name(path, "les_forecast", "_les_update_state")
    assert _function_uses_name(path, "_les_update_state", "_les_decay_value")


def test_multivariate_source_extracts_adj_resolution_helpers() -> None:
    path = "src/foresight/models/multivariate.py"
    source = _read_repo_file(path)

    assert "def _validated_torch_multivariate_model_dims(" in source
    assert "def _maybe_normalize_multivariate_matrix(" in source
    assert "def _maybe_denormalize_multivariate_forecast(" in source
    assert "def _latest_multivariate_window(" in source
    assert "def _prepare_torch_multivariate_training_data(" in source
    assert "def _load_adj_matrix_from_path(" in source
    assert "def _resolve_builtin_adj_matrix(" in source
    assert "def _corr_topk_adj_matrix(" in source
    assert _function_uses_name(
        path,
        "torch_stid_forecast",
        "_validated_torch_multivariate_model_dims",
    )
    assert _function_uses_name(
        path,
        "torch_stid_forecast",
        "_latest_multivariate_window",
    )
    assert _function_uses_name(
        path,
        "torch_stid_forecast",
        "_prepare_torch_multivariate_training_data",
    )
    assert _function_uses_name(
        path,
        "torch_stgcn_forecast",
        "_validated_torch_multivariate_model_dims",
    )
    assert _function_uses_name(
        path,
        "torch_stgcn_forecast",
        "_latest_multivariate_window",
    )
    assert _function_uses_name(
        path,
        "torch_stgcn_forecast",
        "_prepare_torch_multivariate_training_data",
    )
    assert _function_uses_name(
        path,
        "torch_graphwavenet_forecast",
        "_validated_torch_multivariate_model_dims",
    )
    assert _function_uses_name(
        path,
        "torch_graphwavenet_forecast",
        "_latest_multivariate_window",
    )
    assert _function_uses_name(
        path,
        "torch_graphwavenet_forecast",
        "_prepare_torch_multivariate_training_data",
    )
    assert _function_uses_name(path, "_resolve_adj_matrix", "_load_adj_matrix_from_path")
    assert _function_uses_name(path, "_resolve_adj_matrix", "_resolve_builtin_adj_matrix")
    assert _function_uses_name(path, "_resolve_builtin_adj_matrix", "_corr_topk_adj_matrix")


def test_statsmodels_wrap_source_extracts_auto_arima_search_helpers() -> None:
    path = "src/foresight/models/statsmodels_wrap.py"
    source = _read_repo_file(path)

    assert "def _validate_auto_arima_grid_bounds(" in source
    assert "def _normalize_auto_arima_search_config(" in source
    assert "def _iter_auto_arima_candidate_orders(" in source
    assert "def _fit_auto_arima_candidate_result(" in source
    assert _function_uses_name(path, "_fit_auto_arima_best_result", "_validate_auto_arima_grid_bounds")
    assert _function_uses_name(path, "_fit_auto_arima_best_result", "_normalize_auto_arima_search_config")
    assert _function_uses_name(path, "_fit_auto_arima_best_result", "_iter_auto_arima_candidate_orders")
    assert _function_uses_name(path, "_fit_auto_arima_best_result", "_fit_auto_arima_candidate_result")


def test_statsmodels_wrap_source_extracts_fourier_order_normalization_helpers() -> None:
    path = "src/foresight/models/statsmodels_wrap.py"
    source = _read_repo_file(path)

    assert "def _repeat_fourier_order(" in source
    assert "def _normalize_sequence_fourier_orders(" in source
    assert _function_uses_name(path, "_normalize_fourier_orders", "_repeat_fourier_order")
    assert _function_uses_name(path, "_normalize_fourier_orders", "_normalize_sequence_fourier_orders")
