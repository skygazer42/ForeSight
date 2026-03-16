#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

GROUP_CORE_CLASSES = "Core classes"
GROUP_DATA_PREPARATION = "Data preparation"
GROUP_HIERARCHICAL_FORECASTING = "Hierarchical forecasting"
GROUP_INTERVALS_AND_TUNING = "Intervals and tuning"
SOURCE_FORESIGHT_DATA = "foresight.data"
SOURCE_FORESIGHT_EVAL_FORECAST = "foresight.eval_forecast"
SOURCE_FORESIGHT_SERIALIZATION = "foresight.serialization"
SOURCE_FORESIGHT_MODELS_REGISTRY = "foresight.models.registry"

_CAPABILITY_DESCRIPTIONS: dict[str, str] = {
    "requires_future_covariates": "Model expects known future covariates rather than treating them as optional.",
    "supports_artifact_save": "Model can be persisted and restored through the artifact save/load workflow.",
    "supports_interval_forecast": "Model supports forecast intervals in at least one supported execution path.",
    "supports_interval_forecast_with_x_cols": "Model supports forecast intervals when `x_cols` exogenous features are supplied.",
    "supports_quantiles": "Model can emit quantile forecast columns directly.",
    "supports_x_cols": "Model accepts future covariates / exogenous regressors through `x_cols`.",
}

_API_GROUP_ORDER = [
    GROUP_CORE_CLASSES,
    "Forecasting",
    "Evaluation",
    "Artifacts",
    GROUP_DATA_PREPARATION,
    GROUP_HIERARCHICAL_FORECASTING,
    GROUP_INTERVALS_AND_TUNING,
    "Package metadata",
]

_API_METADATA: dict[str, dict[str, str]] = {
    "__version__": {
        "group": "Package metadata",
        "source": "foresight.__init__",
        "purpose": "Installed ForeSight package version.",
    },
    "BaseForecaster": {
        "group": GROUP_CORE_CLASSES,
        "source": "foresight.base",
        "purpose": "Stateful local forecaster base class with fit/predict helpers.",
    },
    "BaseGlobalForecaster": {
        "group": GROUP_CORE_CLASSES,
        "source": "foresight.base",
        "purpose": "Stateful panel/global forecaster base class for long-format data.",
    },
    "align_long_df": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Regularize per-series timestamps to a target frequency, with optional resampling aggregation.",
    },
    "bootstrap_intervals": {
        "group": GROUP_INTERVALS_AND_TUNING,
        "source": "foresight.intervals",
        "purpose": "Construct bootstrap forecast intervals from historical residual behavior.",
    },
    "build_hierarchy_spec": {
        "group": GROUP_HIERARCHICAL_FORECASTING,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Build a hierarchy specification from raw identifier columns.",
    },
    "check_hierarchical_consistency": {
        "group": GROUP_HIERARCHICAL_FORECASTING,
        "source": "foresight.hierarchical",
        "purpose": "Validate whether hierarchical forecasts reconcile cleanly.",
    },
    "clip_long_df_outliers": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Clip per-series numeric outliers in long-format data without dropping rows.",
    },
    "eval_hierarchical_forecast_df": {
        "group": GROUP_HIERARCHICAL_FORECASTING,
        "source": SOURCE_FORESIGHT_EVAL_FORECAST,
        "purpose": "Score reconciled hierarchical forecasts against held-out history, including bottom-up exogenous aggregation when requested.",
    },
    "eval_model": {
        "group": "Evaluation",
        "source": SOURCE_FORESIGHT_EVAL_FORECAST,
        "purpose": "Walk-forward evaluation for a single univariate series or packaged dataset.",
    },
    "eval_model_long_df": {
        "group": "Evaluation",
        "source": SOURCE_FORESIGHT_EVAL_FORECAST,
        "purpose": "Walk-forward evaluation for long-format panel/global forecasting data.",
    },
    "eval_multivariate_model_df": {
        "group": "Evaluation",
        "source": SOURCE_FORESIGHT_EVAL_FORECAST,
        "purpose": "Evaluate multivariate forecasters on wide data frames.",
    },
    "enrich_long_df_calendar": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Append deterministic calendar and cyclical time features onto long-format panel data.",
    },
    "fit_long_df_scaler": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Fit reversible per-series or global scaling statistics for long-format numeric columns.",
    },
    "forecast_model": {
        "group": "Forecasting",
        "source": "foresight.forecast",
        "purpose": "Run a one-off forecast for a single series and return a forecast dataframe.",
    },
    "forecast_model_long_df": {
        "group": "Forecasting",
        "source": "foresight.forecast",
        "purpose": "Run a one-off forecast for long-format panel/global inputs, optionally with a separate future_df for known future covariates.",
    },
    "infer_series_frequency": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Infer a sensible pandas-compatible series frequency from timestamps.",
    },
    "inverse_transform_long_df_with_scaler": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Reverse fitted long-format scaling statistics to restore original numeric units.",
    },
    "load_forecaster": {
        "group": "Artifacts",
        "source": SOURCE_FORESIGHT_SERIALIZATION,
        "purpose": "Load a persisted forecaster object from disk.",
    },
    "load_forecaster_artifact": {
        "group": "Artifacts",
        "source": SOURCE_FORESIGHT_SERIALIZATION,
        "purpose": "Inspect the structured artifact payload before reconstructing an object.",
    },
    "make_panel_sequence_blocks": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Expose packed panel sequence tensors as explicit past/future target, covariate, and time blocks for encoder-decoder style models.",
    },
    "make_panel_sequence_tensors": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Build packed sequence-model training and prediction bundles from long-format panel data for global neural workflows.",
    },
    "make_panel_window_arrays": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Convert long-format panel series into dense training arrays plus window metadata for sklearn-style estimators.",
    },
    "make_panel_window_frame": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Build step-wise panel training windows from long-format data with target, seasonal, and exogenous lag features.",
    },
    "make_supervised_frame": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Build sklearn-style supervised training tables from long or wide time-series inputs.",
    },
    "make_forecaster": {
        "group": "Forecasting",
        "source": SOURCE_FORESIGHT_MODELS_REGISTRY,
        "purpose": "Create a stateless local forecasting callable from the registry.",
    },
    "make_forecaster_object": {
        "group": "Forecasting",
        "source": SOURCE_FORESIGHT_MODELS_REGISTRY,
        "purpose": "Create a stateful local forecaster object with fit/predict/save support.",
    },
    "make_global_forecaster": {
        "group": "Forecasting",
        "source": SOURCE_FORESIGHT_MODELS_REGISTRY,
        "purpose": "Create a stateless global/panel forecasting callable from the registry.",
    },
    "make_global_forecaster_object": {
        "group": "Forecasting",
        "source": SOURCE_FORESIGHT_MODELS_REGISTRY,
        "purpose": "Create a stateful global forecaster object for panel workflows.",
    },
    "make_multivariate_forecaster": {
        "group": "Forecasting",
        "source": SOURCE_FORESIGHT_MODELS_REGISTRY,
        "purpose": "Create a multivariate forecaster callable for wide matrix forecasting.",
    },
    "prepare_long_df": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Normalize and validate long-format panel data before forecasting/evaluation, with separate missing-value policies for target, historic covariates, and future covariates.",
    },
    "reconcile_hierarchical_forecasts": {
        "group": GROUP_HIERARCHICAL_FORECASTING,
        "source": "foresight.hierarchical",
        "purpose": "Reconcile hierarchical forecasts with top-down or bottom-up methods, with optional bottom-up exogenous aggregation via exog_agg.",
    },
    "save_forecaster": {
        "group": "Artifacts",
        "source": SOURCE_FORESIGHT_SERIALIZATION,
        "purpose": "Persist a fitted forecaster and its schema/version metadata to disk.",
    },
    "split_panel_window_arrays": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Chronologically split panel-window training arrays into train, validation, and test partitions by window origin.",
    },
    "split_panel_window_frame": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Chronologically split panel-window training rows into train, validation, and test partitions by window origin.",
    },
    "split_panel_sequence_blocks": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Chronologically split structured panel sequence blocks into train, validation, and test partitions.",
    },
    "split_panel_sequence_tensors": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Chronologically split packed panel sequence windows into train, validation, and test tensor partitions.",
    },
    "split_long_df": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Chronologically split each long-format series into train, validation, and test partitions.",
    },
    "to_long": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Convert wide or column-mapped inputs into ForeSight long format with role-aware historic_x_cols / future_x_cols support.",
    },
    "transform_long_df_with_scaler": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Apply fitted scaling statistics to long-format numeric columns for training or evaluation workflows.",
    },
    "tune_model": {
        "group": GROUP_INTERVALS_AND_TUNING,
        "source": "foresight.tuning",
        "purpose": "Grid-search a local forecasting model against backtest metrics.",
    },
    "tune_model_long_df": {
        "group": GROUP_INTERVALS_AND_TUNING,
        "source": "foresight.tuning",
        "purpose": "Grid-search a panel/global model on long-format data.",
    },
    "validate_long_df": {
        "group": GROUP_DATA_PREPARATION,
        "source": SOURCE_FORESIGHT_DATA,
        "purpose": "Check that long-format inputs satisfy required schema and null rules.",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path(root: Path) -> None:
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _normalized_text(text: str) -> str:
    return text.rstrip() + "\n"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_normalized_text(text), encoding="utf-8")


def _bool_cell(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _render_models_doc() -> str:
    root = _repo_root()
    _ensure_src_on_path(root)

    from foresight.models import registry as registry_mod

    keys = sorted(registry_mod.list_models())
    specs = [registry_mod.get_model_spec(key) for key in keys]
    capability_keys = sorted({name for spec in specs for name in spec.capabilities})

    missing_capability_docs = [
        name for name in capability_keys if name not in _CAPABILITY_DESCRIPTIONS
    ]
    if missing_capability_docs:
        joined = ", ".join(missing_capability_docs)
        raise RuntimeError(f"Missing capability descriptions for: {joined}")

    interface_counts = Counter(str(spec.interface) for spec in specs)
    requires_counts = Counter(",".join(spec.requires) or "core" for spec in specs)

    lines: list[str] = []
    lines.append("# Model Capability Matrix")
    lines.append("")
    lines.append(
        "This page is generated from the central model registry. Do not edit it by hand; run "
        "`python tools/generate_model_capability_docs.py`."
    )
    lines.append("")
    lines.append(
        "ForeSight currently registers "
        f"**{len(specs)}** models: "
        f"**{interface_counts.get('local', 0)}** local, "
        f"**{interface_counts.get('global', 0)}** global, and "
        f"**{interface_counts.get('multivariate', 0)}** multivariate."
    )
    lines.append("")
    lines.append("## Capability keys")
    lines.append("")
    lines.append("| key | meaning |")
    lines.append("| --- | --- |")
    for name in capability_keys:
        lines.append(f"| `{name}` | {_CAPABILITY_DESCRIPTIONS[name]} |")
    lines.append("")
    lines.append("## Query the same metadata from code")
    lines.append("")
    lines.append("```bash")
    lines.append("foresight models list --format json")
    lines.append("foresight models info xgb-step-lag-global")
    lines.append("```")
    lines.append("")
    lines.append("```python")
    lines.append("from foresight.models import get_model_spec, list_models")
    lines.append("")
    lines.append('spec = get_model_spec("xgb-step-lag-global")')
    lines.append("print(spec.capabilities)")
    lines.append("print(len(list_models()))")
    lines.append("```")
    lines.append("")
    lines.append("## Dependency summary")
    lines.append("")
    lines.append("| requires extra | model_count |")
    lines.append("| --- | ---: |")
    for requires, count in sorted(requires_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{requires}` | {count} |")
    lines.append("")
    lines.append("## Full registry matrix")
    lines.append("")
    lines.append(
        "| key | interface | requires | `supports_x_cols` | `supports_quantiles` | "
        "`supports_interval_forecast` | `supports_interval_forecast_with_x_cols` | "
        "`supports_artifact_save` | `requires_future_covariates` |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for spec in specs:
        capabilities = dict(spec.capabilities)
        requires = ",".join(spec.requires) or "core"
        lines.append(
            f"| `{spec.key}` | `{spec.interface}` | `{requires}` | "
            f"{_bool_cell(capabilities.get('supports_x_cols'))} | "
            f"{_bool_cell(capabilities.get('supports_quantiles'))} | "
            f"{_bool_cell(capabilities.get('supports_interval_forecast'))} | "
            f"{_bool_cell(capabilities.get('supports_interval_forecast_with_x_cols'))} | "
            f"{_bool_cell(capabilities.get('supports_artifact_save'))} | "
            f"{_bool_cell(capabilities.get('requires_future_covariates'))} |"
        )

    return _normalized_text("\n".join(lines))


def _render_api_doc() -> str:
    root = _repo_root()
    _ensure_src_on_path(root)

    import foresight

    exports = list(foresight.__all__)
    missing = sorted(set(exports) - set(_API_METADATA))
    extra = sorted(set(_API_METADATA) - set(exports))
    if missing or extra:
        problems: list[str] = []
        if missing:
            problems.append("missing metadata for: " + ", ".join(missing))
        if extra:
            problems.append("metadata without export: " + ", ".join(extra))
        raise RuntimeError(
            "API doc metadata is out of sync with foresight.__all__: " + "; ".join(problems)
        )

    lines: list[str] = []
    lines.append("# Python API Reference")
    lines.append("")
    lines.append(
        "This page documents the stable root-package entry points exported by `foresight.__all__`. "
        "Import these names directly from `foresight`."
    )
    lines.append("")
    lines.append("```python")
    lines.append("from foresight import (")
    lines.append("    eval_model,")
    lines.append("    forecast_model,")
    lines.append("    load_forecaster_artifact,")
    lines.append("    make_forecaster_object,")
    lines.append("    save_forecaster,")
    lines.append(")")
    lines.append("```")
    lines.append("")

    for group in _API_GROUP_ORDER:
        names = [name for name in exports if _API_METADATA[name]["group"] == group]
        if not names:
            continue
        lines.append(f"## {group}")
        lines.append("")
        lines.append("| symbol | source | purpose |")
        lines.append("| --- | --- | --- |")
        for name in names:
            meta = _API_METADATA[name]
            lines.append(f"| `{name}` | `{meta['source']}` | {meta['purpose']} |")
        lines.append("")

    lines.append("## Notable data contracts")
    lines.append("")
    lines.append(
        "- `to_long(...)` accepts `historic_x_cols`, `future_x_cols`, and legacy `x_cols` (aliasing future covariates)."
    )
    lines.append(
        "- `prepare_long_df(...)` supports separate `historic_x_missing` / `future_x_missing` policies after role-aware conversion."
    )
    lines.append(
        "- `forecast_model_long_df(...)` accepts `future_df=...` so known-future covariates can arrive in a separate dataframe from observed history."
    )
    lines.append(
        "- Lag-based regression models accept either contiguous `lags=n` or explicit `target_lags=(1, 7, 14)`; the sklearn `*-step-lag-global` family also supports `historic_x_lags` / `future_x_lags` when `x_cols` are supplied."
    )
    lines.append(
        '- `reconcile_hierarchical_forecasts(...)` supports `exog_agg={"promo": "sum", "temp": "mean"}` for bottom-up exogenous aggregation.'
    )
    lines.append("")

    lines.append("## Root package export list")
    lines.append("")
    for name in exports:
        lines.append(f"- `{name}`")

    return _normalized_text("\n".join(lines))


def _expected_docs() -> dict[Path, str]:
    root = _repo_root()
    return {
        root / "docs" / "models.md": _render_models_doc(),
        root / "docs" / "api.md": _render_api_doc(),
    }


def _check_docs(expected: dict[Path, str]) -> list[Path]:
    stale: list[Path] = []
    for path, text in expected.items():
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        if actual != text:
            stale.append(path)
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate registry-driven public capability docs and API reference pages."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated docs differ from the checked-in files.",
    )
    args = parser.parse_args(argv)

    expected = _expected_docs()
    if args.check:
        stale = _check_docs(expected)
        if stale:
            rels = ", ".join(str(path.relative_to(_repo_root())) for path in stale)
            print(f"Generated docs are out of date: {rels}", file=sys.stderr)
            return 1
        print("OK: generated docs are up to date: docs/models.md, docs/api.md")
        return 0

    for path, text in expected.items():
        _write_text(path, text)
        print(f"Wrote: {path.relative_to(_repo_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
