from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

from foresight import __version__
from foresight.models import make_forecaster_object
from foresight.serialization import load_forecaster_artifact, save_forecaster


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cli_env() -> dict[str, str]:
    repo_root = _repo_root()
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        capture_output=True,
        text=True,
        env=_cli_env(),
    )


def test_root_public_surface_contract_matches_documented_exports() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; import foresight; print(json.dumps(sorted(foresight.__all__)))",
        ],
        capture_output=True,
        text=True,
        env=_cli_env(),
    )

    assert proc.returncode == 0
    assert json.loads(proc.stdout) == sorted(
        [
            "__version__",
            "align_long_df",
            "BaseForecaster",
            "BaseGlobalForecaster",
            "bootstrap_intervals",
            "build_hierarchy_spec",
            "check_hierarchical_consistency",
            "clip_long_df_outliers",
            "detect_anomalies",
            "detect_anomalies_long_df",
            "eval_hierarchical_forecast_df",
            "eval_model",
            "eval_model_long_df",
            "eval_multivariate_model_df",
            "enrich_long_df_calendar",
            "fit_long_df_scaler",
            "forecast_model",
            "forecast_model_long_df",
            "infer_series_frequency",
            "inverse_transform_long_df_with_scaler",
            "load_forecaster",
            "load_forecaster_artifact",
            "make_local_xreg_eval_bundle",
            "make_local_xreg_forecast_bundle",
            "make_panel_sequence_blocks",
            "make_panel_sequence_tensors",
            "make_panel_window_arrays",
            "make_panel_window_frame",
            "make_panel_window_predict_arrays",
            "make_panel_window_predict_frame",
            "make_supervised_arrays",
            "make_supervised_frame",
            "make_supervised_predict_arrays",
            "make_supervised_predict_frame",
            "make_forecaster",
            "make_forecaster_object",
            "make_global_forecaster",
            "make_global_forecaster_object",
            "make_multivariate_forecaster",
            "prepare_long_df",
            "reconcile_hierarchical_forecasts",
            "save_forecaster",
            "split_supervised_frame",
            "split_supervised_arrays",
            "split_panel_window_arrays",
            "split_panel_window_frame",
            "split_panel_sequence_blocks",
            "split_panel_sequence_tensors",
            "split_long_df",
            "to_long",
            "transform_long_df_with_scaler",
            "tune_model",
            "tune_model_long_df",
            "validate_long_df",
        ]
    )


def test_models_info_contract_exposes_support_metadata() -> None:
    proc = _run_cli("models", "info", "naive-last")

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["key"] == "naive-last"
    assert payload["required_extra"] == "core"
    assert payload["stability"] == "stable"
    assert payload["capabilities"]["supports_artifact_save"] is True
    assert payload["capabilities"]["supports_interval_forecast"] is True


def test_doctor_contract_exposes_machine_readable_environment_sections() -> None:
    proc = _run_cli("doctor")

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert "package" in payload
    assert "dependencies" in payload
    assert "datasets" in payload
    assert payload["summary"]["status"] in {"ok", "warn", "error"}


def test_artifact_contract_preserves_current_and_legacy_schema_versions(tmp_path: Path) -> None:
    forecaster = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    current_path = tmp_path / "current.pkl"
    legacy_path = tmp_path / "legacy.pkl"

    save_forecaster(forecaster, current_path)
    current = load_forecaster_artifact(current_path)

    with legacy_path.open("wb") as handle:
        pickle.dump(
            {
                "metadata": {
                    "package_version": __version__,
                    "model_key": "naive-last",
                    "model_params": {},
                    "train_schema": {"kind": "local", "n_obs": 5},
                },
                "forecaster": forecaster,
            },
            handle,
        )

    legacy = load_forecaster_artifact(legacy_path)

    assert current["artifact_schema_version"] == 1
    assert legacy["artifact_schema_version"] == 0


def test_support_contract_docs_define_public_surface_and_ci_matrix() -> None:
    repo_root = _repo_root()
    compatibility = (repo_root / "docs" / "compatibility.md").read_text(encoding="utf-8")
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "## Supported Public Surface" in compatibility
    assert "## CI-Backed Support Matrix" in compatibility
    assert "## Artifact Compatibility Contract" in compatibility
    assert "## Support Contract" in readme
    assert "stable public surface" in readme.lower()
