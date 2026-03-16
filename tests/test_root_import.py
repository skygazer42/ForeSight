import json
import os
import subprocess
import sys
from pathlib import Path


def test_root_import_does_not_eagerly_import_registry() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import foresight; "
                "mods = [k for k in sys.modules if k == 'foresight.models.registry']; "
                "print(json.dumps(mods))"
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert proc.returncode == 0
    assert json.loads(proc.stdout) == []


def test_root_import_exports_stable_public_api() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import json; import foresight; print(json.dumps(sorted(foresight.__all__)))"),
        ],
        capture_output=True,
        text=True,
        env=env,
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
            "make_panel_sequence_tensors",
            "make_panel_window_arrays",
            "make_panel_window_frame",
            "make_supervised_frame",
            "make_forecaster",
            "make_forecaster_object",
            "make_global_forecaster",
            "make_global_forecaster_object",
            "make_multivariate_forecaster",
            "prepare_long_df",
            "reconcile_hierarchical_forecasts",
            "save_forecaster",
            "split_panel_sequence_tensors",
            "split_long_df",
            "to_long",
            "transform_long_df_with_scaler",
            "tune_model",
            "tune_model_long_df",
            "validate_long_df",
        ]
    )
