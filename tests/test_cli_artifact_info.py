from __future__ import annotations

import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import foresight.services.cli_workflows as workflows_mod
from foresight.models import make_forecaster_object
from foresight.pipeline import make_ensemble_object, make_pipeline_object
from foresight.serialization import save_forecaster


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_artifact_info_reports_local_artifact_metadata(tmp_path: Path) -> None:
    artifact_path = tmp_path / "naive-last.pkl"
    forecaster = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    save_forecaster(
        forecaster,
        artifact_path,
        extra={"artifact_type": "forecast-local", "unique_id": "series=0"},
    )

    proc = _run_cli("artifact", "info", "--artifact", str(artifact_path))

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["artifact_schema_version"] == 1
    assert payload["forecaster_type"] == "local"
    assert payload["is_fitted"] is True
    assert payload["metadata"]["model_key"] == "naive-last"
    assert payload["metadata"]["train_schema"]["kind"] == "local"
    assert payload["extra"]["artifact_type"] == "forecast-local"
    assert payload["extra"]["unique_id"] == "series=0"
    assert "RUN start" in proc.stderr
    assert "RUN done" in proc.stderr


def test_cli_artifact_info_surfaces_pipeline_composition_summary(tmp_path: Path) -> None:
    artifact_path = tmp_path / "pipeline.pkl"
    forecaster = make_pipeline_object(
        base="naive-last",
        transforms=("diff1", "standardize"),
    ).fit([1, 2, 3, 4, 5])
    save_forecaster(forecaster, artifact_path)

    proc = _run_cli("artifact", "info", "--artifact", str(artifact_path))

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["composition_summary"] == {
        "kind": "pipeline",
        "base": "naive-last",
        "transforms": ["diff1", "standardize"],
    }


def test_cli_artifact_info_surfaces_weighted_ensemble_composition_summary_in_markdown(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "ensemble.pkl"
    forecaster = make_ensemble_object(
        members={"baseline": "mean", "recent": "naive-last"},
        agg="mean",
        weights={"baseline": 1.0, "recent": 3.0},
    ).fit([1, 2, 3, 4, 5])
    save_forecaster(forecaster, artifact_path)

    proc = _run_cli(
        "artifact",
        "info",
        "--artifact",
        str(artifact_path),
        "--format",
        "markdown",
    )

    assert proc.returncode == 0
    assert "## Composition" in proc.stdout
    assert "| kind | ensemble |" in proc.stdout
    assert "| agg | mean |" in proc.stdout
    assert '| members | ["mean", "naive-last"] |' in proc.stdout
    assert '| member_names | ["baseline", "recent"] |' in proc.stdout
    assert "| weights | [1.0, 3.0] |" in proc.stdout


def test_cli_artifact_validate_reports_valid_local_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "naive-last.pkl"
    forecaster = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    save_forecaster(forecaster, artifact_path)

    proc = _run_cli("artifact", "validate", "--artifact", str(artifact_path))

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["valid"] is True
    assert payload["artifact_schema_version"] == 1
    assert payload["forecaster_type"] == "local"
    assert payload["model_key"] == "naive-last"


def test_cli_artifact_validate_rejects_forecast_local_xreg_artifact_missing_future_context(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "naive-last-xreg.pkl"
    forecaster = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    save_forecaster(
        forecaster,
        artifact_path,
        extra={
            "artifact_type": "forecast-local",
            "unique_id": "series=0",
            "ds": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            "x_cols": ["promo"],
            "train_exog": [[0.0], [1.0], [0.0], [1.0], [0.0]],
            "future_exog": [[1.0], [0.0]],
            "max_horizon": 2,
        },
    )

    proc = _run_cli("artifact", "validate", "--artifact", str(artifact_path))

    assert proc.returncode == 2
    assert "Local forecast artifact is missing required future ds context" in proc.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_cli_artifact_validate_rejects_forecast_global_artifact_missing_cutoff_context(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "panel_future.csv"
    artifact_path = tmp_path / "ridge-global.pkl"
    csv_path.write_text(
        "\n".join(
            [
                "store,ds,y,promo",
                "s0,2020-01-01,0.0,0",
                "s0,2020-01-02,0.5,0",
                "s0,2020-01-03,1.0,0",
                "s0,2020-01-04,1.5,0",
                "s0,2020-01-05,2.0,0",
                "s0,2020-01-06,2.5,0",
                "s0,2020-01-07,3.0,1",
                "s0,2020-01-08,3.5,0",
                "s0,2020-01-09,4.0,0",
                "s0,2020-01-10,4.5,0",
                "s0,2020-01-11,5.0,0",
                "s0,2020-01-12,5.5,0",
                "s0,2020-01-13,6.0,0",
                "s0,2020-01-14,6.5,1",
                "s0,2020-01-15,7.0,0",
                "s0,2020-01-16,7.5,0",
                "s0,2020-01-17,8.0,0",
                "s0,2020-01-18,8.5,0",
                "s0,2020-01-19,,1",
                "s0,2020-01-20,,0",
                "s1,2020-01-01,1.0,0",
                "s1,2020-01-02,1.5,0",
                "s1,2020-01-03,2.0,0",
                "s1,2020-01-04,2.5,0",
                "s1,2020-01-05,3.0,0",
                "s1,2020-01-06,3.5,0",
                "s1,2020-01-07,4.0,1",
                "s1,2020-01-08,4.5,0",
                "s1,2020-01-09,5.0,0",
                "s1,2020-01-10,5.5,0",
                "s1,2020-01-11,6.0,0",
                "s1,2020-01-12,6.5,0",
                "s1,2020-01-13,7.0,0",
                "s1,2020-01-14,7.5,1",
                "s1,2020-01-15,8.0,0",
                "s1,2020-01-16,8.5,0",
                "s1,2020-01-17,9.0,0",
                "s1,2020-01-18,9.5,0",
                "s1,2020-01-19,,1",
                "s1,2020-01-20,,0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--id-cols",
        "store",
        "--parse-dates",
        "--horizon",
        "2",
        "--model-param",
        "lags=5",
        "--model-param",
        "alpha=0.5",
        "--model-param",
        "x_cols=promo",
        "--model-param",
        "add_time_features=true",
        "--model-param",
        "id_feature=ordinal",
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )
    assert fit_proc.returncode == 0

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["extra"] = dict(payload.get("extra", {}))
    payload["extra"].pop("cutoff", None)
    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle)

    proc = _run_cli("artifact", "validate", "--artifact", str(artifact_path))

    assert proc.returncode == 2
    assert "Global artifact prediction requires a cutoff" in proc.stderr


def test_cli_artifact_validate_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    artifact_path = tmp_path / "naive-last.pkl"
    forecaster = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    save_forecaster(forecaster, artifact_path)

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["artifact_schema_version"] = 999
    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle)

    proc = _run_cli("artifact", "validate", "--artifact", str(artifact_path))

    assert proc.returncode == 2
    assert "Unsupported artifact schema version" in proc.stderr


def test_cli_artifact_info_rejects_non_dict_extra_payload(tmp_path: Path) -> None:
    artifact_path = tmp_path / "naive-last.pkl"
    forecaster = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    save_forecaster(forecaster, artifact_path)

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["extra"] = []
    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle)

    proc = _run_cli("artifact", "info", "--artifact", str(artifact_path))

    assert proc.returncode == 2
    assert "extra" in proc.stderr


def test_cli_artifact_info_promotes_runtime_tracking_summary(tmp_path: Path) -> None:
    artifact_path = tmp_path / "naive-last.pkl"
    forecaster = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    save_forecaster(forecaster, artifact_path)

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["metadata"]["train_schema"]["runtime"] = {
        "family": "torch",
        "tracking": {
            "tensorboard": {"log_dir": "/tmp/runs", "run_name": "demo-tb"},
            "mlflow": {"experiment_name": "foresight-exp", "run_name": "demo-mlflow"},
            "wandb": {"project": "foresight", "run_name": "demo-wandb", "mode": "offline"},
        },
    }
    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle)

    proc = _run_cli("artifact", "info", "--artifact", str(artifact_path))

    assert proc.returncode == 0
    result = json.loads(proc.stdout)
    assert result["tracking_backends"] == ["mlflow", "tensorboard", "wandb"]
    assert result["tracking_summary"] == {
        "mlflow": "foresight-exp / demo-mlflow",
        "tensorboard": "demo-tb @ /tmp/runs",
        "wandb": "foresight / demo-wandb [offline]",
    }
    assert result["tracking"]["tensorboard"]["run_name"] == "demo-tb"
    assert result["tracking"]["mlflow"]["experiment_name"] == "foresight-exp"
    assert result["tracking"]["wandb"]["project"] == "foresight"
    runtime = result["metadata"]["train_schema"]["runtime"]
    assert runtime["family"] == "torch"
    assert "tracking" not in runtime


def test_cli_artifact_info_can_emit_grouped_markdown_summary(tmp_path: Path) -> None:
    artifact_path = tmp_path / "naive-last.pkl"
    forecaster = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    save_forecaster(forecaster, artifact_path)

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["metadata"]["train_schema"]["runtime"] = {
        "family": "torch",
        "tracking": {
            "tensorboard": {"log_dir": "/tmp/runs", "run_name": "demo-tb"},
            "mlflow": {"experiment_name": "foresight-exp", "run_name": "demo-mlflow"},
        },
    }
    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle)

    proc = _run_cli(
        "artifact",
        "info",
        "--artifact",
        str(artifact_path),
        "--format",
        "markdown",
    )

    assert proc.returncode == 0
    assert "## Summary" in proc.stdout
    assert "| artifact_schema_version | 1 |" in proc.stdout
    assert "| model_key | naive-last |" in proc.stdout
    assert "## Tracking" in proc.stdout
    assert "| backend | summary |" in proc.stdout
    assert "| backends | mlflow, tensorboard |" in proc.stdout
    assert "| mlflow | foresight-exp / demo-mlflow |" in proc.stdout
    assert "| tensorboard | demo-tb @ /tmp/runs |" in proc.stdout
    assert "## Tracking Details" in proc.stdout
    assert "| tensorboard.run_name | demo-tb |" in proc.stdout
    assert "## Metadata" in proc.stdout
    assert "| train_schema.runtime.family | torch |" in proc.stdout


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_cli_artifact_info_reports_future_override_schema_for_global_forecast_artifact(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "panel_future.csv"
    artifact_path = tmp_path / "ridge-global.pkl"
    csv_path.write_text(
        "\n".join(
            [
                "store,ds,y,promo",
                "s0,2020-01-01,0.0,0",
                "s0,2020-01-02,0.5,0",
                "s0,2020-01-03,1.0,0",
                "s0,2020-01-04,1.5,0",
                "s0,2020-01-05,2.0,0",
                "s0,2020-01-06,2.5,0",
                "s0,2020-01-07,3.0,1",
                "s0,2020-01-08,3.5,0",
                "s0,2020-01-09,4.0,0",
                "s0,2020-01-10,4.5,0",
                "s0,2020-01-11,5.0,0",
                "s0,2020-01-12,5.5,0",
                "s0,2020-01-13,6.0,0",
                "s0,2020-01-14,6.5,1",
                "s0,2020-01-15,7.0,0",
                "s0,2020-01-16,7.5,0",
                "s0,2020-01-17,8.0,0",
                "s0,2020-01-18,8.5,0",
                "s0,2020-01-19,,1",
                "s0,2020-01-20,,0",
                "s1,2020-01-01,1.0,0",
                "s1,2020-01-02,1.5,0",
                "s1,2020-01-03,2.0,0",
                "s1,2020-01-04,2.5,0",
                "s1,2020-01-05,3.0,0",
                "s1,2020-01-06,3.5,0",
                "s1,2020-01-07,4.0,1",
                "s1,2020-01-08,4.5,0",
                "s1,2020-01-09,5.0,0",
                "s1,2020-01-10,5.5,0",
                "s1,2020-01-11,6.0,0",
                "s1,2020-01-12,6.5,0",
                "s1,2020-01-13,7.0,0",
                "s1,2020-01-14,7.5,1",
                "s1,2020-01-15,8.0,0",
                "s1,2020-01-16,8.5,0",
                "s1,2020-01-17,9.0,0",
                "s1,2020-01-18,9.5,0",
                "s1,2020-01-19,,1",
                "s1,2020-01-20,,0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--id-cols",
        "store",
        "--parse-dates",
        "--horizon",
        "2",
        "--model-param",
        "lags=5",
        "--model-param",
        "alpha=0.5",
        "--model-param",
        "x_cols=promo",
        "--model-param",
        "add_time_features=true",
        "--model-param",
        "id_feature=ordinal",
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )
    assert fit_proc.returncode == 0

    proc = _run_cli("artifact", "info", "--artifact", str(artifact_path))

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["future_override_schema"] == {
        "supported": True,
        "artifact_interface": "global",
        "default_context_source": "saved_future_context",
        "requires_time_col": True,
        "required_covariate_columns": ["promo"],
        "saved_id_cols": ["store"],
        "allow_unique_id_column": True,
        "allow_saved_raw_id_columns": True,
        "allow_omitting_id_columns": False,
        "max_horizon": 2,
    }

    md_proc = _run_cli(
        "artifact",
        "info",
        "--artifact",
        str(artifact_path),
        "--format",
        "markdown",
    )
    assert md_proc.returncode == 0
    assert "## Future Override" in md_proc.stdout
    assert '| required_covariate_columns | ["promo"] |' in md_proc.stdout
    assert '| saved_id_cols | ["store"] |' in md_proc.stdout


def test_cli_artifact_diff_reports_no_differences_for_identical_artifacts(tmp_path: Path) -> None:
    artifact_path = tmp_path / "naive-last.pkl"
    forecaster = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    save_forecaster(
        forecaster,
        artifact_path,
        extra={"artifact_type": "forecast-local", "unique_id": "series=0"},
    )

    proc = _run_cli(
        "artifact",
        "diff",
        "--left-artifact",
        str(artifact_path),
        "--right-artifact",
        str(artifact_path),
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["equal"] is True
    assert payload["difference_count"] == 0
    assert payload["differences"] == {}


def test_cli_artifact_diff_reports_metadata_and_extra_differences(tmp_path: Path) -> None:
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    save_forecaster(
        make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5]),
        left_path,
        extra={"artifact_type": "forecast-local", "unique_id": "series=0"},
    )
    save_forecaster(
        make_forecaster_object("mean").fit([1, 2, 3, 4, 5, 6]),
        right_path,
        extra={"artifact_type": "forecast-local", "unique_id": "series=1"},
    )

    proc = _run_cli(
        "artifact",
        "diff",
        "--left-artifact",
        str(left_path),
        "--right-artifact",
        str(right_path),
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["equal"] is False
    assert payload["difference_count"] >= 3
    assert payload["differences"]["metadata.model_key"] == {
        "left": "naive-last",
        "right": "mean",
    }
    assert payload["differences"]["metadata.train_schema.n_obs"] == {
        "left": 5,
        "right": 6,
    }
    assert payload["differences"]["extra.unique_id"] == {
        "left": "series=0",
        "right": "series=1",
    }


def test_cli_artifact_diff_can_filter_to_tracking_prefix(tmp_path: Path) -> None:
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    save_forecaster(
        make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5]),
        left_path,
    )
    save_forecaster(
        make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5]),
        right_path,
    )

    for path, run_name in ((left_path, "run-left"), (right_path, "run-right")):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        payload["metadata"]["train_schema"]["runtime"] = {
            "tracking": {
                "mlflow": {"experiment_name": "foresight-exp", "run_name": run_name},
                "wandb": {"project": "foresight", "run_name": run_name},
            }
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    proc = _run_cli(
        "artifact",
        "diff",
        "--left-artifact",
        str(left_path),
        "--right-artifact",
        str(right_path),
        "--path-prefix",
        "tracking",
        "--format",
        "csv",
    )

    assert proc.returncode == 0
    lines = [line.strip() for line in proc.stdout.strip().splitlines()]
    assert lines[0] == "path,left,right"
    assert "tracking.mlflow.run_name,run-left,run-right" in lines
    assert "tracking.wandb.run_name,run-left,run-right" in lines
    assert all(line == "path,left,right" or line.startswith("tracking.") for line in lines)


def test_cli_artifact_diff_can_emit_csv_rows(tmp_path: Path) -> None:
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    save_forecaster(
        make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5]),
        left_path,
        extra={"artifact_type": "forecast-local", "unique_id": "series=0"},
    )
    save_forecaster(
        make_forecaster_object("mean").fit([1, 2, 3, 4, 5, 6]),
        right_path,
        extra={"artifact_type": "forecast-local", "unique_id": "series=1"},
    )

    proc = _run_cli(
        "artifact",
        "diff",
        "--left-artifact",
        str(left_path),
        "--right-artifact",
        str(right_path),
        "--format",
        "csv",
    )

    assert proc.returncode == 0
    lines = [line.strip() for line in proc.stdout.strip().splitlines()]
    assert lines[0] == "path,left,right"
    assert "metadata.model_key,naive-last,mean" in lines
    assert "extra.unique_id,series=0,series=1" in lines


def test_cli_artifact_diff_can_emit_grouped_markdown_report(tmp_path: Path) -> None:
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    save_forecaster(
        make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5]),
        left_path,
        extra={"artifact_type": "forecast-local", "unique_id": "series=0"},
    )
    save_forecaster(
        make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5]),
        right_path,
        extra={"artifact_type": "forecast-local", "unique_id": "series=1"},
    )

    for path, run_name, log_dir, tag in (
        (left_path, "run-left", "/tmp/runs-left", "left-tag"),
        (right_path, "run-right", "/tmp/runs-right", "right-tag"),
    ):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        payload["metadata"]["tag"] = tag
        payload["metadata"]["train_schema"]["runtime"] = {
            "family": "torch",
            "tracking": {
                "tensorboard": {"log_dir": log_dir, "run_name": run_name},
                "mlflow": {"experiment_name": "foresight-exp", "run_name": run_name},
            },
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    proc = _run_cli(
        "artifact",
        "diff",
        "--left-artifact",
        str(left_path),
        "--right-artifact",
        str(right_path),
        "--format",
        "md",
    )

    assert proc.returncode == 0
    assert "## Summary" in proc.stdout
    assert "| equal | false |" in proc.stdout
    assert "## Tracking Summary" in proc.stdout
    assert "| backend | left | right |" in proc.stdout
    assert "| mlflow | foresight-exp / run-left | foresight-exp / run-right |" in proc.stdout
    assert (
        "| tensorboard | run-left @ /tmp/runs-left | run-right @ /tmp/runs-right |" in proc.stdout
    )
    assert "## Tracking Details" in proc.stdout
    assert "| tensorboard.run_name | run-left | run-right |" in proc.stdout
    assert "## Metadata" in proc.stdout
    assert "| tag | left-tag | right-tag |" in proc.stdout
    assert "| train_schema.runtime.family | torch | torch |" not in proc.stdout
    assert "## Extra" in proc.stdout
    assert "| unique_id | series=0 | series=1 |" in proc.stdout


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_cli_artifact_diff_surfaces_future_override_contract_changes(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "panel_future.csv"
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    csv_path.write_text(
        "\n".join(
            [
                "store,ds,y,promo",
                "s0,2020-01-01,0.0,0",
                "s0,2020-01-02,0.5,0",
                "s0,2020-01-03,1.0,0",
                "s0,2020-01-04,1.5,0",
                "s0,2020-01-05,2.0,0",
                "s0,2020-01-06,2.5,0",
                "s0,2020-01-07,3.0,1",
                "s0,2020-01-08,3.5,0",
                "s0,2020-01-09,4.0,0",
                "s0,2020-01-10,4.5,0",
                "s0,2020-01-11,5.0,0",
                "s0,2020-01-12,5.5,0",
                "s0,2020-01-13,6.0,0",
                "s0,2020-01-14,6.5,1",
                "s0,2020-01-15,7.0,0",
                "s0,2020-01-16,7.5,0",
                "s0,2020-01-17,8.0,0",
                "s0,2020-01-18,8.5,0",
                "s0,2020-01-19,,1",
                "s0,2020-01-20,,0",
                "s1,2020-01-01,1.0,0",
                "s1,2020-01-02,1.5,0",
                "s1,2020-01-03,2.0,0",
                "s1,2020-01-04,2.5,0",
                "s1,2020-01-05,3.0,0",
                "s1,2020-01-06,3.5,0",
                "s1,2020-01-07,4.0,1",
                "s1,2020-01-08,4.5,0",
                "s1,2020-01-09,5.0,0",
                "s1,2020-01-10,5.5,0",
                "s1,2020-01-11,6.0,0",
                "s1,2020-01-12,6.5,0",
                "s1,2020-01-13,7.0,0",
                "s1,2020-01-14,7.5,1",
                "s1,2020-01-15,8.0,0",
                "s1,2020-01-16,8.5,0",
                "s1,2020-01-17,9.0,0",
                "s1,2020-01-18,9.5,0",
                "s1,2020-01-19,,1",
                "s1,2020-01-20,,0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--id-cols",
        "store",
        "--parse-dates",
        "--horizon",
        "2",
        "--model-param",
        "lags=5",
        "--model-param",
        "alpha=0.5",
        "--model-param",
        "x_cols=promo",
        "--model-param",
        "add_time_features=true",
        "--model-param",
        "id_feature=ordinal",
        "--format",
        "json",
        "--save-artifact",
        str(left_path),
    )
    assert fit_proc.returncode == 0

    with left_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["extra"] = dict(payload.get("extra", {}))
    payload["extra"]["id_cols"] = []
    with right_path.open("wb") as handle:
        pickle.dump(payload, handle)

    proc = _run_cli(
        "artifact",
        "diff",
        "--left-artifact",
        str(left_path),
        "--right-artifact",
        str(right_path),
    )
    assert proc.returncode == 0
    diff_payload = json.loads(proc.stdout)
    assert diff_payload["differences"]["future_override_schema.saved_id_cols"] == {
        "left": ["store"],
        "right": [],
    }
    assert diff_payload["differences"]["future_override_schema.allow_saved_raw_id_columns"] == {
        "left": True,
        "right": False,
    }

    md_proc = _run_cli(
        "artifact",
        "diff",
        "--left-artifact",
        str(left_path),
        "--right-artifact",
        str(right_path),
        "--format",
        "markdown",
    )
    assert md_proc.returncode == 0
    assert "## Future Override" in md_proc.stdout
    assert '| saved_id_cols | ["store"] | [] |' in md_proc.stdout
    assert "| allow_saved_raw_id_columns | true | false |" in md_proc.stdout
    assert "## Other" not in md_proc.stdout or "future_override_schema" not in md_proc.stdout

    prefix_md_proc = _run_cli(
        "artifact",
        "diff",
        "--left-artifact",
        str(left_path),
        "--right-artifact",
        str(right_path),
        "--path-prefix",
        "future_override_schema",
        "--format",
        "markdown",
    )
    assert prefix_md_proc.returncode == 0
    assert "## Summary" in prefix_md_proc.stdout
    assert "| equal | false |" in prefix_md_proc.stdout
    assert "| difference_count | 2 |" in prefix_md_proc.stdout
    assert "| path_prefix | future_override_schema |" in prefix_md_proc.stdout
    assert "## Future Override" in prefix_md_proc.stdout
    assert '| saved_id_cols | ["store"] | [] |' in prefix_md_proc.stdout
    assert "| allow_saved_raw_id_columns | true | false |" in prefix_md_proc.stdout
    assert "## Tracking Summary" not in prefix_md_proc.stdout
    assert "## Tracking Details" not in prefix_md_proc.stdout
    assert "## Metadata" not in prefix_md_proc.stdout
    assert "## Extra" not in prefix_md_proc.stdout
    assert "## Other" not in prefix_md_proc.stdout


def test_cli_artifact_diff_accepts_markdown_alias_and_emits_compact_no_diff_summary(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "naive-last.pkl"
    save_forecaster(
        make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5]),
        artifact_path,
        extra={"artifact_type": "forecast-local", "unique_id": "series=0"},
    )

    proc = _run_cli(
        "artifact",
        "diff",
        "--left-artifact",
        str(artifact_path),
        "--right-artifact",
        str(artifact_path),
        "--format",
        "markdown",
    )

    assert proc.returncode == 0
    assert "## Summary" in proc.stdout
    assert "No differences found." in proc.stdout
    assert "| field | value |" not in proc.stdout
    assert "## Tracking Summary" not in proc.stdout
    assert "## Metadata" not in proc.stdout


def test_cli_artifact_diff_emits_compact_no_diff_summary_for_future_override_path_prefix(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "naive-last.pkl"
    save_forecaster(
        make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5]),
        artifact_path,
        extra={"artifact_type": "forecast-local", "unique_id": "series=0"},
    )

    proc = _run_cli(
        "artifact",
        "diff",
        "--left-artifact",
        str(artifact_path),
        "--right-artifact",
        str(artifact_path),
        "--path-prefix",
        "future_override_schema",
        "--format",
        "markdown",
    )

    assert proc.returncode == 0
    assert "## Summary" in proc.stdout
    assert "No differences found under `future_override_schema`." in proc.stdout
    assert "| field | value |" not in proc.stdout
    assert "## Future Override" not in proc.stdout
    assert "## Metadata" not in proc.stdout


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_cli_artifact_diff_can_filter_to_runtime_path_prefix(tmp_path: Path) -> None:
    left_path = tmp_path / "torch-left.pkl"
    right_path = tmp_path / "torch-right.pkl"
    y = np.sin(np.arange(64, dtype=float) / 6.0)

    save_forecaster(
        make_forecaster_object(
            "torch-multidim-rnn-direct",
            lags=16,
            hidden_size=8,
            epochs=1,
            batch_size=8,
            optimizer="adamw",
            seed=0,
            device="cpu",
        ).fit(y),
        left_path,
    )
    save_forecaster(
        make_forecaster_object(
            "torch-multidim-rnn-direct",
            lags=16,
            hidden_size=8,
            epochs=1,
            batch_size=12,
            optimizer="adamw",
            seed=0,
            device="cpu",
        ).fit(y),
        right_path,
    )

    proc = _run_cli(
        "artifact",
        "diff",
        "--left-artifact",
        str(left_path),
        "--right-artifact",
        str(right_path),
        "--path-prefix",
        "metadata.train_schema.runtime",
        "--format",
        "csv",
    )

    assert proc.returncode == 0
    lines = [line.strip() for line in proc.stdout.strip().splitlines()]
    assert lines[0] == "path,left,right"
    assert "metadata.train_schema.runtime.training.batch_size,8,12" in lines
    assert "metadata.model_params.batch_size,8,12" not in lines
    assert all(
        line == "path,left,right" or line.startswith("metadata.train_schema.runtime.")
        for line in lines
    )


def test_readme_documents_artifact_inspection_commands() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "foresight artifact info --artifact /tmp/naive-last.pkl" in readme
    assert "--format markdown" in readme
    assert "foresight artifact validate --artifact /tmp/naive-last.pkl" in readme
    assert "--path-prefix metadata.train_schema.runtime" in readme
    assert "--path-prefix tracking_summary" in readme
    assert "--path-prefix future_override_schema" in readme
    assert "tracking_summary" in readme
    assert "Only load artifacts from trusted sources." in readme


def test_docs_index_documents_artifact_inspection_commands() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs_index = (repo_root / "docs" / "index.md").read_text(encoding="utf-8")

    assert "foresight artifact info --artifact /tmp/naive-last.pkl" in docs_index
    assert "--format markdown" in docs_index
    assert "foresight artifact validate --artifact /tmp/naive-last.pkl" in docs_index
    assert "foresight artifact diff" in docs_index
    assert "--path-prefix tracking_summary" in docs_index
    assert "--path-prefix future_override_schema" in docs_index
    assert "Only load artifacts from trusted sources." in docs_index


def test_docs_artifacts_page_documents_artifact_workflow() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs_index = (repo_root / "docs" / "index.md").read_text(encoding="utf-8")
    artifacts_doc = (repo_root / "docs" / "artifacts.md").read_text(encoding="utf-8")

    assert "[Artifact workflow](artifacts.md)" in docs_index
    assert "foresight artifact info --artifact /tmp/naive-last.pkl" in artifacts_doc
    assert "--format markdown" in artifacts_doc
    assert "foresight artifact validate --artifact /tmp/naive-last.pkl" in artifacts_doc
    assert "--path-prefix metadata.train_schema.runtime" in artifacts_doc
    assert "--path-prefix tracking_summary" in artifacts_doc
    assert "--path-prefix future_override_schema" in artifacts_doc
    assert "tracking_summary" in artifacts_doc
    assert "load_forecaster_artifact" in artifacts_doc
    assert "Only load artifacts from trusted sources." in artifacts_doc


def test_cli_and_legacy_docs_warn_about_trusted_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cli_index = (repo_root / "docs" / "cli" / "index.md").read_text(encoding="utf-8")
    legacy_index = (repo_root / "docs" / "legacy-index.md").read_text(encoding="utf-8")

    assert "Only load artifacts from trusted sources." in cli_index
    assert "Only load artifacts from trusted sources." in legacy_index


def test_api_docs_describe_artifact_loading_as_pickle_backed_and_trusted_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    api_doc = (repo_root / "docs" / "api.md").read_text(encoding="utf-8")
    api_reference = (repo_root / "docs" / "api-reference" / "index.md").read_text(encoding="utf-8")

    assert "pickle-backed artifact payload" in api_doc
    assert "Only load artifacts from trusted sources." in api_reference
    assert "不恢复预测器对象" not in api_reference


def test_guide_and_cli_reference_warn_on_artifact_inspection_commands() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    guide = (repo_root / "docs" / "guide" / "artifacts.md").read_text(encoding="utf-8")
    cli_index = (repo_root / "docs" / "cli" / "index.md").read_text(encoding="utf-8")

    diff_start = cli_index.index("## artifact diff")
    diff_end = cli_index.find("\n---", diff_start)
    diff_section = cli_index[diff_start : diff_end if diff_end != -1 else len(cli_index)]

    assert "Only load artifacts from trusted sources." in guide
    assert "可信来源" in guide
    assert "Only load artifacts from trusted sources." in diff_section


def test_cli_artifact_help_warns_about_trusted_artifacts() -> None:
    commands = [
        ("artifact", "--help"),
        ("forecast", "artifact", "--help"),
        ("artifact", "info", "--help"),
        ("artifact", "validate", "--help"),
        ("artifact", "diff", "--help"),
    ]

    for args in commands:
        proc = _run_cli(*args)
        assert proc.returncode == 0
        normalized_help = " ".join(proc.stdout.split())
        assert "Only load artifacts from trusted sources." in normalized_help


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_cli_artifact_info_exposes_structured_torch_runtime_metadata(tmp_path: Path) -> None:
    artifact_path = tmp_path / "torch-local.pkl"
    y = np.sin(np.arange(64, dtype=float) / 6.0)
    forecaster = make_forecaster_object(
        "torch-multidim-rnn-direct",
        lags=16,
        hidden_size=8,
        epochs=1,
        batch_size=8,
        optimizer="adamw",
        scheduler="plateau",
        scheduler_patience=1,
        scheduler_plateau_factor=0.5,
        seed=0,
        device="cpu",
    ).fit(y)
    save_forecaster(forecaster, artifact_path)

    proc = _run_cli("artifact", "info", "--artifact", str(artifact_path))

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    runtime = payload["metadata"]["train_schema"]["runtime"]
    assert runtime["family"] == "torch"
    assert runtime["device"] == "cpu"
    assert runtime["optimizer"]["name"] == "adamw"
    assert runtime["scheduler"]["name"] == "plateau"
    assert runtime["prediction"]["mode"] == "point"


def test_flatten_artifact_summary_rows_uses_iterative_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = workflows_mod._flatten_artifact_summary_rows
    calls = {"count": 0}

    def _counting_flatten(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(workflows_mod, "_flatten_artifact_summary_rows", _counting_flatten)

    rows = workflows_mod._flatten_artifact_summary_rows(
        {
            "metadata": {
                "model_key": "naive-last",
                "train_schema": {
                    "kind": "local",
                    "runtime": {"family": "core"},
                },
            },
            "extra": {"artifact_type": "forecast-local"},
        }
    )

    assert calls["count"] == 1
    assert rows == [
        {"field": "extra.artifact_type", "value": "forecast-local"},
        {"field": "metadata.model_key", "value": "naive-last"},
        {"field": "metadata.train_schema.kind", "value": "local"},
        {"field": "metadata.train_schema.runtime.family", "value": "core"},
    ]
