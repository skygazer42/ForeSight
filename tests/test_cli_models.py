import json
import os
import subprocess
import sys
from pathlib import Path


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


def test_models_list_contains_naive_last():
    proc = _run_cli("models", "list")
    assert proc.returncode == 0
    assert "naive-last" in proc.stdout
    assert "weighted-moving-average" in proc.stdout
    assert "moving-median" in proc.stdout
    assert "seasonal-drift" in proc.stdout


def test_root_list_shortcut_matches_models_list():
    proc = _run_cli("--list")
    baseline = _run_cli("models", "list")

    assert proc.returncode == 0
    assert baseline.returncode == 0
    assert proc.stdout == baseline.stdout


def test_root_list_models_shortcut_matches_models_list():
    proc = _run_cli("--list-models")
    baseline = _run_cli("models", "list")

    assert proc.returncode == 0
    assert baseline.returncode == 0
    assert proc.stdout == baseline.stdout


def test_models_info_outputs_json():
    proc = _run_cli("models", "info", "naive-last")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["key"] == "naive-last"
    assert payload["required_extra"] == "core"
    assert payload["package_install_command"] == "pip install foresight-ts"
    assert payload["editable_install_command"] == "pip install -e ."
    assert payload["stability"] == "stable"
    assert payload["capabilities"]["supports_x_cols"] is False
    assert payload["capabilities"]["supports_interval_forecast"] is True
    assert payload["capabilities"]["supports_artifact_save"] is True


def test_root_help_mentions_forecast_and_tuning_commands():
    proc = _run_cli("--help")
    assert proc.returncode == 0
    assert "doctor" in proc.stdout
    assert "forecast" in proc.stdout
    assert "tuning" in proc.stdout


def test_docs_development_examples_use_current_cli_workflows():
    repo_root = Path(__file__).resolve().parents[1]
    doc = (repo_root / "docs" / "DEVELOPMENT.md").read_text(encoding="utf-8")

    assert "foresight eval run --model naive-last" in doc
    assert "foresight leaderboard models" in doc
    assert "foresight forecast csv --model naive-last" in doc
    assert "foresight tuning run --model moving-average" in doc
    assert "    make_forecaster," in doc
    assert "    eval_model," in doc
    assert "foresight eval naive-last" not in doc
    assert "foresight leaderboard naive" not in doc
