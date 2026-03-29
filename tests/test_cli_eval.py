import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import foresight


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


def test_eval_naive_last_outputs_json(tmp_path: Path):
    out = tmp_path / "metrics.json"
    proc = _run_cli(
        "eval",
        "naive-last",
        "--dataset",
        "catfish",
        "--y-col",
        "Total",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["model"] == "naive-last"
    assert payload["dataset"] == "catfish"
    assert "mae" in payload
    assert out.exists()


def test_eval_seasonal_naive_outputs_json(tmp_path: Path):
    out = tmp_path / "metrics_seasonal.json"
    proc = _run_cli(
        "eval",
        "seasonal-naive",
        "--dataset",
        "catfish",
        "--y-col",
        "Total",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--season-length",
        "12",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["model"] == "seasonal-naive"
    assert payload["dataset"] == "catfish"
    assert "mae" in payload
    assert out.exists()


def test_docs_install_mentions_forecast_tuning_and_detect_smoke_commands():
    repo_root = Path(__file__).resolve().parents[1]
    doc = (repo_root / "docs" / "INSTALL.md").read_text(encoding="utf-8")

    assert "python -m foresight doctor" in doc
    assert "python -m foresight doctor --format text" in doc
    assert "python -m foresight --data-dir /path/to/root doctor --format text --strict" in doc
    assert "python -m foresight doctor --require-extra torch --strict" in doc
    assert "python -m foresight forecast --help" in doc
    assert "python -m foresight tuning --help" in doc
    assert "python -m foresight forecast csv --help" in doc
    assert "python -m foresight detect --help" in doc
    assert "    make_forecaster," in doc
    assert "    eval_model," in doc


def test_root_package_exports_high_level_forecasting_helpers():
    expected = {
        "bootstrap_intervals",
        "forecast_model",
        "forecast_model_long_df",
        "make_forecaster",
        "make_forecaster_object",
        "make_global_forecaster",
        "make_global_forecaster_object",
        "save_forecaster",
        "load_forecaster",
        "prepare_long_df",
        "tune_model",
    }

    for name in expected:
        assert name in dir(foresight)
        assert getattr(foresight, name) is not None


def test_examples_use_root_package_for_high_level_workflow_helpers():
    repo_root = Path(__file__).resolve().parents[1]
    quickstart = (repo_root / "examples" / "quickstart_eval.py").read_text(encoding="utf-8")
    leaderboard = (repo_root / "examples" / "leaderboard.py").read_text(encoding="utf-8")
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "from foresight.eval_forecast import eval_model" not in quickstart
    assert "from foresight.forecast import forecast_model" not in quickstart
    assert "from foresight import eval_model, forecast_model" in quickstart
    assert "from foresight.models.registry import make_forecaster, make_global_forecaster" not in quickstart

    assert "from foresight.eval_forecast import eval_model" not in leaderboard
    assert "from foresight import eval_model" in leaderboard
    assert "from foresight.intervals import bootstrap_intervals" not in readme
    assert "    bootstrap_intervals," in readme
    assert "    detect_anomalies," in readme
    assert "    eval_hierarchical_forecast_df," in readme
    assert "foresight detect csv --path ./anomaly.csv" in readme
    assert "hier_payload = eval_hierarchical_forecast_df(" in readme


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_eval_csv_supports_local_sarimax_with_x_cols(tmp_path: Path):
    csv_path = tmp_path / "sarimax_eval.csv"
    rows = ["ds,y,promo"]
    promo = ([0, 1, 0, 1, 0] * 6)[:30]
    for i, p in enumerate(promo, start=1):
        rows.append(f"2020-01-{i:02d},{10.0 + 5.0 * float(p)},{p}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    proc = _run_cli(
        "eval",
        "csv",
        "--model",
        "sarimax",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--model-param",
        "order=0,0,0",
        "--model-param",
        "seasonal_order=0,0,0,0",
        "--model-param",
        "trend=c",
        "--model-param",
        "x_cols=promo",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    payload = json.loads(proc.stdout)
    assert payload["model"] == "sarimax"
    assert payload["n_series"] == 1
    assert payload["n_series_skipped"] == 0
    assert payload["n_points"] == 18
    assert payload["mae"] < 1e-3


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_eval_csv_supports_local_auto_arima_with_x_cols(tmp_path: Path):
    csv_path = tmp_path / "auto_arima_eval.csv"
    rows = ["ds,y,promo"]
    promo = ([0, 1, 0, 1, 0] * 6)[:30]
    for i, p in enumerate(promo, start=1):
        rows.append(f"2020-01-{i:02d},{10.0 + 5.0 * float(p)},{p}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    proc = _run_cli(
        "eval",
        "csv",
        "--model",
        "auto-arima",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--model-param",
        "max_p=0",
        "--model-param",
        "max_d=0",
        "--model-param",
        "max_q=0",
        "--model-param",
        "max_P=0",
        "--model-param",
        "max_D=0",
        "--model-param",
        "max_Q=0",
        "--model-param",
        "trend=c",
        "--model-param",
        "x_cols=promo",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    payload = json.loads(proc.stdout)
    assert payload["model"] == "auto-arima"
    assert payload["n_series"] == 1
    assert payload["n_series_skipped"] == 0
    assert payload["n_points"] == 18
    assert payload["mae"] < 1e-3
