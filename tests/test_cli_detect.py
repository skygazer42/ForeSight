from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_detect_run_outputs_json_rows() -> None:
    proc = _run_cli(
        "detect",
        "run",
        "--dataset",
        "catfish",
        "--y-col",
        "Total",
        "--model",
        "naive-last",
        "--score-method",
        "forecast-residual",
        "--threshold-method",
        "quantile",
        "--threshold-quantile",
        "0.95",
        "--min-train-size",
        "12",
        "--step-size",
        "3",
        "--n-windows",
        "4",
        "--format",
        "json",
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["model"] == "naive-last"
    assert payload[0]["score_method"] == "forecast-residual"
    assert "is_anomaly" in payload[0]


def test_detect_csv_supports_rolling_zscore_without_model(tmp_path: Path) -> None:
    csv_path = tmp_path / "anomaly.csv"
    csv_path.write_text(
        "\n".join(
            [
                "ds,y",
                "2020-01-01,2",
                "2020-01-02,2",
                "2020-01-03,2",
                "2020-01-04,2",
                "2020-01-05,2",
                "2020-01-06,2",
                "2020-01-07,2",
                "2020-01-08,2",
                "2020-01-09,2",
                "2020-01-10,20",
                "2020-01-11,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "detect",
        "csv",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--score-method",
        "rolling-zscore",
        "--threshold-method",
        "zscore",
        "--threshold-k",
        "3.0",
        "--window",
        "5",
        "--min-history",
        "3",
        "--format",
        "json",
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    flagged = [row for row in payload if row["is_anomaly"]]
    assert flagged
    assert any(str(row["ds"]).startswith("2020-01-10") for row in flagged)
    assert all(row["score_method"] == "rolling-zscore" for row in payload)


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_detect_csv_supports_local_sarimax_with_future_covariates(tmp_path: Path) -> None:
    csv_path = tmp_path / "sarimax_detect.csv"
    rows = ["ds,y,promo"]
    promo = ([0, 1, 0, 1, 0] * 6)[:30]
    for i, p in enumerate(promo, start=1):
        y = 10.0 + 5.0 * float(p)
        if i == 20:
            y += 12.0
        rows.append(f"2020-01-{i:02d},{y},{p}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    proc = _run_cli(
        "detect",
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
        "--score-method",
        "forecast-residual",
        "--threshold-method",
        "mad",
        "--min-train-size",
        "12",
        "--step-size",
        "1",
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
    flagged = [row for row in payload if row["is_anomaly"]]
    assert flagged
    assert any(str(row["ds"]).startswith("2020-01-20") for row in flagged)
    assert all(row["model"] == "sarimax" for row in payload)
    assert all(row["score_method"] == "forecast-residual" for row in payload)
