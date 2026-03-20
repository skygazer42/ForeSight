import os
import json
import subprocess
import sys
from pathlib import Path
import importlib.util

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


def test_cv_run_outputs_csv(tmp_path: Path):
    out = tmp_path / "cv.csv"
    proc = _run_cli(
        "cv",
        "run",
        "--model",
        "naive-last",
        "--dataset",
        "catfish",
        "--y-col",
        "Total",
        "--horizon",
        "2",
        "--step-size",
        "5",
        "--min-train-size",
        "12",
        "--n-windows",
        "2",
        "--format",
        "csv",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    assert proc.stdout.splitlines()[0].startswith("unique_id,ds,cutoff,step,y,yhat,model")
    assert out.exists()


def test_cv_csv_outputs_json_rows(tmp_path: Path):
    csv_path = tmp_path / "toy_cv.csv"
    csv_path.write_text(
        "ds,y\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n2020-01-04,4\n2020-01-05,5\n2020-01-06,6\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "cv",
        "csv",
        "--model",
        "naive-last",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "2",
        "--step-size",
        "1",
        "--min-train-size",
        "3",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["unique_id"] == "series=0"
    assert payload[0]["model"] == "naive-last"
    assert "y" in payload[0]
    assert "yhat" in payload[0]


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_cv_csv_supports_local_sarimax_with_x_cols(tmp_path: Path):
    csv_path = tmp_path / "sarimax_cv.csv"
    rows = ["ds,y,promo"]
    promo = ([0, 1, 0, 1, 0] * 6)[:30]
    for i, p in enumerate(promo, start=1):
        rows.append(f"2020-01-{i:02d},{10.0 + 5.0 * float(p)},{p}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    proc = _run_cli(
        "cv",
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
        "--step-size",
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
    assert isinstance(payload, list)
    assert payload
    assert all(row["model"] == "sarimax" for row in payload)
    assert max(abs(float(row["y"]) - float(row["yhat"])) for row in payload) < 1e-3
