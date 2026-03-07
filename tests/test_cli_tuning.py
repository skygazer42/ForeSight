import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


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


def test_tuning_run_outputs_json_summary(tmp_path: Path) -> None:
    root = tmp_path / "root"
    out = tmp_path / "tuning.json"
    (root / "data").mkdir(parents=True)

    weeks = pd.date_range("2020-01-01", periods=10, freq="W-WED")
    df = pd.DataFrame(
        {
            "store": [1] * 10,
            "dept": [1] * 10,
            "week": [d.date().isoformat() for d in weeks],
            "sales": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    (root / "data" / "store_sales.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    proc = _run_cli(
        "--data-dir",
        str(root),
        "tuning",
        "run",
        "--model",
        "moving-average",
        "--dataset",
        "store_sales",
        "--y-col",
        "sales",
        "--horizon",
        "1",
        "--step",
        "1",
        "--min-train-size",
        "4",
        "--max-windows",
        "3",
        "--grid-param",
        "window=1,3",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["best_params"] == {"window": 1}
    assert payload["metric"] == "mae"
    assert payload["n_trials"] == 2
    assert out.exists()
