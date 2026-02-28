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


def test_eval_csv_outputs_json(tmp_path: Path):
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text(
        "ds,y\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n2020-01-04,4\n2020-01-05,5\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "eval",
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
        "--step",
        "1",
        "--min-train-size",
        "3",
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["model"] == "naive-last"
    assert payload["dataset"] == str(csv_path)
    assert payload["time_col"] == "ds"
    assert payload["y_col"] == "y"
