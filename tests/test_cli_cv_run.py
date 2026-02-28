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
