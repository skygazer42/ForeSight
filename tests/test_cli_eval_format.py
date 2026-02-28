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


def test_eval_naive_last_csv_output(tmp_path: Path):
    out = tmp_path / "metrics.csv"
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
        "--format",
        "csv",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    assert proc.stdout.splitlines()[0].startswith("model,")
    assert out.exists()


def test_eval_naive_last_md_output(tmp_path: Path):
    out = tmp_path / "metrics.md"
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
        "--format",
        "md",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    assert "| model |" in proc.stdout
    assert out.exists()

