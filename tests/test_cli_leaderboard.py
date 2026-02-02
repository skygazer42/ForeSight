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


def test_leaderboard_naive_outputs_json_list(tmp_path: Path):
    out = tmp_path / "leaderboard.json"
    proc = _run_cli(
        "leaderboard",
        "naive",
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
    assert isinstance(payload, list)
    models = {row["model"] for row in payload}
    assert {"naive-last", "seasonal-naive"}.issubset(models)
    assert out.exists()


def test_leaderboard_naive_outputs_csv(tmp_path: Path):
    out = tmp_path / "leaderboard.csv"
    proc = _run_cli(
        "leaderboard",
        "naive",
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
        "--format",
        "csv",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    first_line = proc.stdout.splitlines()[0]
    assert first_line.startswith("model,")
    assert "naive-last" in proc.stdout
    assert out.exists()


def test_leaderboard_naive_outputs_markdown(tmp_path: Path):
    out = tmp_path / "leaderboard.md"
    proc = _run_cli(
        "leaderboard",
        "naive",
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
        "--format",
        "md",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    assert "| model |" in proc.stdout
    assert "naive-last" in proc.stdout
    assert out.exists()
