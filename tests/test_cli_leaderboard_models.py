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


def test_leaderboard_models_outputs_json_list(tmp_path: Path):
    out = tmp_path / "leaderboard_models.json"
    proc = _run_cli(
        "leaderboard",
        "models",
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
        "--models",
        "naive-last,seasonal-naive,mean",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    models = {row["model"] for row in payload}
    assert {"naive-last", "seasonal-naive", "mean"}.issubset(models)
    assert out.exists()


def test_leaderboard_models_outputs_csv(tmp_path: Path):
    out = tmp_path / "leaderboard_models.csv"
    proc = _run_cli(
        "leaderboard",
        "models",
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
        "--models",
        "naive-last,seasonal-naive",
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


def test_leaderboard_models_emits_phase_timing_logs(tmp_path: Path):
    out = tmp_path / "leaderboard_models_logs.json"
    proc = _run_cli(
        "leaderboard",
        "models",
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
        "--models",
        "naive-last,mean",
        "--log-style",
        "plain",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    stderr = proc.stderr
    assert "RUN start" in stderr
    assert "PHASE params" in stderr
    assert "PHASE prepare" in stderr
    assert "PHASE evaluate" in stderr
    assert "PHASE emit" in stderr
