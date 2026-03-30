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


def test_eval_run_naive_last_outputs_json(tmp_path: Path):
    out = tmp_path / "metrics_run.json"
    proc = _run_cli(
        "eval",
        "run",
        "--model",
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
    assert payload["n_windows"] > 0
    assert out.exists()


def test_eval_run_model_param_parsing(tmp_path: Path):
    out = tmp_path / "metrics_run_seasonal.json"
    proc = _run_cli(
        "eval",
        "run",
        "--model",
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
        "--model-param",
        "season_length=12",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["model"] == "seasonal-naive"
    assert payload["n_windows"] > 0
    assert out.exists()


def test_eval_run_emits_phase_timing_logs(tmp_path: Path):
    out = tmp_path / "metrics_run.json"
    proc = _run_cli(
        "eval",
        "run",
        "--model",
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
        "--log-style",
        "plain",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    stderr = proc.stderr
    assert "PHASE params" in stderr
    assert "PHASE eval" in stderr
    assert "PHASE emit" in stderr
    assert "elapsed_ms=" in stderr
