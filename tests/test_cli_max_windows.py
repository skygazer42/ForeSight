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


def test_eval_respects_max_windows():
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
        "1",
        "--min-train-size",
        "12",
        "--max-windows",
        "4",
    )
    payload = json.loads(proc.stdout)
    assert payload["n_windows"] == 4

