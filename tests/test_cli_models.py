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


def test_models_list_contains_naive_last():
    proc = _run_cli("models", "list")
    assert proc.returncode == 0
    assert "naive-last" in proc.stdout


def test_models_info_outputs_json():
    proc = _run_cli("models", "info", "naive-last")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["key"] == "naive-last"
