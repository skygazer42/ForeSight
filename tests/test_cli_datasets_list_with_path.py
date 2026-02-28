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


def test_datasets_list_with_path_mentions_csv_path():
    proc = _run_cli("datasets", "list", "--with-path")
    assert proc.returncode == 0
    out = proc.stdout + proc.stderr
    assert ".csv" in out
