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


def test_datasets_list_outputs_store_sales():
    proc = _run_cli("datasets", "list")
    assert proc.returncode == 0
    assert "store_sales" in (proc.stdout + proc.stderr)


def test_datasets_preview_outputs_columns():
    proc = _run_cli("datasets", "preview", "store_sales", "--nrows", "50")
    assert proc.returncode == 0
    assert "sales" in (proc.stdout + proc.stderr)

