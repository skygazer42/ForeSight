import subprocess
import sys
import os
from pathlib import Path


def test_cli_help_exits_zero():
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    proc = subprocess.run(
        [sys.executable, "-m", "foresight", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert "ForeSight" in (proc.stdout + proc.stderr)
