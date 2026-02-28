import os
import subprocess
import sys
from pathlib import Path


def _run_cli(
    *args: str, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_unknown_dataset_key_is_friendly_without_debug():
    proc = _run_cli("datasets", "preview", "no_such_dataset")
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "Unknown dataset key" in out
    assert "Traceback" not in out


def test_unknown_dataset_key_shows_traceback_with_debug():
    proc = _run_cli("--debug", "datasets", "preview", "no_such_dataset")
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "Traceback" in out
