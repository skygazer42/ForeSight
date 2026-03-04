from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cli_models_list_handles_broken_pipe_cleanly() -> None:
    """
    `python -m foresight models list | head -n 1` should not emit the noisy
    "Exception ignored ... BrokenPipeError" message when the downstream consumer
    closes stdout early.
    """

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    assert src_dir.exists()

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(src_dir), env.get("PYTHONPATH", "")]).strip(os.pathsep)

    proc = subprocess.Popen(
        [sys.executable, "-m", "foresight", "models", "list"],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    first_line = proc.stdout.readline()
    assert first_line.strip()

    # Simulate `head -n 1` closing the pipe early.
    proc.stdout.close()

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise

    err = proc.stderr.read()
    assert "BrokenPipeError" not in err
    assert "Exception ignored in:" not in err
