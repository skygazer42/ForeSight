#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    # tools/release_check.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> int:
    root = _repo_root()

    env = dict(os.environ)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    # Code quality + tests.
    _run(["ruff", "check", "src", "tests", "tools"], cwd=root, env=env)
    _run(["ruff", "format", "--check", "src", "tests", "tools"], cwd=root, env=env)
    _run([sys.executable, "-m", "pytest", "-q"], cwd=root, env=env)

    # Build in a clean temp dir (avoids `rm -rf dist/` patterns).
    with tempfile.TemporaryDirectory(prefix="foresight_release_build_") as tmp:
        dist_dir = Path(tmp) / "dist"
        _run([sys.executable, "-m", "build", "--outdir", str(dist_dir)], cwd=root, env=env)

        # Optional: validate artifacts if twine is installed.
        if importlib.util.find_spec("twine") is not None:
            artifacts = sorted(dist_dir.glob("*.whl")) + sorted(dist_dir.glob("*.tar.gz"))
            if not artifacts:
                raise RuntimeError(f"No artifacts found under: {dist_dir}")
            _run(
                [sys.executable, "-m", "twine", "check", *[str(p) for p in artifacts]],
                cwd=root,
                env=env,
            )
        else:
            print("NOTE: twine not installed; skipping `twine check`.", flush=True)

    print("OK: release checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
