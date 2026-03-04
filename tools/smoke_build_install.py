#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    # tools/smoke_build_install.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="smoke_build_install",
        description="Build wheel/sdist and run a minimal install+CLI smoke.",
    )
    p.add_argument(
        "--sdist",
        action="store_true",
        help="Also install from sdist (slower; may require network for build isolation).",
    )
    args = p.parse_args(argv)

    root = _repo_root()

    with tempfile.TemporaryDirectory(prefix="foresight_pkg_smoke_") as tmp:
        tmp_path = Path(tmp)
        dist_dir = tmp_path / "dist"
        venv_wheel_dir = tmp_path / "venv_wheel"

        # Build wheel+sdist in an isolated env.
        _run([sys.executable, "-m", "build", "--outdir", str(dist_dir)], cwd=root)

        # Avoid noisy pip self-update prompts in CI logs.
        env = dict(os.environ)
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

        def _smoke(py: Path) -> None:
            # Import smoke.
            _run(
                [str(py), "-c", "import foresight; print(foresight.__version__)"],
                cwd=root,
                env=env,
            )

            # CLI smoke (module invocation avoids PATH concerns).
            _run([str(py), "-m", "foresight", "--version"], cwd=root, env=env)
            _run(
                [str(py), "-m", "foresight", "models", "info", "torch-rnnpaper-elman-srn-direct"],
                cwd=root,
                env=env,
            )
            _run([str(py), "-m", "foresight", "papers", "info", "elman-srn"], cwd=root, env=env)
            _run(
                [
                    str(py),
                    "-m",
                    "foresight",
                    "papers",
                    "models",
                    "bahdanau-attention",
                    "--role",
                    "wrapper",
                    "--format",
                    "json",
                ],
                cwd=root,
                env=env,
            )
            _run(
                [str(py), "-m", "foresight", "datasets", "preview", "catfish", "--nrows", "3"],
                cwd=root,
                env=env,
            )
            _run(
                [
                    str(py),
                    "-m",
                    "foresight",
                    "leaderboard",
                    "sweep",
                    "--datasets",
                    "catfish,ice_cream_interest",
                    "--horizon",
                    "3",
                    "--step",
                    "3",
                    "--min-train-size",
                    "12",
                    "--max-windows",
                    "2",
                    "--models",
                    "naive-last,mean",
                    "--jobs",
                    "2",
                    "--backend",
                    "process",
                    "--progress",
                    "--chunk-size",
                    "0",
                    "--output",
                    str(tmp_path / f"sweep_{py.parent.parent.name}.json"),
                    "--summary-output",
                    str(tmp_path / f"summary_{py.parent.parent.name}.json"),
                    "--summary-format",
                    "json",
                    "--summary-min-datasets",
                    "2",
                    "--failures-output",
                    str(tmp_path / f"failures_{py.parent.parent.name}.txt"),
                ],
                cwd=root,
                env=env,
            )
            _run(
                [
                    str(py),
                    "-m",
                    "foresight",
                    "leaderboard",
                    "summarize",
                    "--input",
                    str(tmp_path / f"sweep_{py.parent.parent.name}.json"),
                    "--format",
                    "json",
                    "--min-datasets",
                    "2",
                ],
                cwd=root,
                env=env,
            )

            docs_out = tmp_path / f"docs_{py.parent.parent.name}"
            _run(
                [
                    str(py),
                    "-m",
                    "foresight",
                    "docs",
                    "rnn",
                    "--output-dir",
                    str(docs_out),
                ],
                cwd=root,
                env=env,
            )

        wheels = sorted(dist_dir.glob("*.whl"))
        if not wheels:
            raise RuntimeError(f"No wheels found in {dist_dir}")
        wheel = wheels[0]

        # Create a clean venv and install the wheel.
        _run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv_wheel_dir)],
            cwd=root,
        )
        py_wheel = _venv_python(venv_wheel_dir)
        _run(
            [
                str(py_wheel),
                "-m",
                "pip",
                "install",
                "--progress-bar",
                "off",
                "--no-deps",
                str(wheel),
            ],
            cwd=root,
            env=env,
        )
        _smoke(py_wheel)

        if bool(args.sdist):
            # Install from sdist (sdist/manifest hygiene).
            sdists = sorted(dist_dir.glob("*.tar.gz"))
            if not sdists:
                raise RuntimeError(f"No sdists found in {dist_dir}")
            sdist = sdists[0]

            venv_sdist_dir = tmp_path / "venv_sdist"
            _run(
                [sys.executable, "-m", "venv", "--system-site-packages", str(venv_sdist_dir)],
                cwd=root,
            )
            py_sdist = _venv_python(venv_sdist_dir)
            _run(
                [
                    str(py_sdist),
                    "-m",
                    "pip",
                    "install",
                    "--progress-bar",
                    "off",
                    "--no-deps",
                    str(sdist),
                ],
                cwd=root,
                env=env,
            )
            _smoke(py_sdist)

    print("OK: build + install smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
