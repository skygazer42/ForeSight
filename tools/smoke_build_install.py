#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import os
import re
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


def _create_venv(*, venv_dir: Path, cwd: Path, env: dict[str, str]) -> None:
    cmd = [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)]
    print(f"+ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if proc.returncode == 0:
        return

    combined = ((proc.stdout or "") + (proc.stderr or "")).lower()
    if "ensurepip" not in combined:
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd,
            output=proc.stdout,
            stderr=proc.stderr,
        )

    if importlib.util.find_spec("virtualenv") is None:
        raise RuntimeError(
            "python -m venv failed because ensurepip is unavailable, and `virtualenv` is not installed."
        )

    print("NOTE: stdlib venv missing ensurepip; falling back to virtualenv", flush=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", flush=True)
    _run(
        [sys.executable, "-m", "virtualenv", "--system-site-packages", str(venv_dir)],
        cwd=cwd,
        env=env,
    )


def _repo_version(root: Path) -> str:
    init_py = (root / "src" / "foresight" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"\s*$', init_py, re.MULTILINE)
    if match is None:
        raise RuntimeError(
            f"Could not determine package version from {root / 'src' / 'foresight' / '__init__.py'}"
        )
    return match.group(1)


def _select_artifact(*, dist_dir: Path, glob_pattern: str, version: str, label: str) -> Path:
    artifacts = sorted(dist_dir.glob(glob_pattern))
    if not artifacts:
        raise RuntimeError(f"No {label}s found in {dist_dir}")

    prefix = f"foresight_ts-{version}"
    versioned = [
        artifact
        for artifact in artifacts
        if artifact.name == f"{prefix}.tar.gz" or artifact.name.startswith(f"{prefix}-")
    ]
    if versioned:
        return versioned[0]

    available = ", ".join(artifact.name for artifact in artifacts)
    raise RuntimeError(
        f"No {label} for version {version} found in {dist_dir}; available: {available}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="smoke_build_install",
        description="Build wheel/sdist and run a minimal install+CLI smoke.",
    )
    p.add_argument(
        "--dist-dir",
        type=str,
        default="",
        help="Optional existing dist directory containing built wheel/sdist artifacts.",
    )
    p.add_argument(
        "--sdist",
        action="store_true",
        help="Also install from sdist (slower; may require network for build isolation).",
    )
    args = p.parse_args(argv)

    root = _repo_root()
    version = _repo_version(root)

    with tempfile.TemporaryDirectory(prefix="foresight_pkg_smoke_") as tmp:
        tmp_path = Path(tmp)
        dist_dir_arg = str(args.dist_dir).strip()
        dist_dir = (
            Path(dist_dir_arg).expanduser().resolve() if dist_dir_arg else (tmp_path / "dist")
        )
        venv_wheel_dir = tmp_path / "venv_wheel"

        if not dist_dir_arg:
            # Build wheel+sdist in an isolated env.
            _run([sys.executable, "-m", "build", "--outdir", str(dist_dir)], cwd=root)
        elif not dist_dir.exists():
            raise RuntimeError(f"--dist-dir does not exist: {dist_dir}")

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
            _run(
                [str(py), "-c", "import foresight; print(sorted(foresight.__all__)[:3])"],
                cwd=root,
                env=env,
            )
            _run(
                [
                    str(py),
                    "-c",
                    "import foresight.pipeline as pipeline; print(sorted(pipeline.__all__)[:3])",
                ],
                cwd=root,
                env=env,
            )
            _run(
                [
                    str(py),
                    "-c",
                    "import foresight.adapters as adapters; print(sorted(adapters.__all__)[:3])",
                ],
                cwd=root,
                env=env,
            )

            # CLI smoke (module invocation avoids PATH concerns).
            _run([str(py), "-m", "foresight", "--version"], cwd=root, env=env)
            _run([str(py), "-m", "foresight", "doctor"], cwd=root, env=env)
            _run([str(py), "-m", "foresight", "doctor", "--format", "text"], cwd=root, env=env)
            _run(
                [str(py), "-m", "foresight", "doctor", "--require-extra", "core"],
                cwd=root,
                env=env,
            )
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

        wheel = _select_artifact(
            dist_dir=dist_dir, glob_pattern="*.whl", version=version, label="wheel"
        )

        # Create a clean venv and install the wheel.
        _create_venv(venv_dir=venv_wheel_dir, cwd=root, env=env)
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
            sdist = _select_artifact(
                dist_dir=dist_dir,
                glob_pattern="*.tar.gz",
                version=version,
                label="sdist",
            )

            venv_sdist_dir = tmp_path / "venv_sdist"
            _create_venv(venv_dir=venv_sdist_dir, cwd=root, env=env)
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
