#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

QUALITY_FORMAT_FILES = [
    "benchmarks/run_benchmarks.py",
    "src/foresight/__init__.py",
    "src/foresight/base.py",
    "src/foresight/datasets/__init__.py",
    "src/foresight/datasets/registry.py",
    "src/foresight/serialization.py",
    "tests/test_packaged_datasets_smoke.py",
    "tests/test_release_tooling.py",
    "tests/test_root_import.py",
    "tests/test_models_registry.py",
    "tools/check_capability_docs.py",
    "tools/generate_model_capability_docs.py",
    "tools/release_check.py",
]


def _repo_root() -> Path:
    # tools/release_check.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _quality_commands() -> list[list[str]]:
    return [
        [sys.executable, "tools/check_capability_docs.py"],
        [sys.executable, "tools/generate_model_capability_docs.py", "--check"],
        ["ruff", "check", "src", "tests", "tools", "benchmarks"],
        ["ruff", "format", "--check", *QUALITY_FORMAT_FILES],
        [sys.executable, "-m", "mypy", "--no-incremental", "--cache-dir=/dev/null"],
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "benchmarks/run_benchmarks.py", "--smoke"],
        ["mkdocs", "build", "--strict"],
    ]


def _display_command(cmd: list[str]) -> list[str]:
    if cmd and Path(cmd[0]).name.lower() in {"python", "python.exe", "py", "py.exe"}:
        return ["python", *cmd[1:]]
    return list(cmd)


def _print_plan(cmds: list[list[str]]) -> None:
    for cmd in cmds:
        print(" ".join(_display_command(cmd)), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run release-quality checks and build package artifacts in a clean temp dir."
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the release-check command plan without executing it.",
    )
    args = parser.parse_args(argv)

    root = _repo_root()

    env = dict(os.environ)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    quality_cmds = _quality_commands()
    if args.plan:
        _print_plan(quality_cmds)
        print(
            " ".join(
                _display_command([sys.executable, "-m", "build", "--outdir", "<temp-dist-dir>"])
            ),
            flush=True,
        )
        if importlib.util.find_spec("twine") is not None:
            print(
                " ".join(
                    _display_command([sys.executable, "-m", "twine", "check", "<temp-dist-dir>/*"])
                ),
                flush=True,
            )
        return 0

    for cmd in quality_cmds:
        _run(cmd, cwd=root, env=env)

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
