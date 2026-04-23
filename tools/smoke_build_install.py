#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

_EXTRA_IMPORT_SMOKES = {
    "sktime": "import sktime",
    "darts": "import darts",
    "gluonts": "import gluonts",
}

_EXTRA_RUNTIME_SMOKE_SYMBOLS = {
    "darts": ("to_darts_bundle", "from_darts_bundle"),
    "gluonts": ("to_gluonts_bundle", "from_gluonts_bundle"),
}

_EXTRA_RUNTIME_SMOKES = {
    "darts": textwrap.dedent(
        """
        import pandas as pd
        from foresight.adapters import from_darts_bundle, to_darts_bundle

        long_df = pd.DataFrame(
            {
                "unique_id": ["series_a", "series_a"],
                "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "y": [1.0, 2.0],
            }
        )
        bundle = to_darts_bundle(long_df)
        restored = from_darts_bundle(bundle)
        assert "target" in bundle
        assert "freq" in bundle
        assert list(restored.columns) == ["unique_id", "ds", "y"]
        """
    ).strip(),
    "gluonts": textwrap.dedent(
        """
        import pandas as pd
        from foresight.adapters import from_gluonts_bundle, to_gluonts_bundle

        long_df = pd.DataFrame(
            {
                "unique_id": ["series_a", "series_a"],
                "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "y": [1.0, 2.0],
            }
        )
        bundle = to_gluonts_bundle(long_df)
        restored = from_gluonts_bundle(bundle)
        assert "target" in bundle
        assert "freq" in bundle
        assert list(restored.columns) == ["unique_id", "ds", "y"]
        """
    ).strip(),
}


def _repo_root() -> Path:
    # tools/smoke_build_install.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def _prepare_storage_env(*, env: dict[str, str]) -> dict[str, str]:
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from storage_paths import prepare_storage_env

    return prepare_storage_env(env=env)


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

    virtualenv_cmd: list[str] | None
    if importlib.util.find_spec("virtualenv") is not None:
        virtualenv_cmd = [
            sys.executable,
            "-m",
            "virtualenv",
            "--system-site-packages",
            str(venv_dir),
        ]
    else:
        virtualenv_exe = shutil.which("virtualenv")
        virtualenv_cmd = (
            [str(virtualenv_exe), "--system-site-packages", str(venv_dir)]
            if virtualenv_exe
            else None
        )

    if virtualenv_cmd is None:
        raise RuntimeError(
            "python -m venv failed because ensurepip is unavailable, and `virtualenv` is not installed."
        )

    print("NOTE: stdlib venv missing ensurepip; falling back to virtualenv", flush=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", flush=True)
    _run(virtualenv_cmd, cwd=cwd, env=env)


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


def _normalize_required_extras(raw: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = str(item).strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _artifact_requirement(*, artifact: Path, extras: list[str]) -> str:
    if not extras:
        return str(artifact)
    joined = ",".join(extras)
    return f"foresight-ts[{joined}] @ {artifact.resolve().as_uri()}"


def _install_artifact(
    *, py: Path, artifact: Path, extras: list[str], cwd: Path, env: dict[str, str]
) -> None:
    cmd = [str(py), "-m", "pip", "install", "--progress-bar", "off"]
    cmd.append("--disable-pip-version-check")
    cmd.append(_artifact_requirement(artifact=artifact, extras=extras))
    _run(cmd, cwd=cwd, env=env)


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
    p.add_argument(
        "--require-extra",
        action="append",
        default=[],
        help="Optional extra to install and verify from the built artifact. Repeatable.",
    )
    args = p.parse_args(argv)

    root = _repo_root()
    version = _repo_version(root)
    required_extras = _normalize_required_extras(list(args.require_extra))
    env = _prepare_storage_env(env=dict(os.environ))
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    tmp_root = Path(str(env["TMPDIR"])).expanduser() if env.get("TMPDIR") else None

    with tempfile.TemporaryDirectory(
        prefix="foresight_pkg_smoke_",
        dir=str(tmp_root) if tmp_root is not None else None,
    ) as tmp:
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
            for extra in required_extras:
                import_smoke = _EXTRA_IMPORT_SMOKES.get(str(extra))
                if import_smoke is not None:
                    _run([str(py), "-c", import_smoke], cwd=root, env=env)
                _run(
                    [str(py), "-m", "foresight", "doctor", "--require-extra", str(extra)],
                    cwd=root,
                    env=env,
                )
                runtime_smoke = _EXTRA_RUNTIME_SMOKES.get(str(extra))
                if runtime_smoke is not None:
                    _run([str(py), "-c", runtime_smoke], cwd=root, env=env)
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
        _install_artifact(
            py=py_wheel,
            artifact=wheel,
            extras=required_extras,
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
            _install_artifact(
                py=py_sdist,
                artifact=sdist,
                extras=required_extras,
                cwd=root,
                env=env,
            )
            _smoke(py_sdist)

    print("OK: build + install smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
