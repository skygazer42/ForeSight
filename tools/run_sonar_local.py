#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _docker_build_command(*, dockerfile: Path, image_tag: str) -> list[str]:
    return [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "-t",
        str(image_tag),
        ".",
    ]


def _docker_test_command(*, image_tag: str, coverage_path: Path) -> list[str]:
    repo_root = _repo_root()
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{repo_root}:/workspace",
        "-w",
        "/workspace",
        str(image_tag),
        "python",
        "tools/run_sonar_test_suite.py",
        "--coverage-path",
        str(coverage_path),
    ]


def _docker_scan_command(
    *,
    token_env: str,
    coverage_path: Path,
    branch: str,
    pull_request: str,
) -> list[str]:
    repo_root = _repo_root()
    cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        str(token_env),
        "-v",
        f"{repo_root}:/usr/src",
        "-w",
        "/usr/src",
        "sonarsource/sonar-scanner-cli:latest",
        "sonar-scanner",
        "-Dsonar.host.url=https://sonarcloud.io",
        f"-Dsonar.python.coverage.reportPaths={coverage_path}",
    ]
    branch_s = str(branch).strip()
    pr_s = str(pull_request).strip()
    if branch_s:
        cmd.append(f"-Dsonar.branch.name={branch_s}")
    if pr_s:
        cmd.append(f"-Dsonar.pullrequest.key={pr_s}")
    return cmd


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a local Sonar test image, run the Sonar pytest suite, then invoke sonar-scanner in Docker."
    )
    parser.add_argument(
        "--dockerfile",
        default="docker/sonar-test.Dockerfile",
        help="Dockerfile path for the local Sonar test image.",
    )
    parser.add_argument(
        "--image-tag",
        default="foresight-sonar-test:latest",
        help="Docker image tag for the local Sonar test image.",
    )
    parser.add_argument(
        "--coverage-path",
        default=".artifacts/sonar-local/coverage.xml",
        help="Coverage XML path relative to the repo root.",
    )
    parser.add_argument(
        "--branch",
        default="",
        help="Optional Sonar branch name for local scanning.",
    )
    parser.add_argument(
        "--pull-request",
        default="",
        help="Optional Sonar pull request key for local scanning.",
    )
    parser.add_argument(
        "--token-env",
        default="SONAR_TOKEN",
        help="Environment variable containing the Sonar token.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip docker build if the image already exists.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running the Sonar pytest suite in Docker.",
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="Skip invoking sonar-scanner.",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    dockerfile = Path(str(args.dockerfile)).expanduser()
    if not dockerfile.is_absolute():
        dockerfile = (repo_root / dockerfile).resolve()
    coverage_path = Path(str(args.coverage_path))
    coverage_abs = (repo_root / coverage_path).resolve()
    coverage_abs.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)

    if not args.skip_build:
        _run(
            _docker_build_command(dockerfile=dockerfile, image_tag=str(args.image_tag)),
            cwd=repo_root,
            env=env,
        )

    if not args.skip_tests:
        _run(
            _docker_test_command(
                image_tag=str(args.image_tag),
                coverage_path=coverage_path,
            ),
            cwd=repo_root,
            env=env,
        )

    if not args.skip_scan:
        token_env = str(args.token_env).strip() or "SONAR_TOKEN"
        if not env.get(token_env):
            raise RuntimeError(f"Missing required environment variable: {token_env}")
        _run(
            _docker_scan_command(
                token_env=token_env,
                coverage_path=coverage_path,
                branch=str(args.branch),
                pull_request=str(args.pull_request),
            ),
            cwd=repo_root,
            env=env,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
