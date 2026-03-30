from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

_SONAR_SCAN_ACTION_SHA = "a31c9398be7ace6bbfaf30c0bd5d415f843d45e9"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_common_dir(repo_root: Path) -> Path:
    git_path = repo_root / ".git"
    if git_path.is_dir():
        return git_path
    text = git_path.read_text(encoding="utf-8").strip()
    _, gitdir = text.split(":", 1)
    return Path(gitdir.strip()).parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ci_sonar_job_uses_shared_sonar_test_suite_and_current_scan_action() -> None:
    workflow = yaml.safe_load((_repo_root() / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    sonar_job = workflow["jobs"]["sonar"]
    steps = sonar_job["steps"]

    test_step = next(step for step in steps if step.get("name") == "Tests")
    scan_step = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("SonarSource/sonarqube-scan-action@")
    )

    assert test_step["run"].strip() == "python tools/run_sonar_test_suite.py --coverage-path coverage.xml"
    assert scan_step["uses"] == f"SonarSource/sonarqube-scan-action@{_SONAR_SCAN_ACTION_SHA}"


def test_sonar_test_dockerfile_installs_required_runtime() -> None:
    dockerfile = (_repo_root() / "docker" / "sonar-test.Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.10" in dockerfile
    assert "openjdk-17-jre-headless" in dockerfile
    assert "git" in dockerfile
    assert "curl" in dockerfile
    assert "unzip" in dockerfile
    assert 'pip install -e ".[dev,torch,stats]"' in dockerfile


def test_run_sonar_test_suite_module_builds_pytest_command() -> None:
    module = _load_module(_repo_root() / "tools" / "run_sonar_test_suite.py", "run_sonar_test_suite")

    cmd = module._sonar_pytest_command("coverage.xml")  # type: ignore[attr-defined]

    assert cmd[:3] == ["python", "-m", "pytest"]
    assert "--cov=foresight" in cmd
    assert "--cov-report=xml:coverage.xml" in cmd
    assert "tests/test_fetch_rnn_paper_metadata.py" in cmd
    assert "tests/test_models_global_regression_validation.py" in cmd
    assert "tests/test_torch_global_validation_messages.py" in cmd


def test_run_sonar_local_module_builds_docker_commands() -> None:
    module = _load_module(_repo_root() / "tools" / "run_sonar_local.py", "run_sonar_local")
    repo_root = _repo_root()
    git_common_dir = _git_common_dir(repo_root)

    build_cmd = module._docker_build_command(  # type: ignore[attr-defined]
        dockerfile=Path("docker/sonar-test.Dockerfile"),
        image_tag="foresight-sonar-test:latest",
    )
    test_cmd = module._docker_test_command(  # type: ignore[attr-defined]
        image_tag="foresight-sonar-test:latest",
        coverage_path=Path(".artifacts/sonar-local/coverage.xml"),
    )
    scan_cmd = module._docker_scan_command(  # type: ignore[attr-defined]
        token_env="SONAR_TOKEN",
        coverage_path=Path(".artifacts/sonar-local/coverage.xml"),
        branch="feat/sonar-local-ci",
        pull_request="",
    )

    assert build_cmd[:3] == ["docker", "build", "-f"]
    assert "docker/sonar-test.Dockerfile" in build_cmd
    assert "foresight-sonar-test:latest" in build_cmd

    assert test_cmd[0:3] == ["docker", "run", "--rm"]
    assert "foresight-sonar-test:latest" in test_cmd
    assert "tools/run_sonar_test_suite.py" in " ".join(test_cmd)
    assert ".artifacts/sonar-local/coverage.xml" in " ".join(test_cmd)

    joined_scan = " ".join(scan_cmd)
    assert scan_cmd[0:3] == ["docker", "run", "--rm"]
    assert "sonarsource/sonar-scanner-cli" in joined_scan
    assert "-Dsonar.host.url=https://sonarcloud.io" in joined_scan
    assert "-Dsonar.branch.name=feat/sonar-local-ci" in joined_scan
    assert ".artifacts/sonar-local/coverage.xml" in joined_scan
    assert f"{repo_root}:{repo_root}" in scan_cmd
    if git_common_dir != repo_root / ".git":
        assert f"{git_common_dir}:{git_common_dir}:ro" in scan_cmd
    assert scan_cmd[scan_cmd.index("-w") + 1] == str(repo_root)


def test_fetch_sonar_issues_module_builds_search_url() -> None:
    module = _load_module(_repo_root() / "tools" / "fetch_sonar_issues.py", "fetch_sonar_issues")

    url = module._issues_search_url(  # type: ignore[attr-defined]
        component_key="skygazer42_ForeSight",
        branch="main",
        pull_request="",
        only_open=True,
        page=1,
        page_size=100,
    )

    assert url.startswith("https://sonarcloud.io/api/issues/search?")
    assert "componentKeys=skygazer42_ForeSight" in url
    assert "branch=main" in url
    assert "resolved=false" in url
    assert "p=1" in url
    assert "ps=100" in url
