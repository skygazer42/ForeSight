from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


def _load_workflow(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _repo_version(repo_root: Path) -> str:
    init_py = (repo_root / "src" / "foresight" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"\s*$', init_py, re.MULTILINE)
    assert match is not None
    return match.group(1)


def _load_smoke_build_install_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "smoke_build_install_for_test",
        repo_root / "tools" / "smoke_build_install.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_check_plan_mentions_docs_and_benchmark_steps() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    proc = subprocess.run(
        [sys.executable, "tools/release_check.py", "--plan"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "python tools/check_capability_docs.py" in proc.stdout
    assert "python tools/generate_model_capability_docs.py --check" in proc.stdout
    assert "ruff format --check src tests tools benchmarks" in proc.stdout
    assert "python benchmarks/run_benchmarks.py --smoke" in proc.stdout
    assert "python tools/smoke_build_install.py --sdist" in proc.stdout
    assert "mkdocs build --strict" in proc.stdout


def test_docs_workflow_builds_and_deploys_mkdocs_site() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

    assert "name: Docs" in workflow
    assert "actions/configure-pages" in workflow
    assert "actions/upload-pages-artifact" in workflow
    assert "actions/deploy-pages" in workflow
    assert "python tools/generate_model_capability_docs.py" in workflow
    assert "python tools/generate_rnn_docs.py" in workflow
    assert "mkdocs build --strict" in workflow


def test_docs_workflow_scopes_write_permissions_to_jobs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = _load_workflow(repo_root / ".github" / "workflows" / "docs.yml")

    top_permissions = workflow.get("permissions") or {}
    build_permissions = workflow["jobs"]["build"].get("permissions") or {}
    deploy_permissions = workflow["jobs"]["deploy"].get("permissions") or {}

    assert top_permissions.get("pages") != "write"
    assert top_permissions.get("id-token") != "write"
    assert build_permissions.get("contents") == "read"
    assert build_permissions.get("pages") != "write"
    assert build_permissions.get("id-token") != "write"
    assert deploy_permissions.get("pages") == "write"
    assert deploy_permissions.get("id-token") == "write"


def test_release_workflow_pins_publish_action_to_full_sha() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "pypa/gh-action-pypi-publish@ed0c53931b1dc9bd32cbe73a98c7f6766f8a527e" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" not in workflow


def test_ci_workflow_includes_sonar_analysis_job() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = _load_workflow(repo_root / ".github" / "workflows" / "ci.yml")
    sonar_job = workflow["jobs"]["sonar"]

    steps = sonar_job["steps"]
    checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v4")
    install_step = next(step for step in steps if step.get("name") == "Install")
    test_step = next(step for step in steps if step.get("name") == "Tests")
    scan_step = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("SonarSource/sonarqube-scan-action@")
    )

    assert checkout["with"]["fetch-depth"] == 0
    assert "pip install -e .[dev,torch,stats]" in install_step["run"]
    assert test_step["run"].strip() == "python tools/run_sonar_test_suite.py --coverage-path coverage.xml"
    scan_args = str(scan_step["with"]["args"])
    assert "-Dsonar.sources=src,tools,.github/workflows" in scan_args
    assert "-Dsonar.tests=tests" in scan_args
    assert "-Dsonar.test.inclusions=tests/**/*.py" in scan_args
    assert "-Dsonar.issue.ignore.multicriteria=e1,e2" in scan_args
    assert "-Dsonar.issue.ignore.multicriteria.e2.ruleKey=pythonsecurity:S2083" in scan_args
    assert (
        "-Dsonar.issue.ignore.multicriteria.e2.resourceKey=**/tools/fetch_rnn_paper_metadata.py"
        in scan_args
    )
    assert "src/foresight/models/regression.py" in scan_args
    assert "src/foresight/models/global_regression.py" in scan_args
    assert "src/foresight/models/statsmodels_wrap.py" in scan_args
    assert "src/foresight/models/multivariate.py" in scan_args
    assert "src/foresight/models/torch_ct_rnn.py" in scan_args
    assert "src/foresight/models/torch_global.py" in scan_args
    assert "src/foresight/models/torch_rnn_paper_zoo.py" in scan_args
    assert "src/foresight/models/torch_probabilistic.py" in scan_args
    assert "src/foresight/models/torch_rnn_zoo.py" in scan_args
    assert "src/foresight/models/torch_seq2seq.py" in scan_args
    assert "src/foresight/models/torch_ssm.py" in scan_args
    assert "src/foresight/models/torch_xformer.py" in scan_args
    assert "tests/**" in scan_args
    assert scan_step["uses"] == "SonarSource/sonarqube-scan-action@v7"
    assert scan_step["env"]["SONAR_TOKEN"] == "${{ secrets.SONAR_TOKEN }}"
    assert scan_step["env"]["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"


def test_ci_quality_workflow_formats_full_source_directories() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "ruff format --check src tests tools benchmarks" in workflow
    assert "benchmarks/run_benchmarks.py \\" not in workflow


def test_sonar_project_configuration_targets_maintained_code() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = (repo_root / "sonar-project.properties").read_text(encoding="utf-8")

    assert "sonar.organization=skygazer42" in config
    assert "sonar.projectKey=skygazer42_ForeSight" in config
    assert "sonar.sources=src,tools,.github/workflows" in config
    assert "sonar.tests=tests" in config
    assert "sonar.python.coverage.reportPaths=coverage.xml" in config
    assert "src/foresight/cli_catalog.py" in config
    assert "sonar.cpd.exclusions=" in config
    assert "src/foresight/cli.py" in config
    assert "src/foresight/cli_leaderboard.py" in config
    assert "src/foresight/models/runtime.py" in config
    assert "src/foresight/models/catalog/**" in config
    assert "src/foresight/models/regression.py" in config
    assert "src/foresight/models/global_regression.py" in config
    assert "src/foresight/models/statsmodels_wrap.py" in config
    assert "src/foresight/models/multivariate.py" in config
    assert "src/foresight/models/torch_ct_rnn.py" in config
    assert "src/foresight/models/torch_global.py" in config
    assert "src/foresight/models/torch_rnn_paper_zoo.py" in config
    assert "src/foresight/models/torch_probabilistic.py" in config
    assert "src/foresight/models/torch_rnn_zoo.py" in config
    assert "src/foresight/models/torch_seq2seq.py" in config
    assert "src/foresight/models/torch_ssm.py" in config
    assert "src/foresight/models/torch_xformer.py" in config
    assert "tests/**" in config
    assert "sonar.issue.ignore.multicriteria=e1,e2" in config
    assert "sonar.issue.ignore.multicriteria.e1.ruleKey=pythonsecurity:S2083" in config
    assert "sonar.issue.ignore.multicriteria.e2.ruleKey=pythonsecurity:S2083" in config
    assert (
        "sonar.issue.ignore.multicriteria.e2.resourceKey=**/tools/fetch_rnn_paper_metadata.py"
        in config
    )


def test_automatic_analysis_configuration_scopes_autoscan() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = (repo_root / ".sonarcloud.properties").read_text(encoding="utf-8")

    assert "sonar.organization=skygazer42" in config
    assert "sonar.projectKey=skygazer42_ForeSight" in config
    assert "sonar.sources=src,tools,.github/workflows" in config
    assert "sonar.tests=tests" in config
    assert "src/foresight/cli_catalog.py" in config
    assert "src/foresight/models/runtime.py" in config
    assert "src/foresight/models/regression.py" in config
    assert "src/foresight/models/global_regression.py" in config
    assert "src/foresight/models/torch_global.py" in config
    assert "src/foresight/models/torch_nn.py" in config
    assert "src/foresight/models/torch_seq2seq.py" in config
    assert "src/foresight/models/torch_xformer.py" in config
    assert "sonar.cpd.exclusions=" in config
    assert "src/foresight/models/catalog/ml.py" in config
    assert "src/foresight/models/catalog/torch_global.py" in config
    assert "src/foresight/models/statsmodels_wrap.py" in config


def test_release_docs_cover_docs_site_and_benchmark_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    release_doc = (repo_root / "docs" / "RELEASE.md").read_text(encoding="utf-8")

    assert "python tools/generate_model_capability_docs.py" in release_doc
    assert "python tools/generate_rnn_docs.py" in release_doc
    assert "python benchmarks/run_benchmarks.py --smoke" in release_doc
    assert "python tools/smoke_build_install.py --sdist" in release_doc
    assert "foresight doctor" in release_doc
    assert "python -m foresight doctor --format text" in release_doc
    assert "doctor --strict" in release_doc
    assert "doctor --require-extra torch --strict" in release_doc
    assert "mkdocs build --strict" in release_doc
    assert ".github/workflows/docs.yml" in release_doc


def test_release_docs_warn_about_version_scoped_uploads() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    release_doc = (repo_root / "docs" / "RELEASE.md").read_text(encoding="utf-8")

    assert "dist/foresight_ts-<version>*" in release_doc
    assert "older builds" in release_doc


def test_architecture_import_check_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/check_architecture_imports.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_workflow_smokes_doctor_and_root_import_on_installed_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "python tools/smoke_build_install.py --sdist --dist-dir dist" in workflow
    assert "CLI smoke (installed wheel)" not in workflow
    assert "CLI smoke (installed sdist in clean venv)" not in workflow
    assert "/tmp/foresight_sdist_venv/bin/python -m foresight doctor" not in workflow


def test_smoke_build_install_script_runs_doctor_and_root_import_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "tools" / "smoke_build_install.py").read_text(encoding="utf-8")

    assert '--dist-dir' in script
    assert '"import foresight; print(foresight.__version__)"' in script
    assert '"import foresight; print(sorted(foresight.__all__)[:3])"' in script
    assert '"foresight", "doctor"' in script
    assert '"foresight", "doctor", "--format", "text"' in script
    assert '"foresight", "doctor", "--require-extra", "core"' in script
    assert 'sys.executable, "-m", "virtualenv"' in script


def test_smoke_build_install_prefers_current_version_artifacts_from_dist_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_smoke_build_install_module()
    current_version = _repo_version(repo_root)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    (dist_dir / "foresight_ts-0.0.1-py3-none-any.whl").write_text("", encoding="utf-8")
    (dist_dir / f"foresight_ts-{current_version}-py3-none-any.whl").write_text("", encoding="utf-8")
    (dist_dir / "foresight_ts-0.0.1.tar.gz").write_text("", encoding="utf-8")
    (dist_dir / f"foresight_ts-{current_version}.tar.gz").write_text("", encoding="utf-8")

    installed_artifacts: list[str] = []

    monkeypatch.setattr(module, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(module, "_create_venv", lambda **kwargs: None)
    monkeypatch.setattr(module, "_venv_python", lambda venv_dir: venv_dir / "bin" / "python")

    def _fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        if len(cmd) >= 8 and cmd[1:4] == ["-m", "pip", "install"]:
            installed_artifacts.append(Path(cmd[-1]).name)

    monkeypatch.setattr(module, "_run", _fake_run)

    assert module.main(["--sdist", "--dist-dir", str(dist_dir)]) == 0
    assert installed_artifacts == [
        f"foresight_ts-{current_version}-py3-none-any.whl",
        f"foresight_ts-{current_version}.tar.gz",
    ]
