from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


def _load_workflow(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
    assert "python tools/smoke_build_install.py" in proc.stdout
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
    assert "coverage.xml" in test_step["run"]
    assert "tests/test_fetch_rnn_paper_metadata.py" in test_step["run"]
    assert "tests/test_models_adida.py" in test_step["run"]
    assert "tests/test_models_global_regression_validation.py" in test_step["run"]
    assert "tests/test_models_intermittent.py" in test_step["run"]
    assert "tests/test_models_intermittent_more.py" in test_step["run"]
    assert "tests/test_model_validation_messages.py" in test_step["run"]
    assert "tests/test_models_theta.py" in test_step["run"]
    assert "tests/test_models_theta_auto.py" in test_step["run"]
    assert "tests/test_no_mergeable_nested_ifs.py" in test_step["run"]
    assert "tests/test_no_nested_conditionals.py" in test_step["run"]
    assert "tests/test_no_float_literal_comparisons.py" in test_step["run"]
    assert "tests/test_forecasting_internals.py" in test_step["run"]
    assert "tests/test_sonar_coverage_recent_fixes.py" in test_step["run"]
    assert "tests/test_sonar_torch_rename_coverage_smoke.py" in test_step["run"]
    assert "tests/test_torch_global_validation_messages.py" in test_step["run"]
    assert (
        "tests/test_models_optional_deps_torch.py::test_torch_global_models_smoke_when_installed"
        in test_step["run"]
    )
    assert (
        "tests/test_models_torch_xformer_seq2seq_smoke.py::test_torch_xformer_and_rnn_global_smoke"
        in test_step["run"]
    )
    assert (
        "tests/test_models_torch_crossformer_pyraformer_smoke.py::test_torch_crossformer_and_pyraformer_global_smoke"
        in test_step["run"]
    )
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
    assert (
        scan_step["uses"]
        == "SonarSource/sonarqube-scan-action@a31c9398be7ace6bbfaf30c0bd5d415f843d45e9"
    )
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
    assert "python tools/smoke_build_install.py" in release_doc
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

    assert "python -m foresight doctor" in workflow
    assert "python -m foresight doctor --format text" in workflow
    assert "python -m foresight doctor --require-extra core --strict" in workflow
    assert 'python -c "import foresight; print(foresight.__version__); print(sorted(foresight.__all__)[:3])"' in workflow
    assert "/tmp/foresight_sdist_venv/bin/python -m foresight doctor" in workflow
    assert "/tmp/foresight_sdist_venv/bin/python -m foresight doctor --format text" in workflow
    assert "/tmp/foresight_sdist_venv/bin/python -m foresight doctor --require-extra core --strict" in workflow
    assert '/tmp/foresight_sdist_venv/bin/python -c "import foresight; print(foresight.__version__); print(sorted(foresight.__all__)[:3])"' in workflow


def test_smoke_build_install_script_runs_doctor_and_root_import_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "tools" / "smoke_build_install.py").read_text(encoding="utf-8")

    assert '"import foresight; print(foresight.__version__)"' in script
    assert '"import foresight; print(sorted(foresight.__all__)[:3])"' in script
    assert '"foresight", "doctor"' in script
    assert '"foresight", "doctor", "--format", "text"' in script
    assert '"foresight", "doctor", "--require-extra", "core", "--strict"' in script
    assert 'sys.executable, "-m", "virtualenv"' in script
