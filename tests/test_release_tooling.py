from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
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


def _load_storage_paths_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "storage_paths_for_test",
        repo_root / "tools" / "storage_paths.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_release_check_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "release_check_for_test",
        repo_root / "tools" / "release_check.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_tagged_release_fixture(tmp_path: Path, *, version: str, tag: str) -> Path:
    root = tmp_path / "repo"
    package_dir = root / "src" / "foresight"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")

    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release-test@example.invalid",
            "commit",
            "-m",
            "release fixture",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "tag", tag], cwd=root, check=True, capture_output=True, text=True)
    return root


def test_release_version_check_rejects_head_tag_mismatch(tmp_path: Path) -> None:
    module = _load_release_check_module()
    root = _make_tagged_release_fixture(tmp_path, version="0.2.12", tag="v0.3.0")

    with pytest.raises(
        RuntimeError, match="Package version 0\\.2\\.12 does not match release tag v0\\.3\\.0"
    ):
        module._validate_release_version_tag(root)


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
    assert "python -m pytest -q tests/test_public_contract.py" in proc.stdout
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


def test_release_workflow_grants_oidc_for_trusted_publishing() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = _load_workflow(repo_root / ".github" / "workflows" / "release.yml")

    permissions = workflow["jobs"]["build"].get("permissions") or {}

    assert permissions.get("contents") == "read"
    assert permissions.get("id-token") == "write"


def test_ci_quality_workflow_formats_full_source_directories() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "ruff format --check src tests tools benchmarks" in workflow
    assert "benchmarks/run_benchmarks.py \\" not in workflow


def test_ci_workflow_runs_public_contract_suite() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "name: Contract tests" in workflow
    assert "python -m pytest -q tests/test_public_contract.py" in workflow


def test_ci_workflow_includes_expected_core_jobs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = _load_workflow(repo_root / ".github" / "workflows" / "ci.yml")
    jobs = set((workflow.get("jobs") or {}).keys())

    assert "quality" in jobs
    assert "contract" in jobs
    assert "test-core" in jobs
    assert "test-optional" in jobs
    assert "package" in jobs
    assert "sonar" not in jobs


def test_release_docs_cover_docs_site_and_benchmark_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    release_doc = (repo_root / "docs" / "RELEASE.md").read_text(encoding="utf-8")

    assert "python tools/generate_model_capability_docs.py" in release_doc
    assert "python tools/generate_rnn_docs.py" in release_doc
    assert "python -m pytest -q tests/test_public_contract.py" in release_doc
    assert "python benchmarks/run_benchmarks.py --smoke" in release_doc
    assert "python tools/smoke_build_install.py --sdist" in release_doc
    assert "python tools/smoke_build_install.py --sdist --require-extra sktime" in release_doc
    assert "python tools/smoke_build_install.py --sdist --require-extra darts" in release_doc
    assert "python tools/smoke_build_install.py --sdist --require-extra gluonts" in release_doc
    assert "foresight doctor" in release_doc
    assert "python -m foresight doctor --format text" in release_doc
    assert "doctor --strict" in release_doc
    assert "doctor --require-extra torch --strict" in release_doc
    assert "mkdocs build --strict" in release_doc
    assert ".github/workflows/docs.yml" in release_doc


def test_compatibility_docs_define_public_surface_and_ci_backed_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    compatibility = (repo_root / "docs" / "compatibility.md").read_text(encoding="utf-8")

    assert "## Supported Public Surface" in compatibility
    assert "## CI-Backed Support Matrix" in compatibility
    assert "## Artifact Compatibility Contract" in compatibility
    assert "3.10" in compatibility
    assert "3.11" in compatibility


def test_mypy_targets_include_public_model_support_surfaces() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    assert '"src/foresight/cli.py"' in pyproject
    assert '"src/foresight/cli_catalog.py"' in pyproject
    assert '"src/foresight/adapters/darts.py"' in pyproject
    assert '"src/foresight/adapters/gluonts.py"' in pyproject
    assert '"src/foresight/adapters/sktime.py"' in pyproject
    assert '"src/foresight/pipeline.py"' in pyproject
    assert '"src/foresight/models/specs.py"' in pyproject
    assert '"src/foresight/models/resolution.py"' in pyproject
    assert '"src/foresight/models/registry.py"' in pyproject


def test_dev_dependencies_include_mkdocs_material_for_docs_site() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    assert '"mkdocs-material' in pyproject


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

    assert "--dist-dir" in script
    assert "--require-extra" in script
    assert '"import foresight; print(foresight.__version__)"' in script
    assert '"import foresight; print(sorted(foresight.__all__)[:3])"' in script
    assert '"import foresight.pipeline as pipeline; print(sorted(pipeline.__all__)[:3])"' in script
    assert '"import foresight.adapters as adapters; print(sorted(adapters.__all__)[:3])"' in script
    assert '"foresight", "doctor"' in script
    assert '"foresight", "doctor", "--format", "text"' in script
    assert '"foresight", "doctor", "--require-extra", "core"' in script
    assert '"virtualenv"' in script
    assert 'shutil.which("virtualenv")' in script


def test_smoke_build_install_supports_requested_adapter_extras(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_smoke_build_install_module()
    current_version = _repo_version(repo_root)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    wheel = dist_dir / f"foresight_ts-{current_version}-py3-none-any.whl"
    sdist = dist_dir / f"foresight_ts-{current_version}.tar.gz"
    wheel.write_text("", encoding="utf-8")
    sdist.write_text("", encoding="utf-8")

    calls: list[list[str]] = []

    monkeypatch.setattr(module, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(module, "_create_venv", lambda **kwargs: None)
    monkeypatch.setattr(module, "_venv_python", lambda venv_dir: venv_dir / "bin" / "python")

    def _fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        calls.append(list(cmd))

    monkeypatch.setattr(module, "_run", _fake_run)

    assert (
        module.main(
            [
                "--sdist",
                "--dist-dir",
                str(dist_dir),
                "--require-extra",
                "sktime",
                "--require-extra",
                "gluonts",
            ]
        )
        == 0
    )

    install_specs = [
        cmd[-1] for cmd in calls if len(cmd) >= 4 and cmd[1:4] == ["-m", "pip", "install"]
    ]
    assert any(
        "foresight-ts[sktime,gluonts] @" in spec and wheel.name in spec for spec in install_specs
    )
    assert any(
        "foresight-ts[sktime,gluonts] @" in spec and sdist.name in spec for spec in install_specs
    )
    assert any(cmd[-2:] == ["--require-extra", "sktime"] for cmd in calls)
    assert any(cmd[-2:] == ["--require-extra", "gluonts"] for cmd in calls)
    assert any(cmd[-1] == "import sktime" for cmd in calls if "-c" in cmd)
    assert any(cmd[-1] == "import gluonts" for cmd in calls if "-c" in cmd)


def test_smoke_build_install_runs_darts_and_gluonts_adapter_runtime_smokes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_smoke_build_install_module()
    current_version = _repo_version(repo_root)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    wheel = dist_dir / f"foresight_ts-{current_version}-py3-none-any.whl"
    sdist = dist_dir / f"foresight_ts-{current_version}.tar.gz"
    wheel.write_text("", encoding="utf-8")
    sdist.write_text("", encoding="utf-8")

    calls: list[list[str]] = []

    monkeypatch.setattr(module, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(module, "_create_venv", lambda **kwargs: None)
    monkeypatch.setattr(module, "_venv_python", lambda venv_dir: venv_dir / "bin" / "python")

    def _fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        calls.append(list(cmd))

    monkeypatch.setattr(module, "_run", _fake_run)

    assert (
        module.main(
            [
                "--sdist",
                "--dist-dir",
                str(dist_dir),
                "--require-extra",
                "darts",
                "--require-extra",
                "gluonts",
            ]
        )
        == 0
    )

    base_import_smokes = {
        "import foresight; print(foresight.__version__)",
        "import foresight; print(sorted(foresight.__all__)[:3])",
        "import foresight.pipeline as pipeline; print(sorted(pipeline.__all__)[:3])",
        "import foresight.adapters as adapters; print(sorted(adapters.__all__)[:3])",
        "import darts",
        "import gluonts",
    }
    runtime_smokes = [cmd[-1] for cmd in calls if "-c" in cmd and cmd[-1] not in base_import_smokes]

    assert len(runtime_smokes) == 4


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


def test_smoke_build_install_falls_back_to_virtualenv_executable_when_module_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_smoke_build_install_module()
    venv_dir = tmp_path / "venv"
    calls: list[list[str]] = []

    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/virtualenv")

    def _fake_subprocess_run(cmd, cwd, env, capture_output=False, text=False, check=False):
        if cmd[:3] == [sys.executable, "-m", "venv"]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="ensurepip is not available",
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    def _fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        calls.append(list(cmd))

    monkeypatch.setattr(module.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(module, "_run", _fake_run)

    module._create_venv(venv_dir=venv_dir, cwd=repo_root, env={})

    assert calls == [["/usr/bin/virtualenv", "--system-site-packages", str(venv_dir)]]


def test_storage_paths_prepare_storage_env_sets_large_storage_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_storage_paths_module()
    root = tmp_path / "storage-root"

    monkeypatch.setenv("USER", "alice")
    env: dict[str, str] = {}
    resolved = module.prepare_storage_env(env=env, candidate_roots=[root])

    assert resolved["TMPDIR"] == str(root / "tmp")
    assert resolved["TEMP"] == str(root / "tmp")
    assert resolved["TMP"] == str(root / "tmp")
    assert resolved["PIP_CACHE_DIR"] == str(root / "cache" / "pip")
    assert resolved["UV_CACHE_DIR"] == str(root / "cache" / "uv")
    assert (root / "tmp").is_dir()
    assert (root / "cache" / "pip").is_dir()
    assert (root / "cache" / "uv").is_dir()


def test_storage_paths_prepare_storage_env_preserves_explicit_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_storage_paths_module()
    root = tmp_path / "storage-root"

    monkeypatch.setenv("USER", "alice")
    env = {
        "TMPDIR": "/custom/tmp",
        "PIP_CACHE_DIR": "/custom/pip-cache",
        "UV_CACHE_DIR": "/custom/uv-cache",
    }
    resolved = module.prepare_storage_env(env=env, candidate_roots=[root])

    assert resolved["TMPDIR"] == "/custom/tmp"
    assert resolved["TEMP"] == "/custom/tmp"
    assert resolved["TMP"] == "/custom/tmp"
    assert resolved["PIP_CACHE_DIR"] == "/custom/pip-cache"
    assert resolved["UV_CACHE_DIR"] == "/custom/uv-cache"


def test_release_and_smoke_tooling_use_storage_paths_helper() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    release_script = (repo_root / "tools" / "release_check.py").read_text(encoding="utf-8")
    smoke_script = (repo_root / "tools" / "smoke_build_install.py").read_text(encoding="utf-8")

    assert "prepare_storage_env" in release_script
    assert "prepare_storage_env" in smoke_script


def test_release_check_plan_includes_optional_backend_smoke_installs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "tools/release_check.py", "--plan"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "python tools/smoke_build_install.py --sdist --require-extra sktime" in proc.stdout
    assert "python tools/smoke_build_install.py --sdist --require-extra darts" in proc.stdout
    assert "python tools/smoke_build_install.py --sdist --require-extra gluonts" in proc.stdout


def test_plan_docs_do_not_contain_workspace_absolute_links() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    plan_doc = (
        repo_root / "docs" / "plans" / "2026-03-09-wave1-torch-local-parity-implementation.md"
    ).read_text(encoding="utf-8")

    assert re.search(r"\]\(/(?:data|home|tmp|workspace|workspaces|Users)/", plan_doc) is None
