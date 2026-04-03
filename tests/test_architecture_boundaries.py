from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _load_checker_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "tools" / "check_architecture_imports.py"
    spec = importlib.util.spec_from_file_location("check_architecture_imports", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_exports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                return list(ast.literal_eval(node.value))
    raise AssertionError(f"__all__ not found in {path}")


def test_architecture_import_check_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/check_architecture_imports.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_architecture_doc_describes_layers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    doc = (repo_root / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "contracts" in doc
    assert "services" in doc
    assert "models/catalog" in doc or "catalog" in doc
    assert "facade" in doc


def test_public_facades_do_not_export_private_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    forecast_exports = _module_exports(repo_root / "src" / "foresight" / "forecast.py")
    eval_exports = _module_exports(repo_root / "src" / "foresight" / "eval_forecast.py")

    assert all(not name.startswith("_") for name in forecast_exports)
    assert all(not name.startswith("_") for name in eval_exports)


def test_architecture_checker_flags_base_registry_relative_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = _load_checker_module()

    repo_root = tmp_path
    src_root = repo_root / "src" / "foresight"
    src_root.mkdir(parents=True)
    (src_root / "base.py").write_text(
        "from .models import registry as registry_mod\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(checker, "REPO_ROOT", repo_root)
    monkeypatch.setattr(checker, "SRC_ROOT", src_root)

    violations: list[str] = []
    checker._base_must_not_import_registry(violations)

    assert violations
    assert "models.registry" in violations[0]


def test_architecture_checker_flags_contracts_importing_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = _load_checker_module()

    repo_root = tmp_path
    src_root = repo_root / "src" / "foresight"
    contracts_root = src_root / "contracts"
    contracts_root.mkdir(parents=True)
    (contracts_root / "frames.py").write_text(
        "from .. import services\n",
        encoding="utf-8",
    )
    (src_root / "base.py").write_text("", encoding="utf-8")
    (src_root / "cli.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(checker, "REPO_ROOT", repo_root)
    monkeypatch.setattr(checker, "SRC_ROOT", src_root)

    violations: list[str] = []
    checker._contracts_must_not_import_services(violations)

    assert violations
    assert "contracts must not import services" in violations[0]


def test_architecture_checker_flags_services_importing_public_facades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = _load_checker_module()

    repo_root = tmp_path
    src_root = repo_root / "src" / "foresight"
    services_root = src_root / "services"
    services_root.mkdir(parents=True)
    (services_root / "forecasting.py").write_text(
        "from ..forecast import forecast_model\n",
        encoding="utf-8",
    )
    (src_root / "base.py").write_text("", encoding="utf-8")
    (src_root / "cli.py").write_text("", encoding="utf-8")
    (src_root / "forecast.py").write_text("", encoding="utf-8")
    (src_root / "eval_forecast.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(checker, "REPO_ROOT", repo_root)
    monkeypatch.setattr(checker, "SRC_ROOT", src_root)

    violations: list[str] = []
    checker._services_must_not_import_public_facades(violations)

    assert violations
    assert "services must not import public facades" in violations[0]


def test_architecture_checker_flags_private_tuple_exports_in_public_facades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = _load_checker_module()

    repo_root = tmp_path
    src_root = repo_root / "src" / "foresight"
    src_root.mkdir(parents=True)
    (src_root / "forecast.py").write_text(
        '__all__ = ("_require_long_df", "forecast_model")\n',
        encoding="utf-8",
    )
    (src_root / "eval_forecast.py").write_text(
        '__all__ = ("eval_model",)\n',
        encoding="utf-8",
    )
    (src_root / "base.py").write_text("", encoding="utf-8")
    (src_root / "cli.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(checker, "REPO_ROOT", repo_root)
    monkeypatch.setattr(checker, "SRC_ROOT", src_root)

    violations: list[str] = []
    checker._facades_must_not_export_private_helpers(violations)

    assert violations
    assert "public facades must not export private helper" in violations[0]


def test_architecture_checker_flags_registry_resolution_runtime_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = _load_checker_module()

    repo_root = tmp_path
    src_root = repo_root / "src" / "foresight"
    models_root = src_root / "models"
    models_root.mkdir(parents=True)
    (models_root / "registry.py").write_text(
        "def _resolve_catalog_key():\n"
        "    return None\n\n"
        "def _build_runtime_bridge():\n"
        "    return None\n\n"
        "def keep_public_surface():\n"
        "    return None\n",
        encoding="utf-8",
    )
    (models_root / "resolution.py").write_text(
        "def get_model_spec():\n    return None\n", encoding="utf-8"
    )
    (models_root / "runtime.py").write_text(
        "def make_forecaster():\n    return None\n", encoding="utf-8"
    )
    (src_root / "base.py").write_text("", encoding="utf-8")
    (src_root / "cli.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(checker, "REPO_ROOT", repo_root)
    monkeypatch.setattr(checker, "SRC_ROOT", src_root)

    violations: list[str] = []
    checker._registry_must_stay_facade(violations)

    assert violations
    assert "models.registry should stay a facade" in violations[0]
    assert any("_build_runtime_bridge" in item for item in violations)
