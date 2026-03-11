from __future__ import annotations

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
