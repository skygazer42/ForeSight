from __future__ import annotations

import types

import pytest

import foresight.optional_deps as optional_deps
from foresight.models import torch_nn


def test_torch_namespace_stub_is_reported_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_stub = types.ModuleType("torch")

    monkeypatch.setattr(optional_deps, "_find_spec", lambda name: object() if name == "torch" else None)
    monkeypatch.setattr(
        optional_deps,
        "_import_module",
        lambda name: torch_stub if name == "torch" else pytest.fail(f"unexpected import: {name}"),
    )

    status = optional_deps.get_dependency_status("torch")

    assert status.name == "torch"
    assert status.spec_found is True
    assert status.available is False
    assert status.version is None
    assert "missing required attributes" in str(status.reason)


def test_require_torch_rejects_namespace_stub_without_nn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_stub = types.ModuleType("torch")

    monkeypatch.setattr(optional_deps, "_find_spec", lambda name: object() if name == "torch" else None)
    monkeypatch.setattr(
        optional_deps,
        "_import_module",
        lambda name: torch_stub if name == "torch" else pytest.fail(f"unexpected import: {name}"),
    )

    with pytest.raises(ImportError, match="Install with: pip install -e"):
        torch_nn._require_torch()
