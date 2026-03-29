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

    with pytest.raises(
        ImportError,
        match='Torch models require PyTorch\\. Install with: pip install "foresight-ts\\[torch\\]" or pip install -e "\\.\\[torch\\]"',
    ):
        torch_nn._require_torch()


def test_dependency_status_exposes_install_commands() -> None:
    status = optional_deps.get_dependency_status("ml").as_dict()

    assert status["recommended_extra"] == "ml"
    assert status["package_install_command"] == 'pip install "foresight-ts[ml]"'
    assert status["editable_install_command"] == 'pip install -e ".[ml]"'


def test_extra_status_exposes_install_commands() -> None:
    status = optional_deps.get_extra_status("torch").as_dict()

    assert status["package_install_command"] == 'pip install "foresight-ts[torch]"'
    assert status["editable_install_command"] == 'pip install -e ".[torch]"'


def test_missing_dependency_message_includes_package_and_editable_commands() -> None:
    msg = optional_deps.missing_dependency_message("ml", subject="ridge_lag_forecast")

    assert msg == (
        'ridge_lag_forecast requires scikit-learn. '
        'Install with: pip install "foresight-ts[ml]" or pip install -e ".[ml]"'
    )
