from __future__ import annotations

from types import SimpleNamespace

import pytest

import foresight.cli_runtime as runtime_mod


def test_config_from_args_uses_shared_arg_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, object]] = []

    shared = SimpleNamespace(
        _string_arg_value=lambda args, name, default="": calls.append(("str", name, default))
        or f"{name}:{default}",
        _bool_arg_value=lambda args, name, default=False: calls.append(("bool", name, default))
        or True,
    )
    monkeypatch.setattr(runtime_mod, "_cli_shared", shared)

    config = runtime_mod._config_from_args(SimpleNamespace())

    assert config == runtime_mod.CliLogConfig(
        style="log_style:auto",
        level="log_level:info",
        log_file="log_file:",
        no_progress=True,
    )
    assert calls == [
        ("str", "log_style", "auto"),
        ("str", "log_level", "info"),
        ("str", "log_file", ""),
        ("bool", "no_progress", False),
    ]
