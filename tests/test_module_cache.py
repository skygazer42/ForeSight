from __future__ import annotations

from typing import Any

from foresight.module_cache import get_cached_module


def test_get_cached_module_imports_once_and_reuses_namespace_slot() -> None:
    namespace: dict[str, Any] = {}

    first = get_cached_module(
        namespace,
        "_cli_shared",
        ".cli_shared",
        "foresight",
    )
    second = get_cached_module(
        namespace,
        "_cli_shared",
        ".cli_shared",
        "foresight",
    )

    assert first is second
    assert namespace["_cli_shared"] is first
