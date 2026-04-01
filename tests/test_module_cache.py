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


def test_get_cached_module_preserves_existing_cached_object() -> None:
    sentinel = object()
    namespace: dict[str, Any] = {"_batch_execution": sentinel}

    loaded = get_cached_module(
        namespace,
        "_batch_execution",
        ".batch_execution",
        "foresight",
    )

    assert loaded is sentinel


def test_get_cached_module_supports_package_relative_imports() -> None:
    namespace: dict[str, Any] = {}

    loaded = get_cached_module(
        namespace,
        "_batch_execution",
        ".batch_execution",
        "foresight",
    )

    assert loaded.__name__ == "foresight.batch_execution"
