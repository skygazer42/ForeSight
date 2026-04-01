from __future__ import annotations

from importlib import import_module
from typing import Any


def get_cached_module(
    namespace: dict[str, Any],
    cache_name: str,
    module_name: str,
    package: str,
) -> Any:
    cached = namespace.get(str(cache_name))
    if cached is not None:
        return cached

    module = import_module(str(module_name), str(package))
    namespace[str(cache_name)] = module
    return module


__all__ = ["get_cached_module"]
