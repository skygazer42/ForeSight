from __future__ import annotations

from . import runtime as _runtime
from .catalog import build_catalog
from .specs import ModelSpec

_REGISTRY: dict[str, ModelSpec] = build_catalog(_runtime)


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_model_spec(key: str) -> ModelSpec:
    try:
        return _REGISTRY[key]
    except KeyError as e:
        raise KeyError(f"Unknown model key: {key!r}. Try one of: {', '.join(list_models())}") from e


__all__ = ["get_model_spec", "list_models"]
