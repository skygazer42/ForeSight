from __future__ import annotations

from typing import Any

from ..base import (
    BaseForecaster,
    BaseGlobalForecaster,
    RegistryForecaster,
    RegistryGlobalForecaster,
)
from .specs import ForecasterFn, GlobalForecasterFn, ModelSpec, MultivariateForecasterFn


def _merged_params(spec: ModelSpec, params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(spec.default_params)
    merged.update(params)
    return merged


def _require_interface(*, key: str, spec: ModelSpec, expected: str, alt: str) -> None:
    if str(spec.interface).lower().strip() != expected:
        raise ValueError(
            f"Model {key!r} uses interface={spec.interface!r} (not {expected!r}). {alt}"
        )


def build_local_forecaster(*, key: str, spec: ModelSpec, params: dict[str, Any]) -> ForecasterFn:
    _require_interface(
        key=key,
        spec=spec,
        expected="local",
        alt="Use `make_global_forecaster()` instead.",
    )
    return spec.factory(**_merged_params(spec, params))


def build_global_forecaster(
    *,
    key: str,
    spec: ModelSpec,
    params: dict[str, Any],
) -> GlobalForecasterFn:
    _require_interface(
        key=key,
        spec=spec,
        expected="global",
        alt="Use `make_forecaster()` instead.",
    )
    out = spec.factory(**_merged_params(spec, params))
    if not callable(out):
        raise TypeError(f"Global model factory must return a callable, got: {type(out).__name__}")
    return out


def build_multivariate_forecaster(
    *,
    key: str,
    spec: ModelSpec,
    params: dict[str, Any],
) -> MultivariateForecasterFn:
    _require_interface(
        key=key,
        spec=spec,
        expected="multivariate",
        alt="Use `make_forecaster()` or `make_global_forecaster()` instead.",
    )
    out = spec.factory(**_merged_params(spec, params))
    if not callable(out):
        raise TypeError(
            f"Multivariate model factory must return a callable, got: {type(out).__name__}"
        )
    return out


def build_local_forecaster_object(
    *,
    key: str,
    spec: ModelSpec,
    params: dict[str, Any],
) -> BaseForecaster:
    _require_interface(
        key=key,
        spec=spec,
        expected="local",
        alt="Use `make_global_forecaster_object()` instead.",
    )
    merged = _merged_params(spec, params)
    return RegistryForecaster(
        model_key=str(key),
        model_params=merged,
        factory=lambda: spec.factory(**dict(merged)),
    )


def build_global_forecaster_object(
    *,
    key: str,
    spec: ModelSpec,
    params: dict[str, Any],
) -> BaseGlobalForecaster:
    _require_interface(
        key=key,
        spec=spec,
        expected="global",
        alt="Use `make_forecaster_object()` instead.",
    )
    merged = _merged_params(spec, params)
    return RegistryGlobalForecaster(
        model_key=str(key),
        model_params=merged,
        factory=lambda: spec.factory(**dict(merged)),
    )


def rebuild_local_forecaster_runtime(model_key: str, model_params: dict[str, Any]) -> ForecasterFn:
    from .registry import get_model_spec

    spec = get_model_spec(model_key)
    return build_local_forecaster(key=model_key, spec=spec, params=dict(model_params))


def rebuild_global_forecaster_runtime(
    model_key: str,
    model_params: dict[str, Any],
) -> GlobalForecasterFn:
    from .registry import get_model_spec

    spec = get_model_spec(model_key)
    return build_global_forecaster(key=model_key, spec=spec, params=dict(model_params))


__all__ = [
    "build_global_forecaster",
    "build_global_forecaster_object",
    "build_local_forecaster",
    "build_local_forecaster_object",
    "build_multivariate_forecaster",
    "rebuild_global_forecaster_runtime",
    "rebuild_local_forecaster_runtime",
]
