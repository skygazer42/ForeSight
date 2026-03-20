from __future__ import annotations

import functools
from contextlib import nullcontext
from typing import Any

from ..base import (
    BaseForecaster,
    BaseGlobalForecaster,
    RegistryForecaster,
    RegistryGlobalForecaster,
)
from .specs import ForecasterFn, GlobalForecasterFn, ModelSpec, MultivariateForecasterFn

_DEFERRED_TORCH_RUNTIME_PARAM_KEYS = frozenset(
    {
        "tensorboard_log_dir",
        "tensorboard_run_name",
        "tensorboard_flush_secs",
        "mlflow_tracking_uri",
        "mlflow_experiment_name",
        "mlflow_run_name",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_dir",
        "wandb_mode",
    }
)


def _merged_params(spec: ModelSpec, params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(spec.default_params)
    merged.update(params)
    return merged


def _require_interface(*, key: str, spec: ModelSpec, expected: str, alt: str) -> None:
    if str(spec.interface).lower().strip() != expected:
        raise ValueError(
            f"Model {key!r} uses interface={spec.interface!r} (not {expected!r}). {alt}"
        )


def _split_deferred_torch_runtime_params(
    *,
    key: str,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not str(key).startswith("torch-"):
        return dict(params), {}

    deferred = {
        str(name): value
        for name, value in dict(params).items()
        if str(name) in _DEFERRED_TORCH_RUNTIME_PARAM_KEYS
    }
    direct = {
        str(name): value
        for name, value in dict(params).items()
        if str(name) not in _DEFERRED_TORCH_RUNTIME_PARAM_KEYS
    }
    return direct, deferred


def _wrap_deferred_torch_runtime_callable(
    func: Any,
    *,
    deferred_params: dict[str, Any],
) -> Any:
    if not deferred_params or not callable(func):
        return func

    from .runtime import torch_train_config_override

    @functools.wraps(func)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        ctx = torch_train_config_override(deferred_params) if deferred_params else nullcontext()
        with ctx:
            return func(*args, **kwargs)

    return _wrapped


def build_local_forecaster(*, key: str, spec: ModelSpec, params: dict[str, Any]) -> ForecasterFn:
    _require_interface(
        key=key,
        spec=spec,
        expected="local",
        alt="Use `make_global_forecaster()` instead.",
    )
    direct_params, deferred_params = _split_deferred_torch_runtime_params(
        key=key,
        params=_merged_params(spec, params),
    )
    out = spec.factory(**direct_params)
    return _wrap_deferred_torch_runtime_callable(out, deferred_params=deferred_params)


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
    direct_params, deferred_params = _split_deferred_torch_runtime_params(
        key=key,
        params=_merged_params(spec, params),
    )
    out = spec.factory(**direct_params)
    if not callable(out):
        raise TypeError(f"Global model factory must return a callable, got: {type(out).__name__}")
    return _wrap_deferred_torch_runtime_callable(out, deferred_params=deferred_params)


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
    direct_params, deferred_params = _split_deferred_torch_runtime_params(
        key=key,
        params=_merged_params(spec, params),
    )
    out = spec.factory(**direct_params)
    if not callable(out):
        raise TypeError(
            f"Multivariate model factory must return a callable, got: {type(out).__name__}"
        )
    return _wrap_deferred_torch_runtime_callable(out, deferred_params=deferred_params)


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
    direct_params, deferred_params = _split_deferred_torch_runtime_params(
        key=key,
        params=merged,
    )
    return RegistryForecaster(
        model_key=str(key),
        model_params=merged,
        factory=lambda: _wrap_deferred_torch_runtime_callable(
            spec.factory(**dict(direct_params)),
            deferred_params=deferred_params,
        ),
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
    direct_params, deferred_params = _split_deferred_torch_runtime_params(
        key=key,
        params=merged,
    )
    return RegistryGlobalForecaster(
        model_key=str(key),
        model_params=merged,
        factory=lambda: _wrap_deferred_torch_runtime_callable(
            spec.factory(**dict(direct_params)),
            deferred_params=deferred_params,
        ),
    )


def rebuild_local_forecaster_runtime(model_key: str, model_params: dict[str, Any]) -> ForecasterFn:
    from .resolution import get_model_spec

    spec = get_model_spec(model_key)
    return build_local_forecaster(key=model_key, spec=spec, params=dict(model_params))


def rebuild_global_forecaster_runtime(
    model_key: str,
    model_params: dict[str, Any],
) -> GlobalForecasterFn:
    from .resolution import get_model_spec

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
