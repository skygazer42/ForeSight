from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .base import BaseForecaster
from .transforms import TransformState, fit_transform, inverse_forecast, normalize_transform_list

__all__ = [
    "EnsembleLocalForecaster",
    "PipelineLocalForecaster",
    "make_ensemble_object",
    "make_pipeline_object",
]


def _as_1d_float_array(y: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {arr.shape}")
    return arr


def _clone_local_forecaster(forecaster: BaseForecaster) -> BaseForecaster:
    return copy.deepcopy(forecaster)


def _local_forecaster_from_spec(
    spec: str | BaseForecaster,
    *,
    params: dict[str, Any] | None = None,
) -> BaseForecaster:
    if isinstance(spec, BaseForecaster):
        if params:
            raise ValueError("base params are only supported when the base is a model key")
        return _clone_local_forecaster(spec)

    from .models.registry import make_forecaster_object

    return make_forecaster_object(str(spec).strip(), **dict(params or {}))


def _normalize_member_specs(members: Any) -> tuple[str | BaseForecaster, ...]:
    if members is None:
        raise ValueError("members must be provided")
    if isinstance(members, str):
        string_parts = [part.strip() for part in members.split(",") if part.strip()]
        if not string_parts:
            raise ValueError("members must be non-empty")
        return tuple(string_parts)
    if isinstance(members, list | tuple):
        parts: list[str | BaseForecaster] = [member for member in members if str(member).strip()]
        if not parts:
            raise ValueError("members must be non-empty")
        return tuple(parts)
    if not str(members).strip():
        raise ValueError("members must be non-empty")
    return (members,)


class PipelineLocalForecaster(BaseForecaster):
    def __init__(
        self,
        *,
        base: str | BaseForecaster,
        transforms: Any = (),
        **base_params: Any,
    ) -> None:
        base_key = base.model_key if isinstance(base, BaseForecaster) else str(base).strip()
        if not isinstance(base, BaseForecaster) and base_key == "pipeline":
            raise ValueError("pipeline base model cannot be 'pipeline'")
        if not base_key:
            raise ValueError("base must be non-empty")

        self._base_spec = base
        self._base_params = dict(base_params)
        self._transform_names = normalize_transform_list(transforms)
        self._base_forecaster: BaseForecaster | None = None
        self._transform_states: list[TransformState] = []
        self._train_y: np.ndarray | None = None
        super().__init__(
            model_key="pipeline",
            model_params={
                "base": base_key,
                "transforms": tuple(self._transform_names),
                **dict(base_params),
            },
        )

    def fit(self, y: Any) -> PipelineLocalForecaster:
        train = _as_1d_float_array(y).copy()
        states: list[TransformState] = []
        work = train
        for name in self._transform_names:
            work, state = fit_transform(str(name), work)
            states.append(state)

        base_forecaster = _local_forecaster_from_spec(
            self._base_spec,
            params=self._base_params,
        )
        base_forecaster.fit(work)

        self._train_y = train
        self._transform_states = states
        self._base_forecaster = base_forecaster
        self._is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self._base_forecaster is None:
            raise RuntimeError("fit must be called before predict")
        yhat = np.asarray(self._base_forecaster.predict(int(horizon)), dtype=float)
        for state in reversed(self._transform_states):
            yhat = inverse_forecast(state, yhat)
        return np.asarray(yhat, dtype=float)

    def train_schema_summary(self) -> dict[str, Any]:
        return {
            "kind": "local",
            "n_obs": 0 if self._train_y is None else int(self._train_y.size),
            "composition": {
                "kind": "pipeline",
                "base": str(self.model_params["base"]),
                "transforms": list(self._transform_names),
            },
        }


class EnsembleLocalForecaster(BaseForecaster):
    def __init__(
        self,
        *,
        members: Any,
        agg: str = "mean",
    ) -> None:
        member_specs = _normalize_member_specs(members)
        agg_key = str(agg).strip().lower()
        if agg_key not in {"mean", "median"}:
            raise ValueError("agg must be one of: mean, median")

        member_keys = [
            member.model_key if isinstance(member, BaseForecaster) else str(member).strip()
            for member in member_specs
        ]
        if agg_key == "mean" and any(
            not isinstance(member, BaseForecaster) and str(member).strip() == "ensemble-mean"
            for member in member_specs
        ):
            raise ValueError("ensemble-mean cannot include itself")
        if agg_key == "median" and any(
            not isinstance(member, BaseForecaster) and str(member).strip() == "ensemble-median"
            for member in member_specs
        ):
            raise ValueError("ensemble-median cannot include itself")

        self._member_specs = member_specs
        self._member_forecasters: list[BaseForecaster] = []
        self._agg = agg_key
        self._train_y: np.ndarray | None = None
        super().__init__(
            model_key=f"ensemble-{agg_key}",
            model_params={"members": tuple(member_keys), "agg": agg_key},
        )

    def fit(self, y: Any) -> EnsembleLocalForecaster:
        train = _as_1d_float_array(y).copy()
        member_forecasters = [
            _local_forecaster_from_spec(member).fit(train) for member in self._member_specs
        ]
        self._train_y = train
        self._member_forecasters = member_forecasters
        self._is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or not self._member_forecasters:
            raise RuntimeError("fit must be called before predict")
        preds = [
            np.asarray(member.predict(int(horizon)), dtype=float)
            for member in self._member_forecasters
        ]
        arr = np.stack(preds, axis=0)
        if self._agg == "mean":
            return np.asarray(np.mean(arr, axis=0), dtype=float)
        return np.asarray(np.median(arr, axis=0), dtype=float)

    def train_schema_summary(self) -> dict[str, Any]:
        return {
            "kind": "local",
            "n_obs": 0 if self._train_y is None else int(self._train_y.size),
            "composition": {
                "kind": "ensemble",
                "agg": self._agg,
                "members": list(self.model_params["members"]),
            },
        }


def make_pipeline_object(
    *,
    base: str | BaseForecaster = "naive-last",
    transforms: Any = (),
    **base_params: Any,
) -> PipelineLocalForecaster:
    return PipelineLocalForecaster(base=base, transforms=transforms, **base_params)


def make_ensemble_object(
    *,
    members: Any = ("naive-last", "seasonal-naive", "theta"),
    agg: str = "mean",
) -> EnsembleLocalForecaster:
    return EnsembleLocalForecaster(members=members, agg=agg)
