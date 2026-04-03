from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from ..base import BaseForecaster
from ..optional_deps import missing_dependency_message

__all__ = [
    "SktimeForecasterAdapter",
    "make_sktime_forecaster_adapter",
]


def _require_sktime() -> Any:
    try:
        import sktime
    except Exception as e:  # noqa: BLE001
        raise ImportError(missing_dependency_message("sktime", subject="sktime adapter")) from e
    return sktime


def _coerce_sktime_series(y: Any) -> tuple[np.ndarray, pd.Index | None]:
    if isinstance(y, pd.Series):
        return np.asarray(y.to_numpy(dtype=float, copy=False), dtype=float), y.index.copy()

    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {arr.shape}")
    return arr, None


def _normalize_sktime_fh(fh: Any) -> tuple[int, ...]:
    if fh is None:
        raise ValueError("fh is required")

    if isinstance(fh, int):
        if int(fh) <= 0:
            raise ValueError("fh must contain only positive relative steps")
        return tuple(range(1, int(fh) + 1))

    is_relative = getattr(fh, "is_relative", None)
    if is_relative is False:
        raise ValueError("SktimeForecasterAdapter only supports relative forecasting horizons in v1")

    if hasattr(fh, "to_numpy"):
        raw = np.asarray(fh.to_numpy(), dtype=int)
    else:
        raw = np.asarray(list(fh), dtype=int)

    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("fh must be a non-empty 1D relative horizon")
    if np.any(raw <= 0):
        raise ValueError("fh must contain only positive relative steps")
    return tuple(int(v) for v in raw.tolist())


def _future_index_from_fh(train_index: pd.Index | None, fh_steps: tuple[int, ...]) -> pd.Index:
    if train_index is None or len(train_index) == 0:
        return pd.Index(fh_steps)

    last = train_index[-1]
    if isinstance(train_index, pd.RangeIndex):
        step = int(train_index.step or 1)
        start = int(last) + step
        stop = start + step * len(fh_steps)
        if fh_steps == tuple(range(1, len(fh_steps) + 1)):
            return pd.RangeIndex(start=start, stop=stop, step=step)
        return pd.Index([int(last) + step * fh for fh in fh_steps], name=train_index.name)

    freq = getattr(train_index, "freq", None)
    if freq is not None:
        return pd.Index([last + fh * freq for fh in fh_steps], name=train_index.name)

    inferred = getattr(train_index, "inferred_freq", None)
    if inferred:
        offset = pd.tseries.frequencies.to_offset(inferred)
        return pd.Index([last + fh * offset for fh in fh_steps], name=train_index.name)

    if isinstance(last, (int, np.integer)):
        return pd.Index([int(last) + fh for fh in fh_steps], name=train_index.name)

    return pd.Index(fh_steps, name=train_index.name)


def _clone_local_forecaster(forecaster: BaseForecaster) -> BaseForecaster:
    return copy.deepcopy(forecaster)


class SktimeForecasterAdapter:
    def __init__(self, forecaster: str | BaseForecaster, **model_params: Any) -> None:
        _require_sktime()
        if isinstance(forecaster, BaseForecaster) and model_params:
            raise ValueError("model_params are only supported when forecaster is a model key")

        self._forecaster_spec = forecaster
        self._model_params = dict(model_params)
        self._forecaster: BaseForecaster | None = None
        self._train_index: pd.Index | None = None
        self._fit_fh: tuple[int, ...] | None = None

    def _build_forecaster(self) -> BaseForecaster:
        if isinstance(self._forecaster_spec, BaseForecaster):
            return _clone_local_forecaster(self._forecaster_spec)

        from ..models.registry import make_forecaster_object

        return make_forecaster_object(str(self._forecaster_spec).strip(), **dict(self._model_params))

    def fit(self, y: Any, X: Any = None, fh: Any = None) -> SktimeForecasterAdapter:
        if X is not None:
            raise ValueError("SktimeForecasterAdapter does not support X in v1")

        train_y, train_index = _coerce_sktime_series(y)
        forecaster = self._build_forecaster().fit(train_y)

        self._forecaster = forecaster
        self._train_index = train_index
        self._fit_fh = None if fh is None else _normalize_sktime_fh(fh)
        return self

    def predict(self, fh: Any = None, X: Any = None) -> pd.Series:
        if X is not None:
            raise ValueError("SktimeForecasterAdapter does not support X in v1")
        if self._forecaster is None:
            raise RuntimeError("fit must be called before predict")

        fh_steps = self._fit_fh if fh is None else _normalize_sktime_fh(fh)
        if fh_steps is None:
            raise ValueError("fh is required")

        full_horizon = int(max(fh_steps))
        yhat = np.asarray(self._forecaster.predict(full_horizon), dtype=float)
        picked = np.asarray([yhat[step - 1] for step in fh_steps], dtype=float)
        future_index = _future_index_from_fh(self._train_index, fh_steps)
        return pd.Series(picked, index=future_index, dtype=float)


def make_sktime_forecaster_adapter(
    forecaster: str | BaseForecaster,
    **model_params: Any,
) -> SktimeForecasterAdapter:
    return SktimeForecasterAdapter(forecaster, **model_params)
