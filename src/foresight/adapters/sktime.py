from __future__ import annotations

import copy
import inspect
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from ..base import BaseForecaster
from ..optional_deps import missing_dependency_message

__all__ = [
    "SktimeForecasterAdapter",
    "make_sktime_forecaster_adapter",
]

_BETA_X_SUPPORT_ERROR = (
    "SktimeForecasterAdapter supports X only for local single-series xreg forecasters in beta"
)


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


def _normalize_name_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        s = raw.strip()
        return tuple(part.strip() for part in s.split(",") if part.strip()) if s else ()
    if isinstance(raw, Iterable):
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            value = str(item).strip()
            if value and value not in seen:
                seen.add(value)
                out.append(value)
        return tuple(out)
    value = str(raw).strip()
    return (value,) if value else ()


def _configured_x_cols(model_params: dict[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for name in ("future_x_cols", "x_cols", "historic_x_cols"):
        for col in _normalize_name_tuple(model_params.get(name, ())):
            if col not in seen:
                seen.add(col)
                out.append(col)
    return tuple(out)


def _coerce_sktime_X(
    X: Any,
    *,
    expected_rows: int,
    expected_index: pd.Index | None,
    x_cols: tuple[str, ...],
) -> pd.DataFrame:
    explicit_index = isinstance(X, (pd.DataFrame, pd.Series))
    if isinstance(X, pd.DataFrame):
        out = X.copy()
    elif isinstance(X, pd.Series):
        out = X.to_frame()
    else:
        out = pd.DataFrame(X)

    if len(out) != int(expected_rows):
        raise ValueError(f"X must contain exactly {int(expected_rows)} rows")

    if (
        explicit_index
        and isinstance(expected_index, pd.Index)
        and len(expected_index) == len(out)
        and isinstance(out.index, pd.Index)
        and len(out.index) == len(expected_index)
        and not out.index.equals(expected_index)
    ):
        raise ValueError("X index must align with the forecasting horizon")

    if (
        x_cols
        and len(out.columns) == len(x_cols)
        and list(out.columns) != list(x_cols)
    ):
        if isinstance(out.columns, pd.RangeIndex) or all(
            isinstance(col, (int, np.integer)) for col in out.columns.tolist()
        ):
            out.columns = list(x_cols)
    return out


def _forecaster_accepts_X(forecaster: BaseForecaster) -> bool:
    fit_sig = inspect.signature(forecaster.fit)
    predict_sig = inspect.signature(forecaster.predict)
    return "X" in fit_sig.parameters and "X" in predict_sig.parameters


def _supports_beta_X_path(
    *,
    forecaster_spec: str | BaseForecaster,
    forecaster: BaseForecaster,
    model_params: dict[str, Any],
) -> bool:
    x_cols = _configured_x_cols(getattr(forecaster, "model_params", {}) or model_params)
    if not x_cols:
        return False
    if not _forecaster_accepts_X(forecaster):
        return False
    if isinstance(forecaster_spec, BaseForecaster):
        return True

    from ..models.registry import get_model_spec

    spec = get_model_spec(str(forecaster_spec).strip())
    return bool(spec.capabilities.get("supports_x_cols", False))


def _fit_forecaster_with_optional_X(
    forecaster: BaseForecaster,
    train_y: np.ndarray,
    X: pd.DataFrame | None,
) -> BaseForecaster:
    if X is None:
        return forecaster.fit(train_y)
    return forecaster.fit(train_y, X=X)


def _predict_forecaster_with_optional_X(
    forecaster: BaseForecaster,
    horizon: int,
    X: pd.DataFrame | None,
) -> np.ndarray:
    if X is None:
        return np.asarray(forecaster.predict(horizon), dtype=float)
    return np.asarray(forecaster.predict(horizon, X=X), dtype=float)


def _relative_steps_from_absolute_fh(
    train_index: pd.Index | None,
    fh: Any,
) -> tuple[int, ...]:
    if train_index is None or len(train_index) == 0:
        raise ValueError("absolute fh requires a pandas training index in v2")

    if hasattr(fh, "to_pandas"):
        absolute = fh.to_pandas()
    elif isinstance(fh, pd.Index):
        absolute = fh
    else:
        absolute = pd.Index(list(fh))

    if absolute.empty:
        raise ValueError("fh must be a non-empty 1D relative or absolute horizon")

    if isinstance(train_index, pd.RangeIndex):
        step = int(train_index.step or 1)
        last = int(train_index[-1])
        steps: list[int] = []
        for label in absolute.tolist():
            delta = int(label) - last
            if delta <= 0 or delta % step != 0:
                raise ValueError(
                    "absolute fh must be strictly future and aligned to the training index"
                )
            steps.append(delta // step)
        return tuple(steps)

    if isinstance(train_index, pd.DatetimeIndex):
        freq = train_index.freq
        if freq is None:
            inferred = train_index.inferred_freq
            freq = None if inferred is None else pd.tseries.frequencies.to_offset(inferred)
        if freq is None:
            raise ValueError("absolute fh requires a regular pandas training index in v2")

        absolute_dt = pd.DatetimeIndex(absolute)
        last = train_index[-1]
        steps = []
        for label in absolute_dt:
            if label <= last:
                raise ValueError(
                    "absolute fh must be strictly future and aligned to the training index"
                )
            full = pd.date_range(start=last, end=label, freq=freq)
            if len(full) < 2 or full[-1] != label:
                raise ValueError(
                    "absolute fh must be strictly future and aligned to the training index"
                )
            steps.append(len(full) - 1)
        return tuple(steps)

    raise ValueError("absolute fh requires a RangeIndex or DatetimeIndex in v2")


def _normalize_sktime_fh(
    fh: Any,
    *,
    train_index: pd.Index | None = None,
) -> tuple[int, ...]:
    if fh is None:
        raise ValueError("fh is required")

    if isinstance(fh, int):
        if int(fh) <= 0:
            raise ValueError("fh must contain only positive relative steps")
        return tuple(range(1, int(fh) + 1))

    is_relative = getattr(fh, "is_relative", None)
    if is_relative is False:
        return _relative_steps_from_absolute_fh(train_index, fh)

    if train_index is not None and isinstance(fh, pd.Index):
        return _relative_steps_from_absolute_fh(train_index, fh)

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
        self._x_cols: tuple[str, ...] = ()
        self._supports_X: bool = False

    def _build_forecaster(self) -> BaseForecaster:
        if isinstance(self._forecaster_spec, BaseForecaster):
            return _clone_local_forecaster(self._forecaster_spec)

        from ..models.registry import make_forecaster_object

        return make_forecaster_object(
            str(self._forecaster_spec).strip(), **dict(self._model_params)
        )

    def fit(self, y: Any, X: Any = None, fh: Any = None) -> SktimeForecasterAdapter:
        train_y, train_index = _coerce_sktime_series(y)
        forecaster = self._build_forecaster()
        self._x_cols = _configured_x_cols(getattr(forecaster, "model_params", {}) or self._model_params)
        self._supports_X = _supports_beta_X_path(
            forecaster_spec=self._forecaster_spec,
            forecaster=forecaster,
            model_params=self._model_params,
        )
        X_frame = None
        if X is not None:
            if not self._supports_X:
                raise ValueError(_BETA_X_SUPPORT_ERROR)
            X_frame = _coerce_sktime_X(
                X,
                expected_rows=int(train_y.size),
                expected_index=train_index,
                x_cols=self._x_cols,
            )

        forecaster = _fit_forecaster_with_optional_X(forecaster, train_y, X_frame)

        self._forecaster = forecaster
        self._train_index = train_index
        self._fit_fh = None if fh is None else _normalize_sktime_fh(fh, train_index=train_index)
        return self

    def predict(self, fh: Any = None, X: Any = None) -> pd.Series:
        if self._forecaster is None:
            raise RuntimeError("fit must be called before predict")

        fh_steps = (
            self._fit_fh if fh is None else _normalize_sktime_fh(fh, train_index=self._train_index)
        )
        if fh_steps is None:
            raise ValueError("fh is required")

        full_horizon = int(max(fh_steps))
        future_index = _future_index_from_fh(
            self._train_index,
            tuple(range(1, full_horizon + 1)),
        )
        X_frame = None
        if X is not None:
            if not self._supports_X:
                raise ValueError(_BETA_X_SUPPORT_ERROR)
            X_frame = _coerce_sktime_X(
                X,
                expected_rows=full_horizon,
                expected_index=future_index,
                x_cols=self._x_cols,
            )

        yhat = _predict_forecaster_with_optional_X(self._forecaster, full_horizon, X_frame)
        picked = np.asarray([yhat[step - 1] for step in fh_steps], dtype=float)
        output_index = _future_index_from_fh(self._train_index, fh_steps)
        return pd.Series(picked, index=output_index, dtype=float)


def make_sktime_forecaster_adapter(
    forecaster: str | BaseForecaster,
    **model_params: Any,
) -> SktimeForecasterAdapter:
    return SktimeForecasterAdapter(forecaster, **model_params)
