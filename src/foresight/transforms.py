from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "TransformState",
    "MissingValueImputer",
    "StandardScaler",
    "Differencer",
    "BoxCoxTransformer",
    "fit_transform",
    "inverse_forecast",
    "normalize_transform_list",
]


@dataclass(frozen=True)
class TransformState:
    name: str
    params: dict[str, Any]


def _as_1d_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr


class MissingValueImputer:
    def __init__(self, *, strategy: str = "ffill", fill_value: float = 0.0) -> None:
        self.strategy = str(strategy).strip().lower()
        self.fill_value = float(fill_value)
        if self.strategy not in {"ffill", "zero", "mean"}:
            raise ValueError("strategy must be one of: ffill, zero, mean")
        self._mean: float | None = None

    def fit(self, y: Any) -> MissingValueImputer:
        arr = _as_1d_float_array(y)
        if self.strategy == "mean":
            finite = arr[np.isfinite(arr)]
            self._mean = float(np.mean(finite)) if finite.size > 0 else 0.0
        return self

    def transform(self, y: Any) -> np.ndarray:
        arr = _as_1d_float_array(y).astype(float, copy=True)
        mask = np.isnan(arr)
        if not mask.any():
            return arr

        if self.strategy == "zero":
            arr[mask] = self.fill_value
            return arr

        if self.strategy == "mean":
            mean = 0.0 if self._mean is None else float(self._mean)
            arr[mask] = mean
            return arr

        last = np.nan
        for i in range(arr.size):
            if np.isnan(arr[i]):
                if not np.isnan(last):
                    arr[i] = last
            else:
                last = arr[i]
        if np.isnan(arr).any():
            first_valid = arr[~np.isnan(arr)]
            fill = self.fill_value if first_valid.size == 0 else float(first_valid[0])
            arr[np.isnan(arr)] = fill
        return arr

    def inverse_transform(self, y: Any) -> np.ndarray:
        return _as_1d_float_array(y)


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: float | None = None
        self.scale_: float | None = None

    def fit(self, y: Any) -> StandardScaler:
        arr = _as_1d_float_array(y)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std <= 0.0 or not np.isfinite(std):
            std = 1.0
        self.mean_ = mean
        self.scale_ = std
        return self

    def transform(self, y: Any) -> np.ndarray:
        arr = _as_1d_float_array(y)
        mean = 0.0 if self.mean_ is None else float(self.mean_)
        std = 1.0 if self.scale_ is None else float(self.scale_)
        return (arr - mean) / std

    def inverse_transform(self, y: Any) -> np.ndarray:
        arr = _as_1d_float_array(y)
        mean = 0.0 if self.mean_ is None else float(self.mean_)
        std = 1.0 if self.scale_ is None else float(self.scale_)
        return arr * std + mean


class Differencer:
    def __init__(self, *, order: int = 1) -> None:
        self.order = int(order)
        if self.order <= 0:
            raise ValueError("order must be >= 1")
        self._last_values: list[float] = []

    def fit(self, y: Any) -> Differencer:
        arr = _as_1d_float_array(y)
        work = arr.astype(float, copy=True)
        self._last_values = []
        for _ in range(self.order):
            if work.size < 2:
                raise ValueError("Differencer requires at least order+1 points")
            self._last_values.append(float(work[-1]))
            work = np.diff(work)
        return self

    def transform(self, y: Any) -> np.ndarray:
        arr = _as_1d_float_array(y).astype(float, copy=True)
        for _ in range(self.order):
            if arr.size < 2:
                raise ValueError("Differencer requires at least order+1 points")
            arr = np.diff(arr)
        return arr

    def inverse_transform(self, y: Any) -> np.ndarray:
        arr = _as_1d_float_array(y).astype(float, copy=True)
        current = arr
        for last in reversed(self._last_values):
            current = float(last) + np.cumsum(current)
        return current


class BoxCoxTransformer:
    def __init__(self, *, lmbda: float = 0.0) -> None:
        self.lmbda = float(lmbda)

    def fit(self, y: Any) -> BoxCoxTransformer:
        arr = _as_1d_float_array(y)
        if np.any(arr <= 0.0):
            raise ValueError("Box-Cox transform requires strictly positive values")
        return self

    def transform(self, y: Any) -> np.ndarray:
        arr = _as_1d_float_array(y)
        if np.any(arr <= 0.0):
            raise ValueError("Box-Cox transform requires strictly positive values")
        if abs(self.lmbda) < 1e-12:
            return np.log(arr)
        return (np.power(arr, self.lmbda) - 1.0) / self.lmbda

    def inverse_transform(self, y: Any) -> np.ndarray:
        arr = _as_1d_float_array(y)
        if abs(self.lmbda) < 1e-12:
            return np.exp(arr)
        return np.power(self.lmbda * arr + 1.0, 1.0 / self.lmbda)


def _state_from_transformer(name: str, transformer: Any) -> TransformState:
    key = str(name).strip().lower()
    if key == "standardize":
        return TransformState(
            name=key, params={"mean": transformer.mean_, "std": transformer.scale_}
        )
    if key == "diff1":
        return TransformState(
            name=key,
            params={"order": transformer.order, "last_values": list(transformer._last_values)},
        )
    if key == "boxcox":
        return TransformState(name=key, params={"lmbda": transformer.lmbda})
    if key == "log1p":
        return TransformState(name=key, params={})
    raise ValueError(f"Unknown transform: {name!r}")


def _transformer_from_state(state: TransformState) -> Any:
    key = str(state.name).strip().lower()
    if key == "standardize":
        t = StandardScaler()
        t.mean_ = float(state.params["mean"])
        t.scale_ = float(state.params["std"])
        return t
    if key == "diff1":
        t = Differencer(order=int(state.params.get("order", 1)))
        t._last_values = [float(v) for v in state.params["last_values"]]
        return t
    if key == "boxcox":
        return BoxCoxTransformer(lmbda=float(state.params["lmbda"]))
    if key == "log1p":
        return None
    raise ValueError(f"Unknown transform state: {state.name!r}")


def fit_transform(name: str, y: Any) -> tuple[np.ndarray, TransformState]:
    """
    Fit a transform on `y` and return transformed values plus an invertible state.

    Supported names:
      - log1p
      - standardize
      - diff1
    """
    y_arr = _as_1d_float_array(y)
    key = str(name).strip().lower()

    if key == "log1p":
        return np.log1p(y_arr), TransformState(name="log1p", params={})
    if key == "standardize":
        t = StandardScaler().fit(y_arr)
        return t.transform(y_arr), _state_from_transformer(key, t)
    if key == "diff1":
        t = Differencer(order=1).fit(y_arr)
        return t.transform(y_arr), _state_from_transformer(key, t)
    if key == "boxcox":
        t = BoxCoxTransformer().fit(y_arr)
        return t.transform(y_arr), _state_from_transformer(key, t)

    raise ValueError(f"Unknown transform: {name!r}. Try: log1p, standardize, diff1, boxcox")


def inverse_forecast(state: TransformState, yhat: Any) -> np.ndarray:
    """
    Invert a forecast that was produced in the transformed space back to the original space.
    """
    yhat_arr = _as_1d_float_array(yhat)
    key = str(state.name).strip().lower()

    if key == "log1p":
        return np.expm1(yhat_arr)
    transformer = _transformer_from_state(state)
    if transformer is None:
        return np.expm1(yhat_arr)
    return transformer.inverse_transform(yhat_arr)


def normalize_transform_list(transforms: Any) -> tuple[str, ...]:
    """
    Normalize user input into a tuple of transform names.

    Accepts:
      - "" / None -> ()
      - "log1p" -> ("log1p",)
      - ("log1p","diff1") -> ("log1p","diff1")
    """
    if transforms is None:
        return ()

    if isinstance(transforms, str):
        s = transforms.strip()
        if not s:
            return ()
        return (s,)

    if isinstance(transforms, list | tuple):
        out: list[str] = []
        for t in transforms:
            s = str(t).strip()
            if s:
                out.append(s)
        return tuple(out)

    return (str(transforms).strip(),)
