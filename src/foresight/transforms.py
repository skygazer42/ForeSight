from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TransformState:
    name: str
    params: dict[str, float]


def _as_1d_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr


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
        mean = float(np.mean(y_arr))
        std = float(np.std(y_arr))
        if std <= 0.0 or not np.isfinite(std):
            std = 1.0
        yt = (y_arr - mean) / std
        return yt, TransformState(name="standardize", params={"mean": mean, "std": std})

    if key == "diff1":
        if y_arr.size < 2:
            raise ValueError("diff1 transform requires at least 2 points")
        last = float(y_arr[-1])
        yt = np.diff(y_arr)
        return yt, TransformState(name="diff1", params={"last": last})

    raise ValueError(f"Unknown transform: {name!r}. Try: log1p, standardize, diff1")


def inverse_forecast(state: TransformState, yhat: Any) -> np.ndarray:
    """
    Invert a forecast that was produced in the transformed space back to the original space.
    """
    yhat_arr = _as_1d_float_array(yhat)
    key = str(state.name).strip().lower()

    if key == "log1p":
        return np.expm1(yhat_arr)

    if key == "standardize":
        mean = float(state.params["mean"])
        std = float(state.params["std"])
        return yhat_arr * std + mean

    if key == "diff1":
        last = float(state.params["last"])
        return last + np.cumsum(yhat_arr)

    raise ValueError(f"Unknown transform state: {state.name!r}")


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
