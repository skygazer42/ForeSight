from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from foresight.adapters import make_sktime_forecaster_adapter
from foresight.base import BaseForecaster


class LastValueForecaster(BaseForecaster):
    def __init__(self) -> None:
        super().__init__(model_key="example-last-value")
        self._last: float | None = None

    def fit(self, y: Any, X: Any = None) -> "LastValueForecaster":
        values = np.asarray(y, dtype=float)
        self._last = float(values[-1])
        self._is_fitted = True
        return self

    def predict(self, horizon: int, X: Any = None) -> np.ndarray:
        if not self._is_fitted or self._last is None:
            raise RuntimeError("fit must be called before predict")
        return np.asarray([self._last] * int(horizon), dtype=float)

    def train_schema_summary(self) -> dict[str, Any]:
        return {"kind": "local"}


def main() -> None:
    """
    sktime adapter example.

    Run after installing:
        pip install "foresight-ts[sktime]"
    """

    y = pd.Series([5.0, 6.0, 7.0], index=pd.RangeIndex(start=0, stop=3))
    adapter = make_sktime_forecaster_adapter(LastValueForecaster())
    yhat = adapter.fit(y).predict([1, 2])

    print(yhat.to_string())


if __name__ == "__main__":
    main()
