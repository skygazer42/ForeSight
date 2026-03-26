from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .contracts.frames import require_long_df as _contracts_require_long_df


def _as_1d_float_array(y: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {arr.shape}")
    return arr


def _unavailable_local_factory() -> Callable[[Any, int], np.ndarray]:
    raise RuntimeError("serialized local forecaster runtime factory was not rebuilt")


def _unavailable_global_factory() -> Callable[[pd.DataFrame, Any, int], pd.DataFrame]:
    raise RuntimeError("serialized global forecaster runtime factory was not rebuilt")


def _runtime_summary_for_model(
    *, model_key: str, model_params: dict[str, Any]
) -> dict[str, Any] | None:
    from .models.neural_runtime import summarize_model_runtime

    return summarize_model_runtime(
        model_key=str(model_key),
        model_params=dict(model_params),
    )


class BaseForecaster(ABC):
    def __init__(self, *, model_key: str, model_params: dict[str, Any]) -> None:
        self.model_key = str(model_key)
        self.model_params = dict(model_params)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return bool(self._is_fitted)

    @abstractmethod
    def fit(self, y: Any) -> BaseForecaster:
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def train_schema_summary(self) -> dict[str, Any]:
        raise NotImplementedError


class BaseGlobalForecaster(ABC):
    def __init__(self, *, model_key: str, model_params: dict[str, Any]) -> None:
        self.model_key = str(model_key)
        self.model_params = dict(model_params)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return bool(self._is_fitted)

    @abstractmethod
    def fit(self, long_df: Any) -> BaseGlobalForecaster:
        raise NotImplementedError

    @abstractmethod
    def predict(self, cutoff: Any, horizon: int) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def train_schema_summary(self) -> dict[str, Any]:
        raise NotImplementedError


class RegistryForecaster(BaseForecaster):
    def __init__(
        self,
        *,
        model_key: str,
        model_params: dict[str, Any],
        factory: Callable[[], Callable[[Any, int], np.ndarray]],
    ) -> None:
        super().__init__(model_key=model_key, model_params=model_params)
        self._factory = factory
        self._forecaster: Callable[[Any, int], np.ndarray] | None = None
        self._train_y: np.ndarray | None = None

    def _rebuild_runtime(self) -> None:
        from .models.factories import rebuild_local_forecaster_runtime

        self._factory = lambda: rebuild_local_forecaster_runtime(
            self.model_key,
            dict(self.model_params),
        )
        self._forecaster = self._factory() if self.is_fitted else None

    def fit(self, y: Any) -> RegistryForecaster:
        self._train_y = _as_1d_float_array(y).copy()
        self._forecaster = self._factory()
        self._is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self._forecaster is None or self._train_y is None:
            raise RuntimeError("fit must be called before predict")
        return np.asarray(self._forecaster(self._train_y, int(horizon)), dtype=float)

    def train_schema_summary(self) -> dict[str, Any]:
        summary = {
            "kind": "local",
            "n_obs": 0 if self._train_y is None else int(self._train_y.size),
        }
        runtime = _runtime_summary_for_model(
            model_key=self.model_key,
            model_params=self.model_params,
        )
        if runtime is not None:
            summary["runtime"] = runtime
        return summary

    def __getstate__(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_params": dict(self.model_params),
            "_is_fitted": bool(self._is_fitted),
            "_train_y": None if self._train_y is None else self._train_y.copy(),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.model_key = str(state["model_key"])
        self.model_params = dict(state["model_params"])
        self._is_fitted = bool(state.get("_is_fitted", False))
        train_y = state.get("_train_y")
        self._train_y = None if train_y is None else np.asarray(train_y, dtype=float)
        self._factory = _unavailable_local_factory
        self._forecaster = None
        self._rebuild_runtime()


class RegistryGlobalForecaster(BaseGlobalForecaster):
    def __init__(
        self,
        *,
        model_key: str,
        model_params: dict[str, Any],
        factory: Callable[[], Callable[[pd.DataFrame, Any, int], pd.DataFrame]],
    ) -> None:
        super().__init__(model_key=model_key, model_params=model_params)
        self._factory = factory
        self._forecaster: Callable[[pd.DataFrame, Any, int], pd.DataFrame] | None = None
        self._train_df: pd.DataFrame | None = None

    def _rebuild_runtime(self) -> None:
        from .models.factories import rebuild_global_forecaster_runtime

        self._factory = lambda: rebuild_global_forecaster_runtime(
            self.model_key,
            dict(self.model_params),
        )
        self._forecaster = self._factory() if self.is_fitted else None

    def fit(self, long_df: Any) -> RegistryGlobalForecaster:
        df = _contracts_require_long_df(long_df, require_non_empty=False)
        self._train_df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(
            drop=True
        )
        self._forecaster = self._factory()
        self._is_fitted = True
        return self

    def predict(self, cutoff: Any, horizon: int) -> pd.DataFrame:
        if not self.is_fitted or self._forecaster is None or self._train_df is None:
            raise RuntimeError("fit must be called before predict")
        out = self._forecaster(self._train_df, cutoff, int(horizon))
        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"Global forecaster predict must return DataFrame, got: {type(out).__name__}"
            )
        return out

    def train_schema_summary(self) -> dict[str, Any]:
        n_rows = 0 if self._train_df is None else int(len(self._train_df))
        n_series = 0 if self._train_df is None else int(self._train_df["unique_id"].nunique())
        cols = [] if self._train_df is None else [str(c) for c in self._train_df.columns.tolist()]
        summary = {
            "kind": "global",
            "n_rows": n_rows,
            "n_series": n_series,
            "columns": cols,
        }
        runtime = _runtime_summary_for_model(
            model_key=self.model_key,
            model_params=self.model_params,
        )
        if runtime is not None:
            summary["runtime"] = runtime
        return summary

    def __getstate__(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_params": dict(self.model_params),
            "_is_fitted": bool(self._is_fitted),
            "_train_df": None if self._train_df is None else self._train_df.copy(),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.model_key = str(state["model_key"])
        self.model_params = dict(state["model_params"])
        self._is_fitted = bool(state.get("_is_fitted", False))
        train_df = state.get("_train_df")
        self._train_df = None if train_df is None else pd.DataFrame(train_df).copy()
        self._factory = _unavailable_global_factory
        self._forecaster = None
        self._rebuild_runtime()
