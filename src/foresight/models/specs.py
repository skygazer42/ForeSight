from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

LocalForecasterFn = Callable[[Any, int], np.ndarray]
GlobalForecasterFn = Callable[[pd.DataFrame, Any, int], pd.DataFrame]
MultivariateForecasterFn = Callable[[Any, int], np.ndarray]
ModelFactory = Callable[..., Any]
ForecasterFn = LocalForecasterFn


@dataclass(frozen=True)
class ModelSpec:
    key: str
    description: str
    factory: ModelFactory
    default_params: dict[str, Any] = field(default_factory=dict)
    param_help: dict[str, str] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    interface: str = "local"
    capability_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def capabilities(self) -> dict[str, Any]:
        supports_x_cols = "x_cols" in self.param_help
        supports_static_cols = "static_cols" in self.param_help
        supports_quantiles = "quantiles" in self.param_help
        supports_interval_forecast = str(self.interface) == "local" or supports_quantiles
        supports_interval_forecast_with_x_cols = supports_x_cols and supports_quantiles
        supports_artifact_save = str(self.interface) in {"local", "global"} and not (
            str(self.interface) == "local" and supports_x_cols
        )
        requires_future_covariates = False

        capabilities = {
            "supports_x_cols": supports_x_cols,
            "supports_static_cols": supports_static_cols,
            "supports_quantiles": supports_quantiles,
            "supports_interval_forecast": supports_interval_forecast,
            "supports_interval_forecast_with_x_cols": supports_interval_forecast_with_x_cols,
            "supports_artifact_save": supports_artifact_save,
            "requires_future_covariates": requires_future_covariates,
        }
        capabilities.update(dict(self.capability_overrides))
        return capabilities


__all__ = [
    "ForecasterFn",
    "GlobalForecasterFn",
    "LocalForecasterFn",
    "ModelFactory",
    "ModelSpec",
    "MultivariateForecasterFn",
]
