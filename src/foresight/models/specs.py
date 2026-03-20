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
        interface_s = str(self.interface).strip().lower()
        supports_x_cols = "x_cols" in self.param_help
        supports_static_cols = "static_cols" in self.param_help
        supports_quantiles = "quantiles" in self.param_help
        supports_interval_forecast = interface_s == "local" or supports_quantiles
        supports_interval_forecast_with_x_cols = supports_x_cols and supports_quantiles
        supports_artifact_save = interface_s in {"local", "global"} and not (
            interface_s == "local" and supports_x_cols
        )
        supports_panel = interface_s in {"local", "global"}
        supports_univariate = interface_s in {"local", "global"}
        supports_multivariate = interface_s == "multivariate"
        supports_probabilistic = supports_interval_forecast or supports_quantiles
        supports_conformal_eval = interface_s in {"local", "global"}
        supports_future_covariates = supports_x_cols
        supports_historic_covariates = supports_x_cols
        supports_static_covariates = supports_static_cols
        supports_refit_free_cv = False
        requires_future_covariates = False

        capabilities = {
            "supports_panel": supports_panel,
            "supports_univariate": supports_univariate,
            "supports_multivariate": supports_multivariate,
            "supports_probabilistic": supports_probabilistic,
            "supports_conformal_eval": supports_conformal_eval,
            "supports_future_covariates": supports_future_covariates,
            "supports_historic_covariates": supports_historic_covariates,
            "supports_static_covariates": supports_static_covariates,
            "supports_refit_free_cv": supports_refit_free_cv,
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
