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
    stability: str = ""
    capability_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def required_extra(self) -> str:
        reqs = [str(item).strip() for item in self.requires if str(item).strip()]
        return "core" if not reqs else "+".join(reqs)

    @property
    def stability_level(self) -> str:
        raw = str(self.stability).strip().lower()
        if raw in {"stable", "beta", "experimental"}:
            return raw

        key_l = str(self.key).strip().lower()
        reqs = {str(item).strip().lower() for item in self.requires if str(item).strip()}
        experimental_prefixes = (
            "torch-rnnpaper-",
            "torch-graph-attention-",
            "torch-graph-spectral-",
            "torch-graph-structure-",
            "torch-structured-rnn-",
            "torch-reservoir-",
            "hf-timeseries-",
        )
        experimental_keys = {
            "chronos",
            "chronos-bolt",
            "lag-llama",
            "moirai",
            "time-moe",
            "timer-s1",
            "timesfm",
        }
        if key_l.startswith(experimental_prefixes) or key_l in experimental_keys:
            return "experimental"
        if "torch" in reqs or "transformers" in reqs:
            return "beta"
        return "stable"

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
