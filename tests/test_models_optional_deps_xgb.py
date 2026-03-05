import importlib.util

import numpy as np
import pytest

from foresight.models.registry import get_model_spec, make_forecaster

XGB_MODELS = [
    "xgb-dart-lag",
    "xgb-gamma-lag",
    "xgb-huber-lag",
    "xgb-lag",
    "xgb-linear-lag",
    "xgb-mae-lag",
    "xgb-poisson-lag",
    "xgb-quantile-lag",
    "xgb-tweedie-lag",
    "xgbrf-lag",
]


def test_xgb_models_are_registered_as_optional() -> None:
    for key in XGB_MODELS:
        spec = get_model_spec(key)
        assert "xgb" in spec.requires


def test_xgb_models_raise_importerror_when_xgboost_missing() -> None:
    if importlib.util.find_spec("xgboost") is not None:
        pytest.skip("xgboost installed; this test targets the missing-dep path")

    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    for key in XGB_MODELS:
        f = make_forecaster(key)
        with pytest.raises(ImportError):
            f(y, 2)


def test_xgb_models_smoke_when_installed() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed; smoke test requires it")

    y = 1.0 + np.sin(np.arange(160, dtype=float) / 3.0) + 0.1 * np.arange(160, dtype=float)
    horizon = 2

    cases = [
        ("xgb-lag", {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3}),
        ("xgb-dart-lag", {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3}),
        ("xgbrf-lag", {"lags": 12, "n_estimators": 10, "max_depth": 3}),
        ("xgb-linear-lag", {"lags": 12, "n_estimators": 50, "learning_rate": 0.1}),
        ("xgb-mae-lag", {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3}),
        (
            "xgb-huber-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "huber_slope": 1.0,
            },
        ),
        (
            "xgb-quantile-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "quantile_alpha": 0.5,
            },
        ),
        ("xgb-poisson-lag", {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3}),
        ("xgb-gamma-lag", {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3}),
        (
            "xgb-tweedie-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "tweedie_variance_power": 1.5,
            },
        ),
    ]

    for key, params in cases:
        f = make_forecaster(key, **params)
        yhat = f(y, horizon)
        assert yhat.shape == (horizon,)
        assert np.all(np.isfinite(yhat))
