import importlib.util

import numpy as np
import pytest

from foresight.models.registry import get_model_spec, make_forecaster

XGB_MODELS = [
    "xgb-dart-lag",
    "xgb-dart-lag-recursive",
    "xgb-gamma-lag",
    "xgb-huber-lag",
    "xgb-lag",
    "xgb-lag-recursive",
    "xgb-linear-lag",
    "xgb-linear-lag-recursive",
    "xgb-logistic-lag",
    "xgb-mae-lag",
    "xgb-msle-lag",
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

    y_pos = 1.0 + np.sin(np.arange(160, dtype=float) / 3.0) + 0.1 * np.arange(160, dtype=float)
    y_01 = 0.5 + 0.4 * np.sin(np.arange(200, dtype=float) / 5.0)
    horizon = 2

    cases = [
        ("xgb-lag", {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3}, y_pos),
        (
            "xgb-lag-recursive",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-msle-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-logistic-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_01,
        ),
        (
            "xgb-dart-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-dart-lag-recursive",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        ("xgbrf-lag", {"lags": 12, "n_estimators": 10, "max_depth": 3}, y_pos),
        ("xgb-linear-lag", {"lags": 12, "n_estimators": 50, "learning_rate": 0.1}, y_pos),
        ("xgb-linear-lag-recursive", {"lags": 12, "n_estimators": 50, "learning_rate": 0.1}, y_pos),
        (
            "xgb-mae-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-huber-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "huber_slope": 1.0,
            },
            y_pos,
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
            y_pos,
        ),
        (
            "xgb-poisson-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-gamma-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-tweedie-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "tweedie_variance_power": 1.5,
            },
            y_pos,
        ),
    ]

    for key, params, y in cases:
        f = make_forecaster(key, **params)
        yhat = f(y, horizon)
        assert yhat.shape == (horizon,)
        assert np.all(np.isfinite(yhat))
