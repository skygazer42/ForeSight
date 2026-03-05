import importlib.util

import numpy as np
import pytest

from foresight.models.registry import get_model_spec, make_forecaster

XGB_MODELS = [
    "xgb-lag",
    "xgb-dart-lag",
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

    y = np.sin(np.arange(160, dtype=float) / 3.0) + 0.1 * np.arange(160, dtype=float)

    cases = [
        ("xgb-lag", {"lags": 12, "n_estimators": 20, "learning_rate": 0.1, "max_depth": 3}),
        ("xgb-dart-lag", {"lags": 12, "n_estimators": 20, "learning_rate": 0.1, "max_depth": 3}),
        ("xgbrf-lag", {"lags": 12, "n_estimators": 20, "max_depth": 3}),
    ]

    for key, params in cases:
        f = make_forecaster(key, **params)
        yhat = f(y, 3)
        assert yhat.shape == (3,)
        assert np.all(np.isfinite(yhat))
