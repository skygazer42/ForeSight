import importlib.util

import numpy as np
import pytest

from foresight.models.registry import get_model_spec, make_forecaster

CATBOOST_MODELS = [
    "catboost-custom-dirrec-lag",
    "catboost-custom-lag",
    "catboost-custom-lag-recursive",
    "catboost-custom-step-lag",
    "catboost-dirrec-lag",
    "catboost-lag",
    "catboost-lag-recursive",
    "catboost-step-lag",
]


def test_catboost_models_are_registered_as_optional() -> None:
    for key in CATBOOST_MODELS:
        spec = get_model_spec(key)
        assert "catboost" in spec.requires


def test_catboost_models_raise_importerror_when_catboost_missing() -> None:
    if importlib.util.find_spec("catboost") is not None:
        pytest.skip("catboost installed; this test targets the missing-dep path")

    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    for key in CATBOOST_MODELS:
        f = make_forecaster(key)
        with pytest.raises(ImportError):
            f(y, 2)


def test_catboost_models_smoke_when_installed() -> None:
    if importlib.util.find_spec("catboost") is None:
        pytest.skip("catboost not installed; smoke test requires it")

    y = 1.0 + np.sin(np.arange(160, dtype=float) / 3.0) + 0.1 * np.arange(160, dtype=float)
    horizon = 2

    base_params = {
        "lags": 12,
        "iterations": 20,
        "learning_rate": 0.1,
        "depth": 4,
        "random_seed": 0,
        "thread_count": 1,
    }

    cases = [
        ("catboost-custom-lag", dict(base_params), y),
        ("catboost-custom-lag-recursive", dict(base_params), y),
        ("catboost-custom-step-lag", {**base_params, "step_scale": "one_based"}, y),
        ("catboost-custom-dirrec-lag", dict(base_params), y),
        ("catboost-lag", dict(base_params), y),
        ("catboost-lag-recursive", dict(base_params), y),
        ("catboost-step-lag", {**base_params, "step_scale": "one_based"}, y),
        ("catboost-dirrec-lag", dict(base_params), y),
    ]

    for key, params, series in cases:
        f = make_forecaster(key, **params)
        yhat = f(series, horizon)
        assert yhat.shape == (horizon,)
        assert np.all(np.isfinite(yhat))

