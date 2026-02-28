import importlib.util

import numpy as np
import pytest

from foresight.models.registry import get_model_spec, make_forecaster

STATS_MODELS = [
    "auto-arima",
    "sarimax",
    "autoreg",
    "uc-local-level",
    "uc-local-linear-trend",
    "stl-arima",
    "mstl-arima",
    "mstl-auto-arima",
    "tbats-lite",
]


def test_new_statsmodels_models_are_registered_as_optional():
    for key in STATS_MODELS:
        spec = get_model_spec(key)
        assert "stats" in spec.requires


def test_new_statsmodels_models_raise_importerror_when_statsmodels_missing():
    if importlib.util.find_spec("statsmodels") is not None:
        pytest.skip("statsmodels installed; this test targets the missing-dep path")

    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    for key in STATS_MODELS:
        f = make_forecaster(key)
        with pytest.raises(ImportError):
            f(y, 2)


def test_new_statsmodels_models_smoke_when_installed():
    if importlib.util.find_spec("statsmodels") is None:
        pytest.skip("statsmodels not installed; smoke test requires it")

    y = np.sin(np.arange(80, dtype=float) / 3.0) + 0.1 * np.arange(80, dtype=float)

    cases = [
        ("sarimax", {"order": (1, 0, 0), "seasonal_order": (0, 0, 0, 0)}),
        ("autoreg", {"lags": 5, "trend": "c"}),
        ("uc-local-level", {}),
        ("uc-local-linear-trend", {}),
        ("stl-arima", {"period": 12, "order": (1, 0, 0), "seasonal": 7}),
        ("auto-arima", {"max_p": 1, "max_d": 1, "max_q": 1, "information_criterion": "aic"}),
        ("mstl-arima", {"periods": (12,), "order": (1, 0, 0)}),
        ("mstl-auto-arima", {"periods": (12,), "max_p": 1, "max_d": 1, "max_q": 1}),
        ("tbats-lite", {"periods": (12,), "orders": 2, "arima_order": (1, 0, 0)}),
    ]

    for key, params in cases:
        f = make_forecaster(key, **params)
        yhat = f(y, 3)
        assert yhat.shape == (3,)
        assert np.all(np.isfinite(yhat))
