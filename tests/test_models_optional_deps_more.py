import importlib.util
import warnings

import numpy as np
import pytest

from foresight.models.registry import get_model_spec, make_forecaster

STATS_MODELS = [
    "auto-arima",
    "fourier-arima",
    "fourier-auto-arima",
    "fourier-ets",
    "fourier-uc",
    "fourier-autoreg",
    "fourier-sarimax",
    "sarimax",
    "autoreg",
    "uc-local-level",
    "uc-local-linear-trend",
    "uc-seasonal",
    "stl-autoreg",
    "stl-ets",
    "stl-arima",
    "stl-uc",
    "stl-auto-arima",
    "stl-sarimax",
    "mstl-autoreg",
    "mstl-ets",
    "mstl-arima",
    "mstl-uc",
    "mstl-sarimax",
    "mstl-auto-arima",
    "tbats-lite",
    "tbats-lite-autoreg",
    "tbats-lite-ets",
    "tbats-lite-sarimax",
    "tbats-lite-auto-arima",
    "tbats-lite-uc",
]


def test_new_statsmodels_models_are_registered_as_optional():
    for key in STATS_MODELS:
        spec = get_model_spec(key)
        assert "stats" in spec.requires


def test_new_statsmodels_models_raise_importerror_when_statsmodels_missing():
    if importlib.util.find_spec("statsmodels") is not None:
        pytest.skip("statsmodels installed; this test targets the missing-dep path")

    y = [float(i) for i in range(1, 25)]
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
        ("uc-seasonal", {"seasonal": 12}),
        ("stl-autoreg", {"period": 12, "lags": 1, "trend": "c"}),
        ("stl-ets", {"period": 12, "trend": "add"}),
        ("stl-arima", {"period": 12, "order": (1, 0, 0), "seasonal": 7}),
        ("stl-uc", {"period": 12, "level": "local level"}),
        (
            "stl-auto-arima",
            {
                "period": 12,
                "max_p": 0,
                "max_d": 1,
                "max_q": 0,
                "trend": "t",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
        ),
        (
            "stl-sarimax",
            {
                "period": 12,
                "order": (1, 0, 0),
                "seasonal_order": (0, 0, 0, 0),
                "trend": "c",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
        ),
        ("mstl-autoreg", {"periods": (12, 30), "lags": 0, "trend": "ct"}),
        ("mstl-ets", {"periods": (12, 30), "trend": "add"}),
        ("mstl-uc", {"periods": (12, 30), "level": "local linear trend"}),
        (
            "mstl-sarimax",
            {
                "periods": (12, 30),
                "order": (1, 0, 0),
                "seasonal_order": (0, 0, 0, 0),
                "trend": "c",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
        ),
        (
            "auto-arima",
            {
                "max_p": 1,
                "max_d": 1,
                "max_q": 1,
                "trend": "c",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
                "information_criterion": "aic",
            },
        ),
        (
            "fourier-arima",
            {
                "periods": (12,),
                "orders": 2,
                "order": (1, 0, 0),
                "trend": "c",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
        ),
        (
            "fourier-auto-arima",
            {
                "periods": (12,),
                "orders": 2,
                "max_p": 1,
                "max_d": 0,
                "max_q": 1,
                "trend": "c",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
        ),
        (
            "fourier-ets",
            {
                "periods": (12,),
                "orders": 2,
                "trend": None,
                "damped_trend": False,
            },
        ),
        (
            "fourier-uc",
            {
                "periods": (12,),
                "orders": 2,
                "level": "local linear trend",
            },
        ),
        (
            "fourier-autoreg",
            {
                "periods": (12,),
                "orders": 2,
                "lags": 0,
                "trend": "c",
            },
        ),
        (
            "fourier-sarimax",
            {
                "periods": (12,),
                "orders": 2,
                "order": (1, 0, 0),
                "seasonal_order": (0, 0, 0, 0),
                "trend": "c",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
        ),
        ("mstl-arima", {"periods": (12,), "order": (1, 0, 0)}),
        (
            "mstl-auto-arima",
            {
                "periods": (12,),
                "max_p": 1,
                "max_d": 1,
                "max_q": 1,
                "trend": "c",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
        ),
        ("tbats-lite", {"periods": (12,), "orders": 2, "arima_order": (1, 0, 0)}),
        (
            "tbats-lite-autoreg",
            {"periods": (12,), "orders": 2, "include_trend": True, "lags": 0, "trend": "n"},
        ),
        (
            "tbats-lite-ets",
            {"periods": (12,), "orders": 2, "include_trend": True, "trend": None},
        ),
        (
            "tbats-lite-sarimax",
            {
                "periods": (12,),
                "orders": 2,
                "include_trend": True,
                "order": (1, 0, 0),
                "seasonal_order": (0, 0, 0, 0),
                "trend": None,
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
        ),
        (
            "tbats-lite-auto-arima",
            {
                "periods": (12,),
                "orders": 2,
                "include_trend": True,
                "max_p": 1,
                "max_d": 1,
                "max_q": 1,
                "trend": "c",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
        ),
        (
            "tbats-lite-uc",
            {
                "periods": (12,),
                "orders": 2,
                "include_trend": True,
                "level": "local level",
            },
        ),
    ]

    for key, params in cases:
        f = make_forecaster(key, **params)
        yhat = f(y, 3)
        assert yhat.shape == (3,)
        assert np.all(np.isfinite(yhat))


def test_auto_arima_can_search_seasonal_orders_only_when_installed():
    if importlib.util.find_spec("statsmodels") is None:
        pytest.skip("statsmodels not installed; smoke test requires it")

    t = np.arange(1, 73, dtype=float)
    y = 20.0 + 3.0 * np.sin(2.0 * np.pi * t / 12.0)

    f = make_forecaster(
        "auto-arima",
        max_p=0,
        max_d=0,
        max_q=0,
        max_P=1,
        max_D=0,
        max_Q=1,
        seasonal_period=12,
        trend="c",
        information_criterion="aic",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yhat = f(y, 6)

    messages = [str(item.message) for item in caught]
    assert not any(
        "Non-invertible starting seasonal moving average Using zeros as starting parameters"
        in message
        for message in messages
    )
    assert not any(
        "Non-stationary starting seasonal autoregressive Using zeros as starting parameters"
        in message
        for message in messages
    )

    assert yhat.shape == (6,)
    assert np.all(np.isfinite(yhat))
