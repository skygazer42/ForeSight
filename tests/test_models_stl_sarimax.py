import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_stl_sarimax_tracks_trend_plus_seasonality() -> None:
    t = np.arange(120, dtype=float)
    y = 10.0 + 0.2 * t + 2.0 * np.sin(2.0 * np.pi * t / 12.0)

    f = make_forecaster(
        "stl-sarimax",
        period=12,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    yhat = f(y, 12)

    tf = np.arange(120, 132, dtype=float)
    expected = 10.0 + 0.2 * tf + 2.0 * np.sin(2.0 * np.pi * tf / 12.0)

    assert yhat.shape == (12,)
    assert np.allclose(yhat, expected, atol=1e-6)
