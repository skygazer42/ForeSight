import numpy as np

from foresight.models.smoothing import (
    holt_auto_forecast,
    holt_winters_additive_auto_forecast,
    ses_auto_forecast,
)


def test_ses_auto_forecast_shape():
    y = np.arange(30, dtype=float)
    yhat = ses_auto_forecast(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


def test_holt_auto_forecast_shape():
    y = np.arange(30, dtype=float)
    yhat = holt_auto_forecast(y, 4)
    assert yhat.shape == (4,)
    assert np.all(np.isfinite(yhat))


def test_hw_additive_auto_forecast_shape():
    t = np.arange(48, dtype=float)
    y = 10.0 + 0.1 * t + np.sin(2.0 * np.pi * t / 12.0)
    yhat = holt_winters_additive_auto_forecast(y, 3, season_length=12, grid_size=4)
    assert yhat.shape == (3,)
    assert np.all(np.isfinite(yhat))
