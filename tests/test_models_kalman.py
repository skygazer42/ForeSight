import numpy as np

from foresight.models.kalman import (
    kalman_local_level_forecast,
    kalman_local_linear_trend_forecast,
)


def test_kalman_local_level_output_shape():
    y = np.arange(1, 51, dtype=float)
    yhat = kalman_local_level_forecast(y, 4)
    assert yhat.shape == (4,)
    assert np.all(np.isfinite(yhat))


def test_kalman_local_linear_trend_output_shape():
    y = np.arange(1, 51, dtype=float)
    yhat = kalman_local_linear_trend_forecast(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


def test_kalman_local_level_constant_series_is_constant():
    y = np.ones(40, dtype=float) * 3.0
    yhat = kalman_local_level_forecast(y, 3)
    assert np.allclose(yhat, 3.0)
