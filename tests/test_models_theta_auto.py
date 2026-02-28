import numpy as np

from foresight.models.theta import theta_auto_forecast


def test_theta_auto_output_shape():
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    yhat = theta_auto_forecast(y, 3, grid_size=9)
    assert yhat.shape == (3,)
    assert np.all(np.isfinite(yhat))


def test_theta_auto_constant_series_is_constant():
    y = [2.0] * 20
    yhat = theta_auto_forecast(y, 5, grid_size=7)
    assert yhat.shape == (5,)
    assert np.allclose(yhat, 2.0)
