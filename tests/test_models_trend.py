import numpy as np

from foresight.models.trend import poly_trend_forecast


def test_poly_trend_degree_1_exact_for_linear():
    t = np.arange(20, dtype=float)
    y = 2.0 + 3.0 * t
    yhat = poly_trend_forecast(y, 5, degree=1)
    t_future = np.arange(20, 25, dtype=float)
    y_true = 2.0 + 3.0 * t_future
    assert np.max(np.abs(yhat - y_true)) < 1e-8
