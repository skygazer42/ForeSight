import numpy as np

from foresight.models.analog import analog_knn_forecast


def test_analog_knn_output_shape():
    y = np.sin(np.arange(60, dtype=float) / 3.0)
    yhat = analog_knn_forecast(y, 5, lags=12, k=5, normalize=True, weights="uniform")
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


def test_analog_knn_distance_weights_runs():
    y = np.sin(np.arange(80, dtype=float) / 5.0) + 0.1 * np.arange(80, dtype=float)
    yhat = analog_knn_forecast(y, 3, lags=10, k=7, normalize=True, weights="distance")
    assert yhat.shape == (3,)
