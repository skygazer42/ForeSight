import numpy as np

from foresight.models.fourier import fourier_regression_forecast


def test_fourier_regression_repeats_sine_wave():
    period = 12
    t = np.arange(36, dtype=float)
    y = np.sin(2.0 * np.pi * t / float(period))

    horizon = 12
    yhat = fourier_regression_forecast(y, horizon, period=period, order=1, include_trend=False)
    assert yhat.shape == (horizon,)

    t_future = np.arange(36, 36 + horizon, dtype=float)
    y_true = np.sin(2.0 * np.pi * t_future / float(period))
    assert np.max(np.abs(yhat - y_true)) < 1e-6
