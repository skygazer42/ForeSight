import numpy as np

from foresight.models.fourier import fourier_multi_regression_forecast


def test_fourier_multi_output_shape():
    rng = np.random.default_rng(0)
    t = np.arange(120, dtype=float)
    y = 0.1 * t + np.sin(2.0 * np.pi * t / 7.0) + 0.5 * np.sin(2.0 * np.pi * t / 30.0)
    y = y + 0.1 * rng.standard_normal(size=y.shape[0])

    yhat = fourier_multi_regression_forecast(
        y, 10, periods=(7, 30), orders=(3, 2), include_trend=True
    )
    assert yhat.shape == (10,)
    assert np.all(np.isfinite(yhat))


def test_fourier_multi_accepts_scalar_order():
    y = np.arange(60, dtype=float)
    yhat = fourier_multi_regression_forecast(y, 5, periods=(7, 12), orders=2, include_trend=True)
    assert yhat.shape == (5,)
