import numpy as np

from foresight.models.spectral import fft_topk_forecast


def test_fft_topk_output_shape():
    rng = np.random.default_rng(0)
    t = np.arange(80, dtype=float)
    y = 0.05 * t + np.sin(2.0 * np.pi * t / 12.0) + 0.1 * rng.standard_normal(80)
    yhat = fft_topk_forecast(y, 6, top_k=3, include_trend=True)
    assert yhat.shape == (6,)
    assert np.all(np.isfinite(yhat))


def test_fft_topk_constant_series_is_constant():
    y = np.ones(30, dtype=float) * 7.0
    yhat = fft_topk_forecast(y, 4, top_k=3, include_trend=True)
    assert yhat.shape == (4,)
    assert np.allclose(yhat, 7.0)
