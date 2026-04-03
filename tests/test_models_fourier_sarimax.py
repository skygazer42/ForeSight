import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_fourier_sarimax_tracks_future_fourier_signal() -> None:
    t = np.arange(120, dtype=float)
    y = 5.0 + 1.5 * np.sin(2.0 * np.pi * t / 7.0) + 0.75 * np.cos(2.0 * np.pi * t / 30.0)

    f = make_forecaster(
        "fourier-sarimax",
        periods=(7, 30),
        orders=(2, 1),
        order=(0, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
    )
    yhat = f(y, 8)

    tf = np.arange(120, 128, dtype=float)
    expected = 5.0 + 1.5 * np.sin(2.0 * np.pi * tf / 7.0) + 0.75 * np.cos(2.0 * np.pi * tf / 30.0)

    assert yhat.shape == (8,)
    assert np.allclose(yhat, expected, atol=1e-2)
