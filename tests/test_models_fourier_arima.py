import importlib.util
import warnings

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_fourier_arima_tracks_future_fourier_signal() -> None:
    t = np.arange(120, dtype=float)
    y = (
        5.0
        + 1.5 * np.sin(2.0 * np.pi * t / 7.0)
        + 0.75 * np.cos(2.0 * np.pi * t / 30.0)
    )

    f = make_forecaster(
        "fourier-arima",
        periods=(7, 30),
        orders=(2, 1),
        order=(0, 0, 0),
        trend="c",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yhat = f(y, 8)

    messages = [str(item.message) for item in caught]
    assert not any(
        "Maximum Likelihood optimization failed to converge" in message
        for message in messages
    )

    tf = np.arange(120, 128, dtype=float)
    expected = (
        5.0
        + 1.5 * np.sin(2.0 * np.pi * tf / 7.0)
        + 0.75 * np.cos(2.0 * np.pi * tf / 30.0)
    )

    assert yhat.shape == (8,)
    assert np.allclose(yhat, expected, atol=1e-2)
