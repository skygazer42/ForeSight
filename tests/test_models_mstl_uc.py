import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_mstl_uc_tracks_multi_seasonal_trend_signal() -> None:
    t = np.arange(360, dtype=float)
    y = (
        10.0
        + 0.1 * t
        + 2.0 * np.sin(2.0 * np.pi * t / 12.0)
        + 0.75 * np.cos(2.0 * np.pi * t / 30.0)
    )

    f = make_forecaster(
        "mstl-uc",
        periods=(12, 30),
        level="local linear trend",
    )
    yhat = f(y, 12)

    tf = np.arange(360, 372, dtype=float)
    expected = (
        10.0
        + 0.1 * tf
        + 2.0 * np.sin(2.0 * np.pi * tf / 12.0)
        + 0.75 * np.cos(2.0 * np.pi * tf / 30.0)
    )

    assert yhat.shape == (12,)
    assert np.allclose(yhat, expected, atol=0.05)
