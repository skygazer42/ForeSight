import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_uc_seasonal_tracks_repeating_pattern() -> None:
    pattern = np.array([10.0, 13.0, 9.0, 12.0], dtype=float)
    y = np.tile(pattern, 20)

    f = make_forecaster("uc-seasonal", seasonal=4)
    yhat = f(y, 8)

    expected = np.tile(pattern, 2)
    assert yhat.shape == (8,)
    assert np.allclose(yhat, expected, atol=0.5)
