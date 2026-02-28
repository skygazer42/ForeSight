import importlib.util

import numpy as np
import pytest

from foresight.models.regression import rf_lag_direct_forecast


def test_rf_lag_direct_forecast_shape():
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn not installed")

    y = np.arange(80, dtype=float)
    yhat = rf_lag_direct_forecast(y, 3, lags=6, n_estimators=20, random_state=0)
    assert yhat.shape == (3,)
    assert np.all(np.isfinite(yhat))
