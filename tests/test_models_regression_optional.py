import numpy as np
import pytest

from foresight.models.regression import ridge_lag_forecast


def test_ridge_lag_forecast_works_when_sklearn_available():
    pytest.importorskip("sklearn")
    y = np.arange(20, dtype=float)
    pred = ridge_lag_forecast(y, horizon=3, lags=1, alpha=0.0)
    assert np.allclose(pred, np.array([20.0, 21.0, 22.0]), atol=1e-6)
