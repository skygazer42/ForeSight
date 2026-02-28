import numpy as np

from foresight.models.regression import lr_lag_direct_forecast


def test_lr_lag_direct_forecast_shape():
    y = np.arange(50, dtype=float)
    yhat = lr_lag_direct_forecast(y, 5, lags=5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))
