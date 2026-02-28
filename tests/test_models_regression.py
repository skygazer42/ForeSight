import numpy as np

from foresight.models.regression import lr_lag_forecast


def test_lr_lag_forecast_extrapolates_simple_trend_with_lag1():
    y = np.arange(10, dtype=float)  # 0..9
    pred = lr_lag_forecast(y, horizon=3, lags=1)
    assert np.allclose(pred, np.array([10.0, 11.0, 12.0]), atol=1e-9)
