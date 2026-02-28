import numpy as np

from foresight.models.intermittent import croston_classic_forecast, tsb_forecast


def test_croston_classic_all_zero_returns_zero():
    yhat = croston_classic_forecast([0, 0, 0, 0], 3, alpha=0.1)
    assert yhat.shape == (3,)
    assert np.all(yhat == 0.0)


def test_croston_classic_positive_forecast_for_intermittent():
    y = [0, 0, 5, 0, 0, 0, 3, 0, 0]
    yhat = croston_classic_forecast(y, 4, alpha=0.2)
    assert yhat.shape == (4,)
    assert np.all(np.isfinite(yhat))
    assert float(yhat[0]) >= 0.0


def test_tsb_all_zero_returns_zero():
    yhat = tsb_forecast([0, 0, 0, 0], 2, alpha=0.1, beta=0.1)
    assert yhat.shape == (2,)
    assert np.all(yhat == 0.0)


def test_tsb_positive_forecast_for_intermittent():
    y = [0, 0, 5, 0, 0, 0, 3, 0, 0]
    yhat = tsb_forecast(y, 3, alpha=0.2, beta=0.2)
    assert yhat.shape == (3,)
    assert np.all(np.isfinite(yhat))
    assert float(yhat[0]) >= 0.0
