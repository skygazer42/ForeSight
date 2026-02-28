import numpy as np

from foresight.models.intermittent import adida_forecast


def test_adida_all_zero_returns_zero():
    yhat = adida_forecast([0, 0, 0, 0, 0, 0], 5, agg_period=3, base="ses", alpha=0.2)
    assert yhat.shape == (5,)
    assert np.all(yhat == 0.0)


def test_adida_output_shape_and_nonnegative():
    y = [0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 0, 2]
    yhat = adida_forecast(y, 7, agg_period=3, base="ses", alpha=0.2)
    assert yhat.shape == (7,)
    assert np.all(np.isfinite(yhat))
    assert float(yhat.min()) >= 0.0
