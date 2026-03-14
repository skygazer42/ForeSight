import numpy as np
import pytest

from foresight.models.ar import (
    ar_ols_forecast,
    ar_ols_auto_forecast,
    ar_ols_lags_forecast,
    sar_ols_forecast,
    select_ar_order_aic,
)


def test_ar_ols_lags_period2():
    y = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=float)
    yhat = ar_ols_lags_forecast(y, 2, lags=(2,))
    assert yhat.shape == (2,)
    assert np.allclose(yhat, np.array([1.0, 2.0]))


def test_sar_ols_period2_seasonal_lag():
    y = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=float)
    yhat = sar_ols_forecast(y, 2, p=0, P=1, season_length=2)
    assert yhat.shape == (2,)
    assert np.allclose(yhat, np.array([1.0, 2.0]))


def test_select_ar_order_aic_finds_a_low_order_periodic_model():
    y = np.array([1, 2] * 30, dtype=float)
    p = select_ar_order_aic(y, max_p=5)
    assert p in {1, 2}
    assert np.allclose(ar_ols_forecast(y, 4, p=p), np.array([1.0, 2.0, 1.0, 2.0]))


def test_ar_ols_auto_forecast_shape_and_finite():
    y = np.array([1, 2] * 30, dtype=float)
    yhat = ar_ols_auto_forecast(y, 3, max_p=5)
    assert yhat.shape == (3,)
    assert np.all(np.isfinite(yhat))


def test_ar_ols_lags_invalid_raises():
    with pytest.raises(ValueError):
        ar_ols_lags_forecast([1, 2, 3], 1, lags=())
