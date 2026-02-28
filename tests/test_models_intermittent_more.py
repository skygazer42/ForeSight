import numpy as np

from foresight.models.intermittent import (
    croston_optimized_forecast,
    croston_sba_forecast,
    croston_sbj_forecast,
    les_forecast,
)


def test_croston_sba_all_zero_returns_zero():
    yhat = croston_sba_forecast([0, 0, 0, 0], 3, alpha=0.1)
    assert yhat.shape == (3,)
    assert np.all(yhat == 0.0)


def test_croston_sbj_all_zero_returns_zero():
    yhat = croston_sbj_forecast([0, 0, 0, 0], 2, alpha=0.2)
    assert yhat.shape == (2,)
    assert np.all(yhat == 0.0)


def test_croston_opt_output_shape_and_nonnegative():
    y = [0, 0, 5, 0, 0, 0, 3, 0, 0, 2, 0, 0]
    yhat = croston_optimized_forecast(y, 5, grid_size=11)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))
    assert float(yhat.min()) >= 0.0


def test_les_all_zero_returns_zero():
    yhat = les_forecast([0, 0, 0, 0], 4, alpha=0.1, beta=0.1)
    assert yhat.shape == (4,)
    assert np.all(yhat == 0.0)


def test_les_decays_under_no_demand():
    # Trailing zeros should lead to a non-increasing multi-step path.
    y = [0, 0, 3, 0, 0, 0, 0, 0]
    yhat = les_forecast(y, 6, alpha=0.2, beta=0.2)
    assert yhat.shape == (6,)
    assert np.all(np.isfinite(yhat))
    assert float(yhat.min()) >= 0.0
    assert np.all(np.diff(yhat) <= 1e-10)
