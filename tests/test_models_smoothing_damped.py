import numpy as np

from foresight.models.smoothing import holt_damped_forecast, holt_forecast


def test_holt_damped_phi_1_matches_holt():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50).cumsum()
    yhat_holt = holt_forecast(y, 5, alpha=0.3, beta=0.2)
    yhat_damped = holt_damped_forecast(y, 5, alpha=0.3, beta=0.2, phi=1.0)
    assert np.max(np.abs(yhat_holt - yhat_damped)) < 1e-12


def test_holt_damped_phi_0_constant_forecast():
    y = np.arange(20, dtype=float)
    yhat = holt_damped_forecast(y, 4, alpha=0.3, beta=0.2, phi=0.0)
    assert yhat.shape == (4,)
    assert np.allclose(yhat, yhat[0])
