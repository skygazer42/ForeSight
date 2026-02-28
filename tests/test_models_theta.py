import numpy as np

from foresight.models.theta import theta_forecast


def test_theta_alpha_1_uses_half_slope_drift():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = theta_forecast(y, horizon=3, alpha=1.0)
    assert pred.tolist() == [4.5, 5.0, 5.5]
