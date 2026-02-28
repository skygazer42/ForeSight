import numpy as np

from foresight.models.baselines import seasonal_mean_forecast


def test_seasonal_mean_forecast_repeats_season_means():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pred = seasonal_mean_forecast(y, horizon=5, season_length=3)
    assert pred.tolist() == [2.5, 3.5, 4.5, 2.5, 3.5]
