import numpy as np
import pytest

from foresight.models.baselines import (
    drift_forecast,
    mean_forecast,
    median_forecast,
    moving_average_forecast,
)


def test_mean_forecast_repeats_mean():
    pred = mean_forecast([1.0, 2.0, 3.0], horizon=2)
    assert pred.shape == (2,)
    assert pred.tolist() == [2.0, 2.0]


def test_median_forecast_repeats_median():
    pred = median_forecast([1.0, 100.0, 3.0], horizon=3)
    assert pred.shape == (3,)
    assert pred.tolist() == [3.0, 3.0, 3.0]


def test_drift_forecast_linear_trend():
    # First=10, last=16 over 4 steps => slope = (16-10)/(5-1)=1.5
    y = np.array([10.0, 11.0, 13.0, 14.0, 16.0])
    pred = drift_forecast(y, horizon=3)
    assert pred.shape == (3,)
    assert pred.tolist() == [17.5, 19.0, 20.5]


def test_moving_average_forecast_uses_last_window():
    y = [1.0, 2.0, 3.0, 100.0]
    pred = moving_average_forecast(y, horizon=2, window=3)
    # mean([2,3,100]) = 35
    assert pred.tolist() == [35.0, 35.0]


def test_baselines_validate_horizon():
    with pytest.raises(ValueError):
        mean_forecast([1.0], horizon=0)
