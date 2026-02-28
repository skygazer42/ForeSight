import numpy as np

from foresight.models.smoothing import holt_forecast, holt_winters_additive_forecast, ses_forecast


def test_ses_alpha_1_matches_naive_last():
    y = np.array([1.0, 2.0, 3.0])
    pred = ses_forecast(y, horizon=2, alpha=1.0)
    assert pred.tolist() == [3.0, 3.0]


def test_holt_alpha_beta_1_extrapolates_linear_trend():
    y = np.array([1.0, 2.0, 3.0])
    pred = holt_forecast(y, horizon=2, alpha=1.0, beta=1.0)
    assert pred.tolist() == [4.0, 5.0]


def test_holt_winters_additive_repeats_simple_seasonality():
    y = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    pred = holt_winters_additive_forecast(
        y, horizon=4, season_length=2, alpha=1.0, beta=0.0, gamma=1.0
    )
    assert pred.tolist() == [1.0, 2.0, 1.0, 2.0]
