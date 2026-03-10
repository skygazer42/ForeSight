import numpy as np

import pytest

from foresight.models.smoothing import (
    holt_forecast,
    holt_winters_additive_forecast,
    holt_winters_multiplicative_forecast,
    ses_forecast,
)


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


def test_holt_winters_multiplicative_repeats_simple_seasonality() -> None:
    y = np.array([5.0, 15.0, 5.0, 15.0, 5.0, 15.0])
    pred = holt_winters_multiplicative_forecast(
        y, horizon=4, season_length=2, alpha=1.0, beta=0.0, gamma=1.0
    )
    assert pred.tolist() == [5.0, 15.0, 5.0, 15.0]


def test_holt_winters_multiplicative_rejects_non_positive_values() -> None:
    y = np.array([1.0, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="strictly positive"):
        holt_winters_multiplicative_forecast(
            y, horizon=1, season_length=2, alpha=0.5, beta=0.5, gamma=0.5
        )
