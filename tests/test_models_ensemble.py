import numpy as np

from foresight.models.registry import make_forecaster


def test_ensemble_mean_matches_average():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    f = make_forecaster("ensemble-mean", members=("mean", "naive-last"))
    yhat = f(y, 2)
    assert yhat.shape == (2,)
    # mean -> 2.5; naive-last -> 4.0; average -> 3.25
    assert np.allclose(yhat, np.array([3.25, 3.25]))


def test_ensemble_median_matches_median():
    y = np.array([1.0, 2.0, 3.0, 100.0])
    f = make_forecaster("ensemble-median", members=("mean", "naive-last", "median"))
    yhat = f(y, 1)
    assert yhat.shape == (1,)
    # mean=26.5, naive-last=100, median=2.5 -> median = 26.5
    assert np.allclose(yhat, np.array([26.5]))
