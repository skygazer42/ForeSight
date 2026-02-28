import numpy as np
import pytest

from foresight.metrics import (
    interval_coverage,
    interval_score,
    mean_interval_width,
    msis,
    pinball_loss,
)


def test_pinball_loss_median():
    y = np.array([0.0, 1.0])
    yhat = np.array([0.0, 0.0])
    # u=[0,1] -> loss=[0,0.5]
    assert pinball_loss(y, yhat, q=0.5) == pytest.approx(0.25)


def test_pinball_loss_negative_residual():
    y = np.array([0.0])
    yhat = np.array([1.0])
    # u=-1 -> max(-0.5, 0.5)=0.5
    assert pinball_loss(y, yhat, q=0.5) == pytest.approx(0.5)


def test_interval_coverage_and_width():
    y = np.array([0.0, 1.0, 2.0])
    lo = np.array([0.0, 0.5, 2.0])
    hi = np.array([0.0, 1.5, 2.0])
    assert interval_coverage(y, lo, hi) == pytest.approx(1.0)
    assert mean_interval_width(lo, hi) == pytest.approx((0.0 + 1.0 + 0.0) / 3.0)


def test_interval_score_basic():
    y = np.array([0.0, 1.0])
    lo = np.array([-1.0, 0.0])
    hi = np.array([1.0, 2.0])
    # Both points inside -> score is just width = 2
    assert interval_score(y, lo, hi, alpha=0.1) == pytest.approx(2.0)


def test_msis_scales_interval_score():
    y_train = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([10.0, 11.0])
    lo = np.array([9.0, 10.0])
    hi = np.array([11.0, 12.0])
    # interval_score is 2.0; scale is mean(|diff1|)=1
    assert msis(y, lo, hi, y_train=y_train, seasonality=1, alpha=0.1) == pytest.approx(2.0)
