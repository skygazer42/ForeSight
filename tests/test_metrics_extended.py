import numpy as np

from foresight.metrics import mase, mse, rmsse, wape


def test_mse_basic():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 4.0])
    assert mse(y_true, y_pred) == 2.0


def test_wape_basic():
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([2.0, 0.0])
    assert wape(y_true, y_pred) == 1.0


def test_mase_basic():
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    y_true = np.array([5.0, 6.0])
    y_pred = np.array([6.0, 6.0])
    assert mase(y_true, y_pred, y_train=y_train, seasonality=1) == 0.5


def test_rmsse_basic():
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    y_true = np.array([5.0, 6.0])
    y_pred = np.array([6.0, 6.0])
    # mse(err) = (1^2 + 0^2)/2 = 0.5, scale mse = mean((diff)^2) = 1
    assert rmsse(y_true, y_pred, y_train=y_train, seasonality=1) == (0.5) ** 0.5
