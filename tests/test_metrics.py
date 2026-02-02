import numpy as np

from foresight.metrics import mae, mape, rmse, smape


def test_metrics_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 5.0])
    assert mae(y_true, y_pred) == 1.0
    assert round(rmse(y_true, y_pred), 6) == round(((0**2 + 1**2 + 2**2) / 3) ** 0.5, 6)
    assert mape(y_true, y_pred) > 0
    assert smape(y_true, y_pred) > 0

