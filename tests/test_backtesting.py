import numpy as np

from foresight.backtesting import walk_forward
from foresight.models.naive import naive_last


def test_walk_forward_shapes():
    y = np.arange(20, dtype=float)
    out = walk_forward(y, horizon=3, step=3, min_train_size=10, forecaster=naive_last)
    assert out.y_true.shape == out.y_pred.shape
    assert out.y_true.ndim == 2  # (n_windows, horizon)
