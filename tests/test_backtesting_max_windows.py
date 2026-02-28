import numpy as np

from foresight.backtesting import walk_forward
from foresight.models.naive import naive_last


def test_walk_forward_honors_max_windows():
    y = np.arange(50, dtype=float)
    out = walk_forward(
        y, horizon=3, step=1, min_train_size=10, max_windows=5, forecaster=naive_last
    )
    assert out.y_true.shape[0] == 5
