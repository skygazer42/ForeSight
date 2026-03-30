import numpy as np

from foresight.backtesting import walk_forward
from foresight.models.naive import naive_last


def test_walk_forward_shapes():
    y = np.arange(20, dtype=float)
    out = walk_forward(y, horizon=3, step=3, min_train_size=10, forecaster=naive_last)
    assert out.y_true.shape == out.y_pred.shape
    assert out.y_true.ndim == 2  # (n_windows, horizon)


def test_walk_forward_respects_max_windows_from_front() -> None:
    y = np.arange(20, dtype=float)

    out = walk_forward(
        y,
        horizon=3,
        step=3,
        min_train_size=10,
        max_windows=2,
        forecaster=naive_last,
    )

    assert out.y_true.shape == (2, 3)
    assert out.y_pred.shape == (2, 3)
    assert out.train_ends.tolist() == [10, 13]
