import numpy as np
import pytest

from foresight.backtesting import walk_forward


def _first_value_forecaster(train: np.ndarray, horizon: int) -> np.ndarray:
    first = float(np.asarray(train, dtype=float)[0])
    return np.full((int(horizon),), first, dtype=float)


def test_walk_forward_max_train_size_changes_train_window():
    y = np.arange(10, dtype=float)

    expanding = walk_forward(
        y,
        horizon=2,
        step=2,
        min_train_size=4,
        forecaster=_first_value_forecaster,
        max_train_size=None,
    )
    rolling = walk_forward(
        y,
        horizon=2,
        step=2,
        min_train_size=4,
        forecaster=_first_value_forecaster,
        max_train_size=4,
    )

    assert expanding.y_pred.shape == rolling.y_pred.shape
    # Expanding window always starts at 0 -> first value is 0 for all windows
    assert np.allclose(expanding.y_pred, 0.0)
    # Rolling window starts shift -> first values increase with each window
    assert rolling.y_pred[0, 0] == pytest.approx(0.0)
    assert rolling.y_pred[1, 0] == pytest.approx(2.0)
    assert rolling.y_pred[2, 0] == pytest.approx(4.0)


def test_walk_forward_max_train_size_invalid_raises():
    with pytest.raises(ValueError):
        walk_forward(
            [1, 2, 3, 4, 5],
            horizon=1,
            step=1,
            min_train_size=2,
            max_train_size=0,
            forecaster=lambda t, h: np.zeros(h),
        )
