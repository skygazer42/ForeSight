import numpy as np

from foresight.features.lag import make_lagged_xy


def test_make_lagged_xy_shapes_and_values():
    y = np.arange(5, dtype=float)  # [0,1,2,3,4]
    X, yt = make_lagged_xy(y, lags=2)
    assert X.shape == (3, 2)
    assert yt.shape == (3,)
    assert X.tolist() == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    assert yt.tolist() == [2.0, 3.0, 4.0]
