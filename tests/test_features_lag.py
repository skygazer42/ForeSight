import numpy as np
import pytest

import foresight.features.lag as lag_mod
from foresight.features.lag import make_lagged_xy, make_lagged_xy_multi
from foresight.features.tabular import build_column_lag_features


def test_make_lagged_xy_shapes_and_values():
    y = np.arange(5, dtype=float)  # [0,1,2,3,4]
    X, yt = make_lagged_xy(y, lags=2)
    assert X.shape == (3, 2)
    assert yt.shape == (3,)
    assert X.tolist() == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    assert yt.tolist() == [2.0, 3.0, 4.0]


def test_make_lagged_xy_supports_explicit_target_lags() -> None:
    y = np.arange(8, dtype=float)
    X, yt = make_lagged_xy(y, lags=(1, 3, 5))

    assert X.shape == (3, 3)
    assert yt.tolist() == [5.0, 6.0, 7.0]
    assert X.tolist() == [
        [0.0, 2.0, 4.0],
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 6.0],
    ]


def test_lagged_feature_matrix_builds_rows_for_target_windows() -> None:
    arr = np.arange(8, dtype=float)

    X = lag_mod._lagged_feature_matrix(  # type: ignore[attr-defined]
        arr,
        lag_steps=(1, 3, 5),
        t0=5,
        rows=3,
    )

    assert X.tolist() == [
        [4.0, 2.0, 0.0],
        [5.0, 3.0, 1.0],
        [6.0, 4.0, 2.0],
    ]


def test_validated_lag_steps_and_start_index_uses_max_lag_or_explicit_start() -> None:
    lag_steps, t0 = lag_mod._validated_lag_steps_and_start_index(  # type: ignore[attr-defined]
        lags=(1, 3, 5),
        start_t=4,
    )

    assert lag_steps == (5, 3, 1)
    assert t0 == 5


def test_build_column_lag_features_supports_historic_and_future_roles() -> None:
    x = np.arange(10.0, 15.0, dtype=float).reshape(-1, 1)
    t = np.array([3, 4], dtype=int)

    hist, hist_names = build_column_lag_features(
        x,
        t=t,
        lags=(2, 1),
        column_names=("promo",),
        prefix="historic_x",
    )
    futr, futr_names = build_column_lag_features(
        x,
        t=t,
        lags=(1, 0),
        column_names=("promo",),
        prefix="future_x",
        allow_zero=True,
    )

    assert hist_names == ["historic_x_promo_lag2", "historic_x_promo_lag1"]
    assert hist.tolist() == [[11.0, 12.0], [12.0, 13.0]]

    assert futr_names == ["future_x_promo_lag1", "future_x_promo_lag0"]
    assert futr.tolist() == [[12.0, 13.0], [13.0, 14.0]]


def test_make_lagged_xy_multi_shapes_and_values() -> None:
    y = np.arange(7, dtype=float)  # [0,1,2,3,4,5,6]

    X, Y, t_index = make_lagged_xy_multi(y, lags=2, horizon=3)

    assert X.shape == (3, 2)
    assert Y.shape == (3, 3)
    assert t_index.shape == (3,)
    assert X.tolist() == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    assert Y.tolist() == [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]
    assert t_index.tolist() == [2, 3, 4]


def test_make_lagged_xy_multi_rejects_not_enough_points() -> None:
    y = np.arange(4, dtype=float)

    with pytest.raises(ValueError):
        make_lagged_xy_multi(y, lags=2, horizon=3)
