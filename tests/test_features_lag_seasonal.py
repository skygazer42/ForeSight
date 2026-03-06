import numpy as np
import pytest

from foresight.features.lag import build_seasonal_lag_features


def test_build_seasonal_lag_features_values() -> None:
    y = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
    # t=5 is the "next" index (one-step-ahead) relative to y of length 5
    X, names = build_seasonal_lag_features(y, t=[5], seasonal_lags=(2,), seasonal_diff_lags=(2,))
    assert X.shape == (1, 2)
    assert names == ["season_lag_2", "season_diff_2"]
    assert X[0, 0] == pytest.approx(40.0)  # y[5-2] = y[3]
    assert X[0, 1] == pytest.approx(20.0)  # y[4] - y[2] = 50 - 30


def test_build_seasonal_lag_features_rejects_invalid_specs() -> None:
    y = np.asarray([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError):
        build_seasonal_lag_features(y, t=[3], seasonal_lags=(0,))

    with pytest.raises(ValueError):
        build_seasonal_lag_features(y, t=[0], seasonal_diff_lags=(1,))

    with pytest.raises(ValueError):
        build_seasonal_lag_features(y, t=[2], seasonal_lags=(5,))

