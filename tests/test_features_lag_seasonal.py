import numpy as np
import pytest

from foresight.features.lag import build_seasonal_lag_features


def test_build_seasonal_lag_features_values() -> None:
    y = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
    # Use the next one-step-ahead index relative to the five-point history.
    X, names = build_seasonal_lag_features(y, t=[5], seasonal_lags=(2,), seasonal_diff_lags=(2,))
    assert X.shape == (1, 2)
    assert names == ["season_lag_2", "season_diff_2"]
    assert X[0, 0] == pytest.approx(40.0)  # The seasonal lag picks the prior same-phase value.
    assert X[0, 1] == pytest.approx(20.0)  # The seasonal diff matches the expected delta.


def test_build_seasonal_lag_features_rejects_invalid_specs() -> None:
    y = np.asarray([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError):
        build_seasonal_lag_features(y, t=[3], seasonal_lags=(0,))

    with pytest.raises(ValueError):
        build_seasonal_lag_features(y, t=[0], seasonal_diff_lags=(1,))

    with pytest.raises(ValueError):
        build_seasonal_lag_features(y, t=[2], seasonal_lags=(5,))
