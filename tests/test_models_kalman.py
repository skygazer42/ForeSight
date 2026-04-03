import numpy as np
import pytest

import foresight.models.kalman as kalman_mod
from foresight.models.kalman import (
    kalman_local_level_forecast,
    kalman_local_linear_trend_forecast,
)


def test_validated_kalman_input_checks_horizon_and_min_train_size() -> None:
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        kalman_mod._validated_kalman_input(  # type: ignore[attr-defined]
            [1.0],
            horizon=0,
            subject="kalman_local_level_forecast",
            min_train_size=1,
        )

    with pytest.raises(
        ValueError,
        match="kalman_local_linear_trend_forecast requires at least 2 training points",
    ):
        kalman_mod._validated_kalman_input(  # type: ignore[attr-defined]
            [1.0],
            horizon=1,
            subject="kalman_local_linear_trend_forecast",
            min_train_size=2,
        )


def test_kalman_base_variance_falls_back_to_small_positive_floor() -> None:
    base = kalman_mod._kalman_base_variance(np.ones(4, dtype=float))  # type: ignore[attr-defined]

    assert base == pytest.approx(1e-6)


def test_kalman_local_level_output_shape():
    y = np.arange(1, 51, dtype=float)
    yhat = kalman_local_level_forecast(y, 4)
    assert yhat.shape == (4,)
    assert np.all(np.isfinite(yhat))


def test_kalman_local_linear_trend_output_shape():
    y = np.arange(1, 51, dtype=float)
    yhat = kalman_local_linear_trend_forecast(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


def test_kalman_local_level_constant_series_is_constant():
    y = np.ones(40, dtype=float) * 3.0
    yhat = kalman_local_level_forecast(y, 3)
    assert np.allclose(yhat, 3.0)
