import numpy as np
import pytest

import foresight.models.theta as theta_mod
from foresight.models.theta import theta_forecast


def test_validated_theta_input_checks_horizon_and_min_train_size() -> None:
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        theta_mod._validated_theta_input(  # type: ignore[attr-defined]
            [1.0, 2.0],
            horizon=0,
            subject="theta_forecast",
            min_train_size=2,
        )

    with pytest.raises(ValueError, match="theta_forecast requires at least 2 training points"):
        theta_mod._validated_theta_input(  # type: ignore[attr-defined]
            [1.0],
            horizon=1,
            subject="theta_forecast",
            min_train_size=2,
        )


def test_validated_theta_grid_size_returns_normalized_int() -> None:
    grid_size = theta_mod._validated_theta_grid_size(7.0)  # type: ignore[attr-defined]

    assert grid_size == 7


def test_theta_alpha_1_uses_half_slope_drift():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = theta_forecast(y, horizon=3, alpha=1.0)
    assert pred.tolist() == [4.5, 5.0, 5.5]
