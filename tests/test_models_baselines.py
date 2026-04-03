import numpy as np
import pytest

import foresight.models.baselines as baselines_mod
from foresight.models.baselines import (
    drift_forecast,
    mean_forecast,
    median_forecast,
    moving_average_forecast,
)
from foresight.models.registry import make_forecaster


def test_mean_forecast_repeats_mean():
    pred = mean_forecast([1.0, 2.0, 3.0], horizon=2)
    assert pred.shape == (2,)
    assert pred.tolist() == [2.0, 2.0]


def test_median_forecast_repeats_median():
    pred = median_forecast([1.0, 100.0, 3.0], horizon=3)
    assert pred.shape == (3,)
    assert pred.tolist() == [3.0, 3.0, 3.0]


def test_drift_forecast_linear_trend():
    # The input series implies a linear slope of 1.5 per step.
    y = np.array([10.0, 11.0, 13.0, 14.0, 16.0])
    pred = drift_forecast(y, horizon=3)
    assert pred.shape == (3,)
    assert pred.tolist() == [17.5, 19.0, 20.5]


def test_moving_average_forecast_uses_last_window():
    y = [1.0, 2.0, 3.0, 100.0]
    pred = moving_average_forecast(y, horizon=2, window=3)
    # The trailing three values average to 35.
    assert pred.tolist() == [35.0, 35.0]


def test_baselines_validate_horizon():
    with pytest.raises(ValueError):
        mean_forecast([1.0], horizon=0)


def test_constant_forecast_repeats_value_for_horizon() -> None:
    pred = baselines_mod._constant_forecast(3.5, horizon=4)  # type: ignore[attr-defined]

    assert pred.shape == (4,)
    assert pred.tolist() == [3.5, 3.5, 3.5, 3.5]


def test_validated_windowed_baseline_input_checks_window_and_history() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        baselines_mod._validated_windowed_baseline_input(  # type: ignore[attr-defined]
            [1.0, 2.0],
            horizon=1,
            window=0,
            subject="moving_average_forecast",
        )

    with pytest.raises(
        ValueError, match="moving_average_forecast requires at least 3 points, got 2"
    ):
        baselines_mod._validated_windowed_baseline_input(  # type: ignore[attr-defined]
            [1.0, 2.0],
            horizon=1,
            window=3,
            subject="moving_average_forecast",
        )


def _predict_registered_model(
    key: str, train: list[float] | np.ndarray, horizon: int, **params: object
) -> np.ndarray:
    try:
        forecaster = make_forecaster(key, **params)
        return forecaster(train, horizon)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"{key} should be available via the registry, got: {exc}")


def test_weighted_moving_average_forecast_emphasizes_recent_values():
    pred = _predict_registered_model(
        "weighted-moving-average",
        [1.0, 2.0, 10.0],
        horizon=2,
        window=3,
    )
    # Recent values carry larger weights, yielding a forecast of 35 / 6.
    assert pred.shape == (2,)
    assert pred.tolist() == [35.0 / 6.0, 35.0 / 6.0]


def test_moving_median_forecast_uses_last_window():
    pred = _predict_registered_model(
        "moving-median",
        [1.0, 2.0, 100.0, 4.0],
        horizon=3,
        window=3,
    )
    # The trailing three values have a median of 4.
    assert pred.shape == (3,)
    assert pred.tolist() == [4.0, 4.0, 4.0]


def test_seasonal_drift_forecast_extrapolates_last_two_seasons():
    pred = _predict_registered_model(
        "seasonal-drift",
        [10.0, 20.0, 12.0, 22.0],
        horizon=4,
        season_length=2,
    )
    # Last two seasons: [10,20] -> [12,22], per-position delta = [2,2]
    # Forecast continues the seasonal pattern with one more delta per future season.
    assert pred.shape == (4,)
    assert pred.tolist() == [14.0, 24.0, 16.0, 26.0]
