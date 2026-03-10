import numpy as np

from foresight.models.registry import make_forecaster


def test_seasonal_naive_auto_detects_period_and_repeats() -> None:
    y = np.asarray([1.0, 2.0, 3.0] * 10, dtype=float)
    f = make_forecaster(
        "seasonal-naive-auto",
        min_season_length=2,
        max_season_length=6,
        detrend=False,
    )

    yhat = f(y, 6)

    assert yhat.shape == (6,)
    assert yhat.tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_seasonal_naive_auto_falls_back_to_naive_last_when_too_short() -> None:
    y = np.asarray([1.0, 2.0, 3.0], dtype=float)
    f = make_forecaster("seasonal-naive-auto", min_season_length=2, max_season_length=12)

    yhat = f(y, 2)

    assert yhat.tolist() == [3.0, 3.0]

