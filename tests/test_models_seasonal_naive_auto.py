import numpy as np
import pytest

import foresight.models.naive as naive_mod
from foresight.models.registry import make_forecaster


def test_validated_naive_input_checks_nonempty_train_and_horizon() -> None:
    with pytest.raises(ValueError, match="naive_last requires at least 1 training point"):
        naive_mod._validated_naive_input(  # type: ignore[attr-defined]
            [],
            horizon=1,
            subject="naive_last",
        )

    with pytest.raises(ValueError, match="horizon must be >= 1"):
        naive_mod._validated_naive_input(  # type: ignore[attr-defined]
            [1.0],
            horizon=0,
            subject="naive_last",
        )


def test_best_seasonal_naive_lag_prefers_smallest_lag_on_tied_correlation() -> None:
    result = naive_mod._best_seasonal_naive_lag(  # type: ignore[attr-defined]
        np.ones(8, dtype=float),
        min_season_length=2,
        max_season_length=4,
    )

    assert result == (2, pytest.approx(0.0))


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
