import numpy as np
import pytest

import foresight.models.intermittent as intermittent_mod
from foresight.models.intermittent import croston_classic_forecast, tsb_forecast


def test_first_nonzero_index_returns_first_positive_position() -> None:
    idx = intermittent_mod._first_nonzero_index(np.array([0.0, 0.0, 5.0, 0.0]))  # type: ignore[attr-defined]

    assert idx == 2


def test_croston_initial_state_uses_first_nonzero_demand_and_interval() -> None:
    state = intermittent_mod._croston_initial_state(  # type: ignore[attr-defined]
        np.array([0.0, 0.0, 5.0, 0.0], dtype=float)
    )

    assert state == (2, 5.0, 3.0, 1.0)


def test_validated_intermittent_input_checks_horizon_and_nonempty_train() -> None:
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        intermittent_mod._validated_intermittent_input(  # type: ignore[attr-defined]
            [1.0],
            horizon=0,
            subject="croston_classic_forecast",
        )

    with pytest.raises(ValueError, match="croston_classic_forecast requires at least 1 training point"):
        intermittent_mod._validated_intermittent_input(  # type: ignore[attr-defined]
            [],
            horizon=1,
            subject="croston_classic_forecast",
        )


def test_zero_forecast_returns_zero_vector_for_horizon() -> None:
    yhat = intermittent_mod._zero_forecast(3)  # type: ignore[attr-defined]

    assert yhat.shape == (3,)
    assert np.allclose(yhat, 0.0)


def test_validated_alpha_beta_returns_both_parameters_in_range() -> None:
    a, b = intermittent_mod._validated_alpha_beta(0.2, 0.3)  # type: ignore[attr-defined]

    assert a == pytest.approx(0.2)
    assert b == pytest.approx(0.3)


def test_croston_classic_all_zero_returns_zero():
    yhat = croston_classic_forecast([0, 0, 0, 0], 3, alpha=0.1)
    assert yhat.shape == (3,)
    assert np.allclose(yhat, 0.0)


def test_croston_classic_positive_forecast_for_intermittent():
    y = [0, 0, 5, 0, 0, 0, 3, 0, 0]
    yhat = croston_classic_forecast(y, 4, alpha=0.2)
    assert yhat.shape == (4,)
    assert np.all(np.isfinite(yhat))
    assert float(yhat[0]) >= 0.0


def test_tsb_all_zero_returns_zero():
    yhat = tsb_forecast([0, 0, 0, 0], 2, alpha=0.1, beta=0.1)
    assert yhat.shape == (2,)
    assert np.allclose(yhat, 0.0)


def test_tsb_positive_forecast_for_intermittent():
    y = [0, 0, 5, 0, 0, 0, 3, 0, 0]
    yhat = tsb_forecast(y, 3, alpha=0.2, beta=0.2)
    assert yhat.shape == (3,)
    assert np.all(np.isfinite(yhat))
    assert float(yhat[0]) >= 0.0
