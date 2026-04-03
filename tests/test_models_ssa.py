import numpy as np
import pytest

import foresight.models.ssa as ssa_mod
from foresight.models.registry import make_forecaster


def test_validated_ssa_window_length_normalizes_value_within_bounds() -> None:
    length = ssa_mod._validated_ssa_window_length(10, window_length=4.0)  # type: ignore[attr-defined]

    assert length == 4


def test_validated_ssa_rank_caps_requested_rank_to_max_rank() -> None:
    rank = ssa_mod._validated_ssa_rank(7, max_rank=3)  # type: ignore[attr-defined]

    assert rank == 3


def test_ssa_recurrent_forecast_extends_history_from_coefficients() -> None:
    yhat = ssa_mod._ssa_recurrent_forecast(  # type: ignore[attr-defined]
        np.array([1.0, 1.0], dtype=float),
        coeffs=np.array([1.0, 1.0], dtype=float),
        horizon=4,
    )

    assert yhat.tolist() == pytest.approx([2.0, 3.0, 5.0, 8.0])


def test_ssa_constant_series_forecast_is_constant() -> None:
    y = np.ones(60, dtype=float)
    f = make_forecaster("ssa", window_length=20, rank=1)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.allclose(yhat, 1.0)


def test_ssa_smoke_returns_finite() -> None:
    y = np.sin(np.arange(120, dtype=float) / 6.0) + 0.01 * np.arange(120, dtype=float)
    f = make_forecaster("ssa", window_length=30, rank=5)
    yhat = f(y, 7)
    assert yhat.shape == (7,)
    assert np.all(np.isfinite(yhat))
