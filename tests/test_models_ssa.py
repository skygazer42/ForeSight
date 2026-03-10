import numpy as np

from foresight.models.registry import make_forecaster


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

