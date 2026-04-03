import numpy as np
import pytest

import foresight.models.trend as trend_mod
from foresight.models.trend import poly_trend_forecast


def test_validated_poly_trend_inputs_normalizes_horizon_and_degree() -> None:
    x, horizon, degree = trend_mod._validated_poly_trend_inputs(  # type: ignore[attr-defined]
        [1.0, 2.0, 3.0],
        horizon=2.0,
        degree=1.0,
    )

    assert x.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert horizon == 2
    assert degree == 1


def test_poly_design_matrix_matches_increasing_vandermonde() -> None:
    X = trend_mod._poly_design_matrix(  # type: ignore[attr-defined]
        np.array([2.0, 3.0], dtype=float),
        degree=2,
    )

    np.testing.assert_allclose(
        X,
        np.array([[1.0, 2.0, 4.0], [1.0, 3.0, 9.0]], dtype=float),
    )


def test_poly_trend_degree_1_exact_for_linear():
    t = np.arange(20, dtype=float)
    y = 2.0 + 3.0 * t
    yhat = poly_trend_forecast(y, 5, degree=1)
    t_future = np.arange(20, 25, dtype=float)
    y_true = 2.0 + 3.0 * t_future
    assert np.max(np.abs(yhat - y_true)) < 1e-8
