import numpy as np
import pytest

import foresight.models.ar as ar_mod
from foresight.models.ar import ar_ols_forecast


def _gen_ar2(n: int) -> np.ndarray:
    y = np.zeros(n, dtype=float)
    y[0] = 1.0
    y[1] = 2.0
    for t in range(2, n):
        y[t] = 0.3 * y[t - 1] + 0.2 * y[t - 2]
    return y


def test_validated_ar_ols_input_normalizes_horizon_and_order() -> None:
    x, horizon, p = ar_mod._validated_ar_ols_input(  # type: ignore[attr-defined]
        [1.0, 2.0, 3.0],
        horizon=2.0,
        p=1.0,
        subject="ar_ols_forecast",
    )

    assert x.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert horizon == 2
    assert p == 1


def test_ar_ols_design_matrix_builds_intercept_and_lag_columns() -> None:
    X, y = ar_mod._ar_ols_design_matrix(  # type: ignore[attr-defined]
        np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        p=2,
    )

    np.testing.assert_allclose(
        X,
        np.array(
            [
                [1.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(y, np.array([3.0, 4.0], dtype=float))


def test_ar_ols_forecast_matches_noiseless_ar2():
    y = _gen_ar2(60)
    train = y[:50]
    true_future = y[50:53]
    pred = ar_ols_forecast(train, horizon=3, p=2)
    assert np.allclose(pred, true_future, atol=1e-9)
