import numpy as np
import pytest

import foresight.models.fourier as fourier_mod
from foresight.models.fourier import fourier_regression_forecast


def test_fourier_design_matrix_builds_intercept_trend_and_harmonics():
    t = np.array([0.0, 1.0], dtype=float)

    X = fourier_mod._fourier_design_matrix(  # type: ignore[attr-defined]
        t,
        periods=(4,),
        orders=(1,),
        include_trend=True,
    )

    assert X.shape == (2, 4)
    assert np.allclose(X[:, 0], [1.0, 1.0])
    assert np.allclose(X[:, 1], [0.0, 1.0])
    assert np.allclose(X[:, 2], [0.0, 1.0])
    assert np.allclose(X[:, 3], [1.0, 0.0], atol=1e-12)


def test_linear_design_forecast_projects_future_rows_from_lstsq_solution():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=float)
    y = np.array([1.0, 3.0, 5.0], dtype=float)
    X_future = np.array([[1.0, 3.0], [1.0, 4.0]], dtype=float)

    yhat = fourier_mod._linear_design_forecast(  # type: ignore[attr-defined]
        X,
        y,
        X_future,
    )

    assert yhat.tolist() == pytest.approx([7.0, 9.0])


def test_fourier_regression_repeats_sine_wave():
    period = 12
    t = np.arange(36, dtype=float)
    y = np.sin(2.0 * np.pi * t / float(period))

    horizon = 12
    yhat = fourier_regression_forecast(y, horizon, period=period, order=1, include_trend=False)
    assert yhat.shape == (horizon,)

    t_future = np.arange(36, 36 + horizon, dtype=float)
    y_true = np.sin(2.0 * np.pi * t_future / float(period))
    assert np.max(np.abs(yhat - y_true)) < 1e-6
