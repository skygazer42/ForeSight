import numpy as np
import pytest

import foresight.models.spectral as spectral_mod
from foresight.models.spectral import fft_topk_forecast


def test_validated_fft_topk_input_normalizes_horizon_and_top_k() -> None:
    x, horizon, top_k = spectral_mod._validated_fft_topk_input(  # type: ignore[attr-defined]
        [1.0, 2.0, 3.0, 4.0],
        horizon=2.0,
        top_k=3.0,
    )

    assert x.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert horizon == 2
    assert top_k == 3


def test_harmonic_regression_design_matrix_stacks_bias_sin_and_cos_columns() -> None:
    X = spectral_mod._harmonic_regression_design_matrix(  # type: ignore[attr-defined]
        np.array([0.0, 0.25], dtype=float),
        np.array([2.0 * np.pi], dtype=float),
    )

    np.testing.assert_allclose(
        X,
        np.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        atol=1e-12,
    )


def test_fft_topk_output_shape():
    rng = np.random.default_rng(0)
    t = np.arange(80, dtype=float)
    y = 0.05 * t + np.sin(2.0 * np.pi * t / 12.0) + 0.1 * rng.standard_normal(80)
    yhat = fft_topk_forecast(y, 6, top_k=3, include_trend=True)
    assert yhat.shape == (6,)
    assert np.all(np.isfinite(yhat))


def test_fft_topk_constant_series_is_constant():
    y = np.ones(30, dtype=float) * 7.0
    yhat = fft_topk_forecast(y, 4, top_k=3, include_trend=True)
    assert yhat.shape == (4,)
    assert np.allclose(yhat, 7.0)
