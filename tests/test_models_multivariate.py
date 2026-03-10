import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import get_model_spec


def test_var_model_is_registered_as_multivariate_stats_optional():
    spec = get_model_spec("var")
    assert spec.interface == "multivariate"
    assert "stats" in spec.requires


def test_torch_stid_model_is_registered_as_multivariate_torch_optional():
    spec = get_model_spec("torch-stid-multivariate")
    assert spec.interface == "multivariate"
    assert "torch" in spec.requires


def test_torch_graph_models_are_registered_as_multivariate_torch_optional():
    for key in ("torch-stgcn-multivariate", "torch-graphwavenet-multivariate"):
        spec = get_model_spec(key)
        assert spec.interface == "multivariate"
        assert "torch" in spec.requires


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None,
    reason="statsmodels not installed",
)
def test_var_forecast_on_wide_dataframe_returns_horizon_by_target_matrix():
    from foresight.models.multivariate import var_forecast

    train = pd.DataFrame(
        {
            "y_a": np.arange(20.0),
            "y_b": np.arange(20.0) * 0.5 + 1.0,
        }
    )

    fc = var_forecast(train, horizon=3, maxlags=1)

    assert isinstance(fc, np.ndarray)
    assert fc.shape == (3, 2)
    np.testing.assert_allclose(
        fc,
        np.asarray(
            [
                [20.0, 11.0],
                [21.0, 11.5],
                [22.0, 12.0],
            ]
        ),
        atol=1e-6,
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_torch_stid_forecast_on_wide_dataframe_returns_horizon_by_target_matrix():
    from foresight.models.registry import make_multivariate_forecaster

    t = np.arange(64.0)
    train = pd.DataFrame(
        {
            "y_a": np.sin(t / 5.0) + 0.01 * t,
            "y_b": np.cos(t / 7.0) + 0.02 * t,
            "y_c": np.sin(t / 9.0 + 0.5) + 0.015 * t,
        }
    )

    f = make_multivariate_forecaster(
        "torch-stid-multivariate",
        lags=24,
        d_model=16,
        num_blocks=2,
        epochs=2,
        batch_size=16,
        device="cpu",
        seed=0,
        patience=2,
    )
    fc = f(train, horizon=3)

    assert isinstance(fc, np.ndarray)
    assert fc.shape == (3, 3)
    assert np.all(np.isfinite(fc))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize("key", ("torch-stgcn-multivariate", "torch-graphwavenet-multivariate"))
def test_torch_graph_forecasters_smoke_on_wide_dataframe(key: str):
    from foresight.models.registry import make_multivariate_forecaster

    t = np.arange(96.0)
    train = pd.DataFrame(
        {
            "n0": np.sin(t / 6.0) + 0.01 * t,
            "n1": np.cos(t / 7.0) + 0.02 * t,
            "n2": np.sin(t / 9.0 + 0.5) + 0.015 * t,
            "n3": np.cos(t / 8.0 + 0.2) + 0.005 * t,
        }
    )

    f = make_multivariate_forecaster(
        key,
        lags=24,
        d_model=16,
        num_blocks=2,
        kernel_size=3,
        adj="ring",
        epochs=2,
        batch_size=16,
        device="cpu",
        seed=0,
        patience=2,
    )
    fc = f(train, horizon=3)

    assert isinstance(fc, np.ndarray)
    assert fc.shape == (3, 4)
    assert np.all(np.isfinite(fc))
