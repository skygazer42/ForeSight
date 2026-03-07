import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import get_model_spec


def test_var_model_is_registered_as_multivariate_stats_optional():
    spec = get_model_spec("var")
    assert spec.interface == "multivariate"
    assert "stats" in spec.requires


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
