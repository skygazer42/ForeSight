import importlib.util

import numpy as np
import pytest

from foresight.models.regression import (
    elasticnet_lag_direct_forecast,
    gbrt_lag_direct_forecast,
    knn_lag_direct_forecast,
    lasso_lag_direct_forecast,
)


def test_optional_sklearn_models_shapes():
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn not installed")

    y = np.arange(120, dtype=float)
    horizon = 3
    lags = 12

    yhat_lasso = lasso_lag_direct_forecast(y, horizon, lags=lags, alpha=0.001, max_iter=2000)
    assert yhat_lasso.shape == (horizon,)

    yhat_en = elasticnet_lag_direct_forecast(
        y, horizon, lags=lags, alpha=0.001, l1_ratio=0.5, max_iter=2000
    )
    assert yhat_en.shape == (horizon,)

    yhat_knn = knn_lag_direct_forecast(y, horizon, lags=lags, n_neighbors=5, weights="distance")
    assert yhat_knn.shape == (horizon,)

    yhat_gbrt = gbrt_lag_direct_forecast(
        y, horizon, lags=lags, n_estimators=50, learning_rate=0.05, max_depth=3, random_state=0
    )
    assert yhat_gbrt.shape == (horizon,)

    assert np.all(np.isfinite(yhat_lasso))
    assert np.all(np.isfinite(yhat_en))
    assert np.all(np.isfinite(yhat_knn))
    assert np.all(np.isfinite(yhat_gbrt))
