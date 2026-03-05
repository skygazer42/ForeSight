import importlib.util

import numpy as np
import pytest

from foresight.models.regression import (
    adaboost_lag_direct_forecast,
    bagging_lag_direct_forecast,
    decision_tree_lag_direct_forecast,
    extra_trees_lag_direct_forecast,
    hgb_lag_direct_forecast,
    huber_lag_direct_forecast,
    kernel_ridge_lag_direct_forecast,
    linear_svr_lag_direct_forecast,
    mlp_lag_direct_forecast,
    quantile_lag_direct_forecast,
    ridge_lag_direct_forecast,
    sgd_lag_direct_forecast,
    svr_lag_direct_forecast,
)


def test_optional_sklearn_models_more_shapes_and_finite() -> None:
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn not installed")

    y = np.arange(160, dtype=float)
    horizon = 3
    lags = 12

    preds = [
        ridge_lag_direct_forecast(y, horizon, lags=lags, alpha=1.0),
        decision_tree_lag_direct_forecast(y, horizon, lags=lags, max_depth=5, random_state=0),
        extra_trees_lag_direct_forecast(
            y,
            horizon,
            lags=lags,
            n_estimators=30,
            max_depth=None,
            random_state=0,
        ),
        adaboost_lag_direct_forecast(
            y, horizon, lags=lags, n_estimators=30, learning_rate=0.1, random_state=0
        ),
        bagging_lag_direct_forecast(
            y, horizon, lags=lags, n_estimators=20, max_samples=0.8, random_state=0
        ),
        hgb_lag_direct_forecast(
            y, horizon, lags=lags, max_iter=50, learning_rate=0.05, max_depth=3, random_state=0
        ),
        svr_lag_direct_forecast(y, horizon, lags=lags, C=1.0, gamma="scale", epsilon=0.1),
        linear_svr_lag_direct_forecast(
            y, horizon, lags=lags, C=1.0, epsilon=0.0, max_iter=2000, random_state=0
        ),
        kernel_ridge_lag_direct_forecast(
            y, horizon, lags=lags, alpha=1.0, kernel="rbf", gamma=None
        ),
        mlp_lag_direct_forecast(
            y,
            horizon,
            lags=lags,
            hidden_layer_sizes=(32, 32),
            alpha=0.0001,
            max_iter=200,
            random_state=0,
            learning_rate_init=0.001,
        ),
        huber_lag_direct_forecast(y, horizon, lags=lags, epsilon=1.35, alpha=0.0001, max_iter=100),
        quantile_lag_direct_forecast(y, horizon, lags=lags, quantile=0.5, alpha=0.0),
        sgd_lag_direct_forecast(
            y, horizon, lags=lags, alpha=0.0001, penalty="l2", max_iter=2000, random_state=0
        ),
    ]

    for yhat in preds:
        assert yhat.shape == (horizon,)
        assert np.all(np.isfinite(yhat))
