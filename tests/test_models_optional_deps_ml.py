import importlib.util

import pytest

from foresight.models.registry import get_model_spec, make_forecaster

ML_MODELS = [
    # Existing scikit-learn models
    "ridge-lag",
    "rf-lag",
    "lasso-lag",
    "elasticnet-lag",
    "knn-lag",
    "gbrt-lag",
    # New scikit-learn models
    "ridge-lag-direct",
    "decision-tree-lag",
    "extra-trees-lag",
    "adaboost-lag",
    "bagging-lag",
    "hgb-lag",
    "svr-lag",
    "linear-svr-lag",
    "kernel-ridge-lag",
    "mlp-lag",
    "huber-lag",
    "quantile-lag",
    "sgd-lag",
]


def test_ml_models_are_registered_as_optional() -> None:
    for key in ML_MODELS:
        spec = get_model_spec(key)
        assert "ml" in spec.requires


def test_ml_models_raise_importerror_when_sklearn_missing() -> None:
    if importlib.util.find_spec("sklearn") is not None:
        pytest.skip("scikit-learn installed; this test targets the missing-dep path")

    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    for key in ML_MODELS:
        f = make_forecaster(key)
        with pytest.raises(ImportError):
            f(y, 2)
