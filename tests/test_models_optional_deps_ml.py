import importlib.util

import pandas as pd
import pytest

from foresight.models.registry import get_model_spec, make_forecaster, make_global_forecaster

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

GLOBAL_ML_MODELS = [
    "adaboost-step-lag-global",
    "mlp-step-lag-global",
    "huber-step-lag-global",
    "quantile-step-lag-global",
    "sgd-step-lag-global",
    "kernel-ridge-step-lag-global",
    "svr-step-lag-global",
    "linear-svr-step-lag-global",
    "lasso-step-lag-global",
    "elasticnet-step-lag-global",
    "knn-step-lag-global",
    "decision-tree-step-lag-global",
    "bagging-step-lag-global",
    "gbrt-step-lag-global",
    "ridge-step-lag-global",
    "rf-step-lag-global",
    "extra-trees-step-lag-global",
]


def test_ml_models_are_registered_as_optional() -> None:
    for key in ML_MODELS:
        spec = get_model_spec(key)
        assert "ml" in spec.requires

    for key in GLOBAL_ML_MODELS:
        spec = get_model_spec(key)
        assert "ml" in spec.requires
        assert spec.interface == "global"


def test_ml_models_raise_importerror_when_sklearn_missing() -> None:
    if importlib.util.find_spec("sklearn") is not None:
        pytest.skip("scikit-learn installed; this test targets the missing-dep path")

    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    for key in ML_MODELS:
        f = make_forecaster(key)
        with pytest.raises(ImportError):
            f(y, 2)


def test_global_ml_models_raise_importerror_when_sklearn_missing() -> None:
    if importlib.util.find_spec("sklearn") is not None:
        pytest.skip("scikit-learn installed; this test targets the missing-dep path")

    ds = pd.date_range("2020-01-01", periods=12, freq="D")
    df = pd.DataFrame(
        {
            "unique_id": ["s0"] * len(ds),
            "ds": ds,
            "y": [float(i) for i in range(len(ds))],
        }
    )

    for key in GLOBAL_ML_MODELS:
        f = make_global_forecaster(key)
        with pytest.raises(ImportError):
            f(df, ds[8], 2)
