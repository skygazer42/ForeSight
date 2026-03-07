import pytest

from foresight.models.registry import get_model_spec, make_forecaster, make_global_forecaster

GLOBAL_TORCH_MODELS = ["torch-tft-global", "torch-informer-global", "torch-autoformer-global"]

GLOBAL_ML_MODELS: dict[str, str] = {
    "xgb-gamma-step-lag-global": "xgb",
    "xgb-logistic-step-lag-global": "xgb",
    "xgb-msle-step-lag-global": "xgb",
    "xgb-mae-step-lag-global": "xgb",
    "xgb-huber-step-lag-global": "xgb",
    "xgb-poisson-step-lag-global": "xgb",
    "xgb-tweedie-step-lag-global": "xgb",
    "xgb-dart-step-lag-global": "xgb",
    "xgb-linear-step-lag-global": "xgb",
    "xgbrf-step-lag-global": "xgb",
    "adaboost-step-lag-global": "ml",
    "ard-step-lag-global": "ml",
    "bayesian-ridge-step-lag-global": "ml",
    "gamma-step-lag-global": "ml",
    "mlp-step-lag-global": "ml",
    "huber-step-lag-global": "ml",
    "omp-step-lag-global": "ml",
    "passive-aggressive-step-lag-global": "ml",
    "poisson-step-lag-global": "ml",
    "quantile-step-lag-global": "ml",
    "sgd-step-lag-global": "ml",
    "kernel-ridge-step-lag-global": "ml",
    "svr-step-lag-global": "ml",
    "linear-svr-step-lag-global": "ml",
    "lasso-step-lag-global": "ml",
    "elasticnet-step-lag-global": "ml",
    "knn-step-lag-global": "ml",
    "decision-tree-step-lag-global": "ml",
    "bagging-step-lag-global": "ml",
    "gbrt-step-lag-global": "ml",
    "ridge-step-lag-global": "ml",
    "rf-step-lag-global": "ml",
    "extra-trees-step-lag-global": "ml",
    "hgb-step-lag-global": "ml",
    "tweedie-step-lag-global": "ml",
    "xgb-step-lag-global": "xgb",
    "lgbm-step-lag-global": "lgbm",
    "catboost-step-lag-global": "catboost",
}


def test_global_models_are_marked_interface_global():
    for key in GLOBAL_TORCH_MODELS:
        spec = get_model_spec(key)
        assert spec.interface == "global"
        assert "torch" in spec.requires

    for key, extra in GLOBAL_ML_MODELS.items():
        spec = get_model_spec(key)
        assert spec.interface == "global"
        assert extra in spec.requires


def test_make_forecaster_rejects_global_models():
    for key in GLOBAL_TORCH_MODELS:
        with pytest.raises(ValueError):
            make_forecaster(key)

    for key in GLOBAL_ML_MODELS:
        with pytest.raises(ValueError):
            make_forecaster(key)


def test_make_global_forecaster_rejects_local_models():
    with pytest.raises(ValueError):
        make_global_forecaster("naive-last")
