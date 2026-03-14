import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import get_model_spec, make_forecaster, make_global_forecaster

XGB_MODELS = [
    "xgb-custom-lag",
    "xgb-custom-lag-recursive",
    "xgb-custom-dirrec-lag",
    "xgb-custom-mimo-lag",
    "xgb-custom-step-lag",
    "xgb-dirrec-lag",
    "xgb-dart-lag",
    "xgb-dart-lag-recursive",
    "xgb-gamma-lag",
    "xgb-gamma-lag-recursive",
    "xgb-huber-lag",
    "xgb-huber-lag-recursive",
    "xgb-lag",
    "xgb-lag-recursive",
    "xgb-linear-lag",
    "xgb-linear-lag-recursive",
    "xgb-logistic-lag",
    "xgb-logistic-lag-recursive",
    "xgb-mae-lag",
    "xgb-mae-lag-recursive",
    "xgb-mimo-lag",
    "xgb-msle-lag",
    "xgb-msle-lag-recursive",
    "xgb-poisson-lag",
    "xgb-poisson-lag-recursive",
    "xgb-quantile-lag",
    "xgb-quantile-lag-recursive",
    "xgb-step-lag",
    "xgb-tweedie-lag",
    "xgb-tweedie-lag-recursive",
    "xgbrf-lag",
    "xgbrf-lag-recursive",
]

GLOBAL_XGB_MODELS = [
    "xgb-step-lag-global",
    "xgb-gamma-step-lag-global",
    "xgb-logistic-step-lag-global",
    "xgb-msle-step-lag-global",
    "xgb-mae-step-lag-global",
    "xgb-huber-step-lag-global",
    "xgb-poisson-step-lag-global",
    "xgb-tweedie-step-lag-global",
    "xgb-dart-step-lag-global",
    "xgb-linear-step-lag-global",
    "xgbrf-step-lag-global",
]


def test_xgb_models_are_registered_as_optional() -> None:
    for key in XGB_MODELS:
        spec = get_model_spec(key)
        assert "xgb" in spec.requires

    for key in GLOBAL_XGB_MODELS:
        spec = get_model_spec(key)
        assert "xgb" in spec.requires
        assert spec.interface == "global"


def test_xgb_models_raise_importerror_when_xgboost_missing() -> None:
    if importlib.util.find_spec("xgboost") is not None:
        pytest.skip("xgboost installed; this test targets the missing-dep path")

    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    for key in XGB_MODELS:
        f = make_forecaster(key)
        with pytest.raises(ImportError):
            f(y, 2)


def test_global_xgb_models_raise_importerror_when_xgboost_missing() -> None:
    if importlib.util.find_spec("xgboost") is not None:
        pytest.skip("xgboost installed; this test targets the missing-dep path")

    ds = pd.date_range("2020-01-01", periods=12, freq="D")
    df = pd.DataFrame(
        {
            "unique_id": ["s0"] * len(ds),
            "ds": ds,
            "y": [float(i) for i in range(len(ds))],
        }
    )

    for key in GLOBAL_XGB_MODELS:
        with pytest.raises(ImportError):
            f = make_global_forecaster(key)
            f(df, ds[8], 2)


def test_xgb_models_smoke_when_installed() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed; smoke test requires it")

    y_pos = 1.0 + np.sin(np.arange(160, dtype=float) / 3.0) + 0.1 * np.arange(160, dtype=float)
    y_01 = 0.5 + 0.4 * np.sin(np.arange(200, dtype=float) / 5.0)
    horizon = 2

    cases = [
        (
            "xgb-custom-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "booster": "gbtree",
                "objective": "reg:squarederror",
            },
            y_pos,
        ),
        (
            "xgb-custom-lag-recursive",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "booster": "gbtree",
                "objective": "reg:squarederror",
            },
            y_pos,
        ),
        (
            "xgb-custom-step-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "booster": "gbtree",
                "objective": "reg:squarederror",
                "step_scale": "one_based",
                "roll_windows": (3, 12),
                "roll_stats": ("mean", "std", "slope"),
                "diff_lags": (1, 6, 11),
            },
            y_pos,
        ),
        (
            "xgb-custom-dirrec-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "booster": "gbtree",
                "objective": "reg:squarederror",
            },
            y_pos,
        ),
        (
            "xgb-custom-mimo-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "booster": "gbtree",
                "objective": "reg:squarederror",
                "multi_strategy": "multi_output_tree",
            },
            y_pos,
        ),
        (
            "xgb-step-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "roll_windows": (3, 12),
                "roll_stats": ("mean", "std", "slope"),
                "diff_lags": (1, 6, 11),
            },
            y_pos,
        ),
        (
            "xgb-dirrec-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        ("xgb-mimo-lag", {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3}, y_pos),
        ("xgb-lag", {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3}, y_pos),
        (
            "xgb-lag-recursive",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-msle-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-msle-lag-recursive",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-logistic-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_01,
        ),
        (
            "xgb-logistic-lag-recursive",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_01,
        ),
        (
            "xgb-dart-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-dart-lag-recursive",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        ("xgbrf-lag", {"lags": 12, "n_estimators": 10, "max_depth": 3}, y_pos),
        ("xgbrf-lag-recursive", {"lags": 12, "n_estimators": 10, "max_depth": 3}, y_pos),
        ("xgb-linear-lag", {"lags": 12, "n_estimators": 50, "learning_rate": 0.1}, y_pos),
        ("xgb-linear-lag-recursive", {"lags": 12, "n_estimators": 50, "learning_rate": 0.1}, y_pos),
        (
            "xgb-mae-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-mae-lag-recursive",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-huber-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "huber_slope": 1.0,
            },
            y_pos,
        ),
        (
            "xgb-huber-lag-recursive",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "huber_slope": 1.0,
            },
            y_pos,
        ),
        (
            "xgb-quantile-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "quantile_alpha": 0.5,
            },
            y_pos,
        ),
        (
            "xgb-quantile-lag-recursive",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "quantile_alpha": 0.5,
            },
            y_pos,
        ),
        (
            "xgb-poisson-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-poisson-lag-recursive",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-gamma-lag",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-gamma-lag-recursive",
            {"lags": 12, "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            y_pos,
        ),
        (
            "xgb-tweedie-lag",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "tweedie_variance_power": 1.5,
            },
            y_pos,
        ),
        (
            "xgb-tweedie-lag-recursive",
            {
                "lags": 12,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "tweedie_variance_power": 1.5,
            },
            y_pos,
        ),
    ]

    for key, params, y in cases:
        f = make_forecaster(key, **params)
        yhat = f(y, horizon)
        assert yhat.shape == (horizon,)
        assert np.all(np.isfinite(yhat))


def test_xgb_lag_models_validate_derived_feature_params_when_installed() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed; test targets derived-feature validation path")

    y = 1.0 + np.sin(np.arange(120, dtype=float) / 3.0) + 0.01 * np.arange(120, dtype=float)
    horizon = 2

    # Use tiny models because pre-feature-support this will not raise and would otherwise train.
    base_params = {
        "lags": 8,
        "n_estimators": 2,
        "learning_rate": 0.2,
        "max_depth": 2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 0,
        "n_jobs": 1,
        "tree_method": "hist",
        "roll_windows": (9,),  # invalid: roll_window > lags
        "roll_stats": ("mean",),
    }

    keys = [
        "xgb-lag",
        "xgb-lag-recursive",
        "xgb-custom-lag",
        "xgb-custom-lag-recursive",
        "xgb-dirrec-lag",
        "xgb-mimo-lag",
    ]
    for key in keys:
        f = make_forecaster(key, **base_params)
        with pytest.raises(ValueError, match="exceeds lags"):
            _ = f(y, horizon)
