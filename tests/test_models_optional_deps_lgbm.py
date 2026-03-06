import importlib.util

import numpy as np
import pytest

from foresight.models.registry import get_model_spec, make_forecaster

LGBM_MODELS = [
    "lgbm-custom-dirrec-lag",
    "lgbm-custom-lag",
    "lgbm-custom-lag-recursive",
    "lgbm-custom-step-lag",
    "lgbm-dirrec-lag",
    "lgbm-lag",
    "lgbm-lag-recursive",
    "lgbm-step-lag",
]


def test_lgbm_models_are_registered_as_optional() -> None:
    for key in LGBM_MODELS:
        spec = get_model_spec(key)
        assert "lgbm" in spec.requires


def test_lgbm_models_raise_importerror_when_lightgbm_missing() -> None:
    if importlib.util.find_spec("lightgbm") is not None:
        pytest.skip("lightgbm installed; this test targets the missing-dep path")

    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    for key in LGBM_MODELS:
        f = make_forecaster(key)
        with pytest.raises(ImportError):
            f(y, 2)


def test_lgbm_models_smoke_when_installed() -> None:
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm not installed; smoke test requires it")

    y = 1.0 + np.sin(np.arange(160, dtype=float) / 3.0) + 0.1 * np.arange(160, dtype=float)
    horizon = 2

    base_params = {
        "lags": 12,
        "n_estimators": 10,
        "learning_rate": 0.1,
        "max_depth": 3,
        "num_leaves": 15,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 0,
        "n_jobs": 1,
    }

    cases = [
        ("lgbm-custom-lag", dict(base_params), y),
        ("lgbm-custom-lag-recursive", dict(base_params), y),
        (
            "lgbm-custom-step-lag",
            {
                **base_params,
                "step_scale": "one_based",
                "roll_windows": (3, 12),
                "roll_stats": ("mean", "std", "slope"),
                "diff_lags": (1, 6, 11),
            },
            y,
        ),
        ("lgbm-custom-dirrec-lag", dict(base_params), y),
        ("lgbm-lag", dict(base_params), y),
        ("lgbm-lag-recursive", dict(base_params), y),
        (
            "lgbm-step-lag",
            {
                **base_params,
                "step_scale": "one_based",
                "roll_windows": (3, 12),
                "roll_stats": ("mean", "std", "slope"),
                "diff_lags": (1, 6, 11),
            },
            y,
        ),
        ("lgbm-dirrec-lag", dict(base_params), y),
    ]

    for key, params, series in cases:
        f = make_forecaster(key, **params)
        yhat = f(series, horizon)
        assert yhat.shape == (horizon,)
        assert np.all(np.isfinite(yhat))


def test_lgbm_lag_models_validate_derived_feature_params_when_installed() -> None:
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm not installed; test targets derived-feature validation path")

    y = 1.0 + np.sin(np.arange(120, dtype=float) / 3.0) + 0.01 * np.arange(120, dtype=float)
    horizon = 2

    base_params = {
        "lags": 8,
        "n_estimators": 5,
        "learning_rate": 0.2,
        "max_depth": 2,
        "num_leaves": 15,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 0,
        "n_jobs": 1,
        "roll_windows": (9,),  # invalid: roll_window > lags
        "roll_stats": ("mean",),
    }

    keys = [
        "lgbm-lag",
        "lgbm-lag-recursive",
        "lgbm-custom-lag",
        "lgbm-custom-lag-recursive",
        "lgbm-dirrec-lag",
        "lgbm-custom-dirrec-lag",
    ]
    for key in keys:
        f = make_forecaster(key, **base_params)
        with pytest.raises(ValueError, match="exceeds lags"):
            _ = f(y, horizon)
