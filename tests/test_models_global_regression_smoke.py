import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.cv import cross_validation_predictions_long_df
from foresight.eval_predictions import evaluate_quantile_predictions


def _small_panel_long_df() -> pd.DataFrame:
    ds = pd.date_range("2020-01-01", periods=40, freq="D")
    rows = []
    for uid, bias in [("s0", 0.0), ("s1", 1.0)]:
        for i, d in enumerate(ds):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": float(bias + 0.1 * i),
                    # Simple exogenous feature to validate x_cols plumbing.
                    "promo": float(i % 7 == 0),
                }
            )
    return pd.DataFrame(rows)


def _small_panel_positive_long_df() -> pd.DataFrame:
    ds = pd.date_range("2020-01-01", periods=40, freq="D")
    rows = []
    for uid, bias in [("s0", 1.0), ("s1", 2.0)]:
        for i, d in enumerate(ds):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": float(bias + 0.05 * i),
                    "promo": float(i % 7 == 0),
                }
            )
    return pd.DataFrame(rows)


def _small_panel_unit_long_df() -> pd.DataFrame:
    ds = pd.date_range("2020-01-01", periods=40, freq="D")
    rows = []
    for uid, phase in [("s0", 0.0), ("s1", 0.6)]:
        for i, d in enumerate(ds):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": float(0.5 + 0.3 * np.sin(i / 5.0 + phase)),
                    "promo": float(i % 7 == 0),
                }
            )
    return pd.DataFrame(rows)


def _assert_global_point_smoke(
    model: str,
    *,
    model_params: dict[str, object],
    long_df: pd.DataFrame | None = None,
) -> None:
    pred = cross_validation_predictions_long_df(
        model=model,
        long_df=_small_panel_long_df() if long_df is None else long_df,
        horizon=2,
        step_size=2,
        min_train_size=10,
        n_windows=1,
        model_params=model_params,
    )
    assert {"unique_id", "ds", "cutoff", "step", "y", "yhat", "model"}.issubset(set(pred.columns))


def test_ridge_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "ridge-step-lag-global",
        model_params={
            "lags": 5,
            "alpha": 1.0,
            "roll_windows": (3, 5),
            "roll_stats": ("mean", "std"),
            "diff_lags": (1, 4),
            "x_cols": ("promo",),
            "add_time_features": True,
            "id_feature": "ordinal",
        },
    )


def test_decision_tree_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "decision-tree-step-lag-global",
        model_params={
            "lags": 5,
            "max_depth": 4,
            "random_state": 0,
            "roll_windows": (3,),
            "roll_stats": ("mean", "std"),
            "x_cols": ("promo",),
        },
    )


def test_lasso_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "lasso-step-lag-global",
        model_params={
            "lags": 5,
            "alpha": 0.001,
            "max_iter": 5000,
            "roll_windows": (3,),
            "roll_stats": ("mean", "std"),
            "x_cols": ("promo",),
        },
    )


def test_elasticnet_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "elasticnet-step-lag-global",
        model_params={
            "lags": 5,
            "alpha": 0.001,
            "l1_ratio": 0.5,
            "max_iter": 5000,
            "roll_windows": (3, 5),
            "roll_stats": ("mean",),
            "x_cols": ("promo",),
        },
    )


def test_knn_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "knn-step-lag-global",
        model_params={
            "lags": 5,
            "n_neighbors": 3,
            "weights": "distance",
            "diff_lags": (1, 2),
            "x_cols": ("promo",),
        },
    )


def test_kernel_ridge_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "kernel-ridge-step-lag-global",
        model_params={
            "lags": 5,
            "alpha": 0.5,
            "kernel": "rbf",
            "gamma": 0.1,
            "roll_windows": (3,),
            "roll_stats": ("mean", "std"),
            "x_cols": ("promo",),
        },
    )


def test_svr_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "svr-step-lag-global",
        model_params={
            "lags": 5,
            "C": 2.0,
            "gamma": "scale",
            "epsilon": 0.05,
            "roll_windows": (3, 5),
            "roll_stats": ("mean",),
            "x_cols": ("promo",),
        },
    )


def test_linear_svr_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "linear-svr-step-lag-global",
        model_params={
            "lags": 5,
            "C": 0.5,
            "epsilon": 0.1,
            "max_iter": 20000,
            "random_state": 0,
            "diff_lags": (1, 2),
            "x_cols": ("promo",),
        },
    )


def test_huber_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "huber-step-lag-global",
        model_params={
            "lags": 5,
            "epsilon": 1.35,
            "alpha": 0.0001,
            "max_iter": 200,
            "roll_windows": (3,),
            "roll_stats": ("mean",),
            "x_cols": ("promo",),
        },
    )


def test_quantile_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "quantile-step-lag-global",
        model_params={
            "lags": 5,
            "quantile": 0.5,
            "alpha": 0.0,
            "diff_lags": (1, 2),
            "x_cols": ("promo",),
        },
    )


def test_sgd_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "sgd-step-lag-global",
        model_params={
            "lags": 5,
            "alpha": 0.0001,
            "penalty": "l2",
            "max_iter": 2000,
            "random_state": 0,
            "roll_windows": (3,),
            "roll_stats": ("mean",),
            "x_cols": ("promo",),
        },
    )


def test_adaboost_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "adaboost-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "learning_rate": 0.1,
            "random_state": 0,
            "roll_windows": (3,),
            "roll_stats": ("mean",),
            "x_cols": ("promo",),
        },
    )


def test_mlp_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "mlp-step-lag-global",
        model_params={
            "lags": 5,
            "hidden_layer_sizes": (16,),
            "alpha": 0.0001,
            "max_iter": 1000,
            "random_state": 0,
            "learning_rate_init": 0.01,
            "diff_lags": (1, 2),
            "x_cols": ("promo",),
        },
    )


def test_bagging_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "bagging-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "max_samples": 0.8,
            "random_state": 0,
            "diff_lags": (1, 2),
            "x_cols": ("promo",),
        },
    )


def test_gbrt_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "gbrt-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 0,
            "roll_windows": (3, 5),
            "roll_stats": ("mean", "slope"),
            "x_cols": ("promo",),
        },
    )


def test_rf_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "rf-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "max_depth": 4,
            "random_state": 0,
            "roll_windows": (3,),
            "roll_stats": ("mean", "slope"),
            "x_cols": ("promo",),
        },
    )


def test_extra_trees_step_lag_global_smoke() -> None:
    _assert_global_point_smoke(
        "extra-trees-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "max_depth": 4,
            "random_state": 0,
            "diff_lags": (1, 2),
            "x_cols": ("promo",),
        },
    )


def test_hgb_step_lag_global_smoke() -> None:
    df = _small_panel_long_df()
    pred = cross_validation_predictions_long_df(
        model="hgb-step-lag-global",
        long_df=df,
        horizon=2,
        step_size=2,
        min_train_size=10,
        n_windows=1,
        model_params={
            "lags": 5,
            "max_iter": 50,
            "learning_rate": 0.1,
            "max_depth": 3,
            "roll_windows": (3, 5),
            "roll_stats": ("mean", "std", "min", "max", "slope"),
            "diff_lags": (1, 4),
            "x_cols": ("promo",),
            "add_time_features": True,
            "id_feature": "ordinal",
        },
    )
    assert {"unique_id", "ds", "cutoff", "step", "y", "yhat", "model"}.issubset(set(pred.columns))


def test_xgb_step_lag_global_quantiles_smoke() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")

    df = _small_panel_long_df()
    pred = cross_validation_predictions_long_df(
        model="xgb-step-lag-global",
        long_df=df,
        horizon=2,
        step_size=2,
        min_train_size=10,
        n_windows=1,
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "x_cols": ("promo",),
            "quantiles": (0.1, 0.5, 0.9),
        },
    )
    assert {"yhat_p10", "yhat_p50", "yhat_p90"}.issubset(set(pred.columns))
    assert np.allclose(
        pred["yhat"].to_numpy(dtype=float),
        pred["yhat_p50"].to_numpy(dtype=float),
        atol=1e-9,
    )

    q = evaluate_quantile_predictions(pred)
    assert q["quantiles"] == [10, 50, 90]
    assert np.isfinite(float(q["pinball_mean"]))


def test_xgb_dart_step_lag_global_smoke() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")

    _assert_global_point_smoke(
        "xgb-dart-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "x_cols": ("promo",),
        },
    )


def test_xgb_msle_step_lag_global_smoke() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")

    _assert_global_point_smoke(
        "xgb-msle-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "x_cols": ("promo",),
        },
    )


def test_xgb_mae_step_lag_global_smoke() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")

    _assert_global_point_smoke(
        "xgb-mae-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "x_cols": ("promo",),
        },
    )


def test_xgb_huber_step_lag_global_smoke() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")

    _assert_global_point_smoke(
        "xgb-huber-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "huber_slope": 1.0,
            "x_cols": ("promo",),
        },
    )


def test_xgb_poisson_step_lag_global_smoke() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")

    _assert_global_point_smoke(
        "xgb-poisson-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "x_cols": ("promo",),
        },
    )


def test_xgb_tweedie_step_lag_global_smoke() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")

    _assert_global_point_smoke(
        "xgb-tweedie-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tweedie_variance_power": 1.5,
            "x_cols": ("promo",),
        },
    )


def test_xgb_gamma_step_lag_global_smoke() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")

    _assert_global_point_smoke(
        "xgb-gamma-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "x_cols": ("promo",),
        },
        long_df=_small_panel_positive_long_df(),
    )


def test_xgb_logistic_step_lag_global_smoke() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")

    _assert_global_point_smoke(
        "xgb-logistic-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "x_cols": ("promo",),
        },
        long_df=_small_panel_unit_long_df(),
    )


def test_xgb_linear_step_lag_global_smoke() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")

    _assert_global_point_smoke(
        "xgb-linear-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 50,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "x_cols": ("promo",),
        },
    )


def test_xgbrf_step_lag_global_smoke() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")

    _assert_global_point_smoke(
        "xgbrf-step-lag-global",
        model_params={
            "lags": 5,
            "n_estimators": 25,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "x_cols": ("promo",),
        },
    )


def test_lgbm_step_lag_global_quantiles_smoke() -> None:
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm not installed")

    df = _small_panel_long_df()
    pred = cross_validation_predictions_long_df(
        model="lgbm-step-lag-global",
        long_df=df,
        horizon=2,
        step_size=2,
        min_train_size=10,
        n_windows=1,
        model_params={
            "lags": 5,
            "n_estimators": 40,
            "learning_rate": 0.1,
            "max_depth": 3,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "x_cols": ("promo",),
            "quantiles": (0.1, 0.5, 0.9),
        },
    )
    assert {"yhat_p10", "yhat_p50", "yhat_p90"}.issubset(set(pred.columns))
    assert np.allclose(
        pred["yhat"].to_numpy(dtype=float),
        pred["yhat_p50"].to_numpy(dtype=float),
        atol=1e-9,
    )


def test_catboost_step_lag_global_missing_dep_raises_importerror() -> None:
    if importlib.util.find_spec("catboost") is not None:
        pytest.skip("catboost installed; this test targets the missing-dep path")

    df = _small_panel_long_df().loc[:, ["unique_id", "ds", "y"]]
    with pytest.raises(ImportError):
        cross_validation_predictions_long_df(
            model="catboost-step-lag-global",
            long_df=df,
            horizon=2,
            step_size=2,
            min_train_size=10,
            n_windows=1,
            model_params={"lags": 5},
        )
