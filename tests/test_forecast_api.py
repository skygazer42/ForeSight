import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.forecast import forecast_model, forecast_model_long_df


def _small_panel_long_df() -> pd.DataFrame:
    ds = pd.date_range("2020-01-01", periods=18, freq="D")
    rows = []
    for uid, bias in [("s0", 0.0), ("s1", 1.0)]:
        for i, d in enumerate(ds):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": float(bias + 0.5 * i),
                    "promo": float(i % 7 == 0),
                }
            )
    return pd.DataFrame(rows)


def test_forecast_model_returns_future_rows_for_single_series() -> None:
    ds = pd.date_range("2020-01-01", periods=5, freq="D")
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    pred = forecast_model(model="naive-last", y=y, ds=ds, horizon=3)

    assert list(pred.columns) == ["unique_id", "ds", "cutoff", "step", "yhat", "model"]
    assert pred["unique_id"].tolist() == ["series=0", "series=0", "series=0"]
    assert pred["ds"].tolist() == list(pd.date_range("2020-01-06", periods=3, freq="D"))
    assert pred["cutoff"].tolist() == [ds[-1], ds[-1], ds[-1]]
    assert pred["step"].tolist() == [1, 2, 3]
    assert np.allclose(pred["yhat"].to_numpy(dtype=float), np.array([5.0, 5.0, 5.0]))
    assert pred["model"].tolist() == ["naive-last", "naive-last", "naive-last"]


def test_forecast_model_can_emit_bootstrap_interval_quantiles_for_local_models() -> None:
    ds = pd.date_range("2020-01-01", periods=5, freq="D")
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    pred = forecast_model(
        model="naive-last",
        y=y,
        ds=ds,
        horizon=2,
        interval_levels=(80, 90),
        interval_min_train_size=3,
        interval_samples=64,
        interval_seed=0,
    )

    assert list(pred.columns) == [
        "unique_id",
        "ds",
        "cutoff",
        "step",
        "yhat",
        "yhat_lo_80",
        "yhat_hi_80",
        "yhat_lo_90",
        "yhat_hi_90",
        "model",
    ]
    assert np.allclose(pred["yhat"].to_numpy(dtype=float), np.array([5.0, 5.0]))
    assert np.allclose(pred["yhat_lo_80"].to_numpy(dtype=float), np.array([6.0, 6.0]))
    assert np.allclose(pred["yhat_hi_80"].to_numpy(dtype=float), np.array([6.0, 6.0]))
    assert np.allclose(pred["yhat_lo_90"].to_numpy(dtype=float), np.array([6.0, 6.0]))
    assert np.allclose(pred["yhat_hi_90"].to_numpy(dtype=float), np.array([6.0, 6.0]))


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_forecast_model_supports_arima_trend_parameter() -> None:
    ds = pd.date_range("2020-01-01", periods=30, freq="D")
    y = np.arange(1.0, 31.0, dtype=float)

    pred = forecast_model(
        model="arima",
        y=y,
        ds=ds,
        horizon=3,
        model_params={"order": (0, 1, 0), "trend": "t"},
    )

    assert pred["ds"].tolist() == list(pd.date_range("2020-01-31", periods=3, freq="D"))
    assert np.allclose(pred["yhat"].to_numpy(dtype=float), np.array([31.0, 32.0, 33.0]), atol=1e-3)


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_forecast_model_long_df_supports_local_sarimax_with_future_covariates() -> None:
    ds = pd.date_range("2020-01-01", periods=30, freq="D")
    promo = np.array(([0, 1, 0, 1, 0] * 6)[:30], dtype=float)
    y = 10.0 + 5.0 * promo + 0.1 * np.arange(30, dtype=float)

    history = pd.DataFrame(
        {
            "unique_id": ["s0"] * 30,
            "ds": ds,
            "y": y,
            "promo": promo,
        }
    )
    future = pd.DataFrame(
        {
            "unique_id": ["s0"] * 3,
            "ds": pd.date_range("2020-01-31", periods=3, freq="D"),
            "y": [np.nan, np.nan, np.nan],
            "promo": [1.0, 0.0, 1.0],
        }
    )
    long_df = pd.concat([history, future], ignore_index=True, sort=False)

    pred = forecast_model_long_df(
        model="sarimax",
        long_df=long_df,
        horizon=3,
        model_params={
            "order": (0, 0, 0),
            "seasonal_order": (0, 0, 0, 0),
            "trend": "c",
            "x_cols": ("promo",),
        },
    )

    assert pred["unique_id"].tolist() == ["s0", "s0", "s0"]
    assert pred["ds"].tolist() == list(pd.date_range("2020-01-31", periods=3, freq="D"))
    assert pred["cutoff"].tolist() == [pd.Timestamp("2020-01-30")] * 3
    assert pred["step"].tolist() == [1, 2, 3]
    yhat = pred["yhat"].to_numpy(dtype=float)
    assert yhat[0] > yhat[1] + 4.0
    assert yhat[2] > yhat[1] + 4.0
    assert abs(float(yhat[0]) - float(yhat[2])) < 1e-6


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_forecast_model_long_df_supports_local_sarimax_with_future_covariates_for_multiple_series() -> None:
    ds = pd.date_range("2020-01-01", periods=30, freq="D")
    promo = np.array(([0, 1, 0, 1, 0] * 6)[:30], dtype=float)

    rows: list[dict[str, object]] = []
    for uid, bias in [("s0", 0.0), ("s1", 20.0)]:
        y = bias + 10.0 + 5.0 * promo + 0.1 * np.arange(30, dtype=float)
        rows.extend(
            {
                "unique_id": uid,
                "ds": d,
                "y": float(target),
                "promo": float(p),
            }
            for d, target, p in zip(ds, y, promo, strict=True)
        )
        rows.extend(
            {
                "unique_id": uid,
                "ds": d,
                "y": np.nan,
                "promo": float(p),
            }
            for d, p in zip(pd.date_range("2020-01-31", periods=2, freq="D"), [1.0, 0.0], strict=True)
        )
    long_df = pd.DataFrame(rows)

    pred = forecast_model_long_df(
        model="sarimax",
        long_df=long_df,
        horizon=2,
        model_params={
            "order": (0, 0, 0),
            "seasonal_order": (0, 0, 0, 0),
            "trend": "c",
            "x_cols": ("promo",),
        },
    )

    assert pred.shape == (4, 6)
    assert pred["unique_id"].tolist() == ["s0", "s0", "s1", "s1"]
    assert pred["step"].tolist() == [1, 2, 1, 2]
    assert pred["cutoff"].tolist() == [pd.Timestamp("2020-01-30")] * 4
    assert pred["ds"].tolist() == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-01"),
    ]


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_forecast_model_long_df_supports_local_auto_arima_with_future_covariates() -> None:
    ds = pd.date_range("2020-01-01", periods=30, freq="D")
    promo = np.array(([0, 1, 0, 1, 0] * 6)[:30], dtype=float)
    y = 10.0 + 5.0 * promo

    history = pd.DataFrame(
        {
            "unique_id": ["s0"] * 30,
            "ds": ds,
            "y": y,
            "promo": promo,
        }
    )
    future = pd.DataFrame(
        {
            "unique_id": ["s0"] * 3,
            "ds": pd.date_range("2020-01-31", periods=3, freq="D"),
            "y": [np.nan, np.nan, np.nan],
            "promo": [1.0, 0.0, 1.0],
        }
    )
    long_df = pd.concat([history, future], ignore_index=True, sort=False)

    pred = forecast_model_long_df(
        model="auto-arima",
        long_df=long_df,
        horizon=3,
        model_params={
            "max_p": 0,
            "max_d": 0,
            "max_q": 0,
            "max_P": 0,
            "max_D": 0,
            "max_Q": 0,
            "trend": "c",
            "x_cols": ("promo",),
        },
    )

    assert pred["unique_id"].tolist() == ["s0", "s0", "s0"]
    assert pred["ds"].tolist() == list(pd.date_range("2020-01-31", periods=3, freq="D"))
    assert pred["cutoff"].tolist() == [pd.Timestamp("2020-01-30")] * 3
    assert pred["step"].tolist() == [1, 2, 3]
    yhat = pred["yhat"].to_numpy(dtype=float)
    assert yhat[0] > yhat[1] + 4.0
    assert yhat[2] > yhat[1] + 4.0
    assert abs(float(yhat[0]) - float(yhat[2])) < 1e-3


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_forecast_model_long_df_supports_local_sarimax_with_future_covariates_and_intervals() -> None:
    ds = pd.date_range("2020-01-01", periods=30, freq="D")
    promo = np.array(([0, 1, 0, 1, 0] * 6)[:30], dtype=float)
    y = 10.0 + 5.0 * promo + 0.1 * np.arange(30, dtype=float)

    history = pd.DataFrame(
        {
            "unique_id": ["s0"] * 30,
            "ds": ds,
            "y": y,
            "promo": promo,
        }
    )
    future = pd.DataFrame(
        {
            "unique_id": ["s0"] * 3,
            "ds": pd.date_range("2020-01-31", periods=3, freq="D"),
            "y": [np.nan, np.nan, np.nan],
            "promo": [1.0, 0.0, 1.0],
        }
    )
    long_df = pd.concat([history, future], ignore_index=True, sort=False)

    pred = forecast_model_long_df(
        model="sarimax",
        long_df=long_df,
        horizon=3,
        interval_levels=(80,),
        model_params={
            "order": (0, 0, 0),
            "seasonal_order": (0, 0, 0, 0),
            "trend": "c",
            "x_cols": ("promo",),
        },
    )

    assert {"yhat_lo_80", "yhat_hi_80"}.issubset(set(pred.columns))
    yhat = pred["yhat"].to_numpy(dtype=float)
    lo = pred["yhat_lo_80"].to_numpy(dtype=float)
    hi = pred["yhat_hi_80"].to_numpy(dtype=float)
    assert np.isfinite(lo).all()
    assert np.isfinite(hi).all()
    assert np.all(lo <= yhat)
    assert np.all(yhat <= hi)
    assert yhat[0] > yhat[1] + 4.0
    assert yhat[2] > yhat[1] + 4.0


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_forecast_model_long_df_supports_local_auto_arima_with_future_covariates_and_intervals() -> None:
    ds = pd.date_range("2020-01-01", periods=30, freq="D")
    promo = np.array(([0, 1, 0, 1, 0] * 6)[:30], dtype=float)
    y = 10.0 + 5.0 * promo

    history = pd.DataFrame(
        {
            "unique_id": ["s0"] * 30,
            "ds": ds,
            "y": y,
            "promo": promo,
        }
    )
    future = pd.DataFrame(
        {
            "unique_id": ["s0"] * 3,
            "ds": pd.date_range("2020-01-31", periods=3, freq="D"),
            "y": [np.nan, np.nan, np.nan],
            "promo": [1.0, 0.0, 1.0],
        }
    )
    long_df = pd.concat([history, future], ignore_index=True, sort=False)

    pred = forecast_model_long_df(
        model="auto-arima",
        long_df=long_df,
        horizon=3,
        interval_levels=(80,),
        model_params={
            "max_p": 0,
            "max_d": 0,
            "max_q": 0,
            "max_P": 0,
            "max_D": 0,
            "max_Q": 0,
            "trend": "c",
            "x_cols": ("promo",),
        },
    )

    assert {"yhat_lo_80", "yhat_hi_80"}.issubset(set(pred.columns))
    yhat = pred["yhat"].to_numpy(dtype=float)
    lo = pred["yhat_lo_80"].to_numpy(dtype=float)
    hi = pred["yhat_hi_80"].to_numpy(dtype=float)
    assert np.isfinite(lo).all()
    assert np.isfinite(hi).all()
    assert np.all(lo <= yhat)
    assert np.all(yhat <= hi)
    assert yhat[0] > yhat[1] + 4.0
    assert yhat[2] > yhat[1] + 4.0


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed")
def test_forecast_model_long_df_supports_global_models() -> None:
    long_df = _small_panel_long_df()

    pred = forecast_model_long_df(
        model="ridge-step-lag-global",
        long_df=long_df,
        horizon=2,
        model_params={
            "lags": 5,
            "alpha": 0.5,
            "add_time_features": True,
            "id_feature": "ordinal",
        },
    )

    assert list(pred.columns) == ["unique_id", "ds", "cutoff", "step", "yhat", "model"]
    assert pred.shape == (4, 6)
    assert pred["unique_id"].tolist() == ["s0", "s0", "s1", "s1"]
    assert pred["ds"].tolist() == [
        pd.Timestamp("2020-01-19"),
        pd.Timestamp("2020-01-20"),
        pd.Timestamp("2020-01-19"),
        pd.Timestamp("2020-01-20"),
    ]
    assert pred["cutoff"].tolist() == [pd.Timestamp("2020-01-18")] * 4
    assert pred["step"].tolist() == [1, 2, 1, 2]
    assert np.isfinite(pred["yhat"].to_numpy(dtype=float)).all()
    assert pred["model"].tolist() == ["ridge-step-lag-global"] * 4


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed")
def test_forecast_model_long_df_supports_global_models_with_future_covariates() -> None:
    long_df = _small_panel_long_df()
    future_rows = []
    for uid in ["s0", "s1"]:
        future_rows.extend(
            [
                {"unique_id": uid, "ds": pd.Timestamp("2020-01-19"), "y": np.nan, "promo": 1.0},
                {"unique_id": uid, "ds": pd.Timestamp("2020-01-20"), "y": np.nan, "promo": 0.0},
            ]
        )
    augmented = pd.concat([long_df, pd.DataFrame(future_rows)], ignore_index=True, sort=False)

    pred = forecast_model_long_df(
        model="ridge-step-lag-global",
        long_df=augmented,
        horizon=2,
        model_params={
            "lags": 5,
            "alpha": 0.5,
            "x_cols": ("promo",),
            "add_time_features": True,
            "id_feature": "ordinal",
        },
    )

    assert list(pred.columns) == ["unique_id", "ds", "cutoff", "step", "yhat", "model"]
    assert pred.shape == (4, 6)
    assert pred["unique_id"].tolist() == ["s0", "s0", "s1", "s1"]
    assert pred["ds"].tolist() == [
        pd.Timestamp("2020-01-19"),
        pd.Timestamp("2020-01-20"),
        pd.Timestamp("2020-01-19"),
        pd.Timestamp("2020-01-20"),
    ]
    assert pred["cutoff"].tolist() == [pd.Timestamp("2020-01-18")] * 4
    assert pred["step"].tolist() == [1, 2, 1, 2]
    assert np.isfinite(pred["yhat"].to_numpy(dtype=float)).all()


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="xgboost not installed")
def test_forecast_model_long_df_emits_interval_columns_for_quantile_global_models() -> None:
    long_df = _small_panel_long_df()
    future_rows = []
    for uid in ["s0", "s1"]:
        future_rows.extend(
            [
                {"unique_id": uid, "ds": pd.Timestamp("2020-01-19"), "y": np.nan, "promo": 1.0},
                {"unique_id": uid, "ds": pd.Timestamp("2020-01-20"), "y": np.nan, "promo": 0.0},
            ]
        )
    augmented = pd.concat([long_df, pd.DataFrame(future_rows)], ignore_index=True, sort=False)

    pred = forecast_model_long_df(
        model="xgb-step-lag-global",
        long_df=augmented,
        horizon=2,
        interval_levels=(80,),
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

    assert {"yhat_p10", "yhat_p50", "yhat_p90", "yhat_lo_80", "yhat_hi_80"}.issubset(set(pred.columns))
    assert np.allclose(pred["yhat"].to_numpy(dtype=float), pred["yhat_p50"].to_numpy(dtype=float))
    assert np.allclose(pred["yhat_lo_80"].to_numpy(dtype=float), pred["yhat_p10"].to_numpy(dtype=float))
    assert np.allclose(pred["yhat_hi_80"].to_numpy(dtype=float), pred["yhat_p90"].to_numpy(dtype=float))


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed")
def test_forecast_model_long_df_rejects_interval_levels_for_global_models_without_capability() -> None:
    long_df = _small_panel_long_df()

    with pytest.raises(
        ValueError,
        match="ridge-step-lag-global'.*does not support interval_levels in forecast_model_long_df",
    ):
        forecast_model_long_df(
            model="ridge-step-lag-global",
            long_df=long_df,
            horizon=2,
            interval_levels=(80,),
            model_params={
                "lags": 5,
                "alpha": 0.5,
                "add_time_features": True,
                "id_feature": "ordinal",
            },
        )


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed")
def test_forecast_model_long_df_rejects_missing_y_inside_global_history() -> None:
    long_df = _small_panel_long_df()
    long_df.loc[
        (long_df["unique_id"] == "s0") & (long_df["ds"] == pd.Timestamp("2020-01-10")),
        "y",
    ] = np.nan

    with pytest.raises(ValueError, match="missing y values only after the observed history"):
        forecast_model_long_df(
            model="ridge-step-lag-global",
            long_df=long_df,
            horizon=2,
            model_params={
                "lags": 5,
                "alpha": 0.5,
                "add_time_features": True,
                "id_feature": "ordinal",
            },
        )
