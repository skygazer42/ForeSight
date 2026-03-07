import importlib.util

import pandas as pd
import pytest

from foresight.eval_forecast import eval_model_long_df


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_eval_model_long_df_supports_local_sarimax_with_x_cols() -> None:
    promo = ([0, 1, 0, 1, 0] * 6)[:30]
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 30,
            "ds": pd.date_range("2020-01-01", periods=30, freq="D"),
            "y": [10.0 + 5.0 * float(p) for p in promo],
            "promo": promo,
        }
    )

    payload = eval_model_long_df(
        model="sarimax",
        long_df=long_df,
        horizon=3,
        step=3,
        min_train_size=12,
        model_params={
            "order": (0, 0, 0),
            "seasonal_order": (0, 0, 0, 0),
            "trend": "c",
            "x_cols": ("promo",),
        },
    )

    assert payload["model"] == "sarimax"
    assert payload["n_series"] == 1
    assert payload["n_series_skipped"] == 0
    assert payload["n_points"] == 18
    assert payload["mae"] < 1e-3
    assert payload["rmse"] < 1e-3


def test_eval_model_long_df_rejects_x_cols_for_local_models_without_capability() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 20,
            "ds": pd.date_range("2020-01-01", periods=20, freq="D"),
            "y": [float(i) for i in range(20)],
            "promo": [float(i % 2) for i in range(20)],
        }
    )

    with pytest.raises(ValueError, match="theta'.*does not support x_cols in eval_model_long_df"):
        eval_model_long_df(
            model="theta",
            long_df=long_df,
            horizon=2,
            step=2,
            min_train_size=8,
            model_params={"x_cols": ("promo",)},
        )


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_eval_model_long_df_supports_local_auto_arima_with_x_cols() -> None:
    promo = ([0, 1, 0, 1, 0] * 6)[:30]
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 30,
            "ds": pd.date_range("2020-01-01", periods=30, freq="D"),
            "y": [10.0 + 5.0 * float(p) for p in promo],
            "promo": promo,
        }
    )

    payload = eval_model_long_df(
        model="auto-arima",
        long_df=long_df,
        horizon=3,
        step=3,
        min_train_size=12,
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

    assert payload["model"] == "auto-arima"
    assert payload["n_series"] == 1
    assert payload["n_series_skipped"] == 0
    assert payload["n_points"] == 18
    assert payload["mae"] < 1e-3
    assert payload["rmse"] < 1e-3
