import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.eval_forecast import eval_model_long_df


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None,
    reason="statsmodels not installed",
)
def test_eval_multivariate_model_df_reports_overall_and_per_target_metrics():
    from foresight.eval_forecast import eval_multivariate_model_df

    df = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=12, freq="D"),
            "y_a": np.arange(12.0),
            "y_b": np.arange(12.0) * 0.5 + 1.0,
        }
    )

    out = eval_multivariate_model_df(
        model="var",
        df=df,
        target_cols=["y_a", "y_b"],
        horizon=2,
        step=2,
        min_train_size=6,
        model_params={"maxlags": 1},
    )

    assert out["model"] == "var"
    assert out["n_targets"] == 2
    assert out["target_cols"] == ["y_a", "y_b"]
    assert out["n_points"] == 12
    assert out["mae"] == pytest.approx(0.0, abs=1e-8)
    assert out["rmse"] == pytest.approx(0.0, abs=1e-8)
    assert set(out["target_metrics"]) == {"y_a", "y_b"}
    assert out["target_metrics"]["y_a"]["n_points"] == 6
    assert out["target_metrics"]["y_a"]["mae"] == pytest.approx(0.0, abs=1e-8)
    assert out["target_metrics"]["y_b"]["rmse"] == pytest.approx(0.0, abs=1e-8)


def test_eval_model_long_df_rejects_multivariate_models_in_single_target_api():
    long_df = pd.DataFrame(
        {
            "unique_id": ["series_1"] * 8,
            "ds": pd.date_range("2020-01-01", periods=8, freq="D"),
            "y": np.arange(8.0),
        }
    )

    with pytest.raises(ValueError, match="multivariate"):
        eval_model_long_df(
            model="var",
            long_df=long_df,
            horizon=2,
            step=1,
            min_train_size=4,
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_eval_multivariate_model_df_supports_torch_stid() -> None:
    from foresight.eval_forecast import eval_multivariate_model_df

    t = np.arange(32.0)
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=32, freq="D"),
            "y_a": np.sin(t / 5.0) + 0.01 * t,
            "y_b": np.cos(t / 7.0) + 0.02 * t,
            "y_c": np.sin(t / 9.0 + 0.5) + 0.015 * t,
        }
    )

    out = eval_multivariate_model_df(
        model="torch-stid-multivariate",
        df=df,
        target_cols=["y_a", "y_b", "y_c"],
        horizon=2,
        step=2,
        min_train_size=20,
        model_params={
            "lags": 16,
            "d_model": 16,
            "num_blocks": 2,
            "epochs": 2,
            "batch_size": 16,
            "device": "cpu",
            "seed": 0,
            "patience": 2,
        },
    )

    assert out["model"] == "torch-stid-multivariate"
    assert out["n_targets"] == 3
    assert out["target_cols"] == ["y_a", "y_b", "y_c"]
    assert out["n_points"] > 0
    assert np.isfinite(float(out["mae"]))
    assert np.isfinite(float(out["rmse"]))
