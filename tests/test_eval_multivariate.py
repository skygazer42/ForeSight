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
