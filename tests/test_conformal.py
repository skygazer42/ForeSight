import numpy as np
import pandas as pd
import pytest

from foresight.conformal import (
    apply_conformal_intervals,
    fit_conformal_intervals,
    summarize_conformal_predictions,
)
from foresight.eval_forecast import eval_model_long_df


def test_fit_and_apply_conformal_per_step():
    df = pd.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            "yhat": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "step": [1, 2, 3, 1, 2, 3],
        }
    )
    conf = fit_conformal_intervals(df, levels=(0.8, 0.9), per_step=True)
    out = apply_conformal_intervals(df, conf)
    assert "yhat_lo_80" in out.columns
    assert "yhat_hi_80" in out.columns
    assert np.all(out["yhat_lo_80"].to_numpy() <= out["yhat"].to_numpy())
    assert np.all(out["yhat_hi_80"].to_numpy() >= out["yhat"].to_numpy())


def test_fit_conformal_pooled():
    df = pd.DataFrame({"y": [0.0, 2.0], "yhat": [1.0, 1.0], "step": [1, 2]})
    conf = fit_conformal_intervals(df, levels=(0.9,), per_step=False)
    assert conf.per_step is False
    assert conf.radius[0.9].shape == (2,)


def test_summarize_conformal_predictions_reports_calibration_and_sharpness():
    df = pd.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            "yhat": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "step": [1, 2, 3, 1, 2, 3],
        }
    )
    payload = summarize_conformal_predictions(df, levels=(0.8,), per_step=True)

    assert "coverage_80" in payload
    assert "calibration_gap_80" in payload
    assert "sharpness_80" in payload
    assert "interval_score_80" in payload
    assert "winkler_score_80" in payload
    assert payload["sharpness_80"] == pytest.approx(payload["mean_width_80"])
    assert payload["calibration_gap_80"] == pytest.approx(payload["coverage_80"] - 0.8)


def test_eval_model_long_df_with_conformal_outputs_calibration_summary():
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 8,
            "ds": pd.date_range("2020-01-01", periods=8, freq="D"),
            "y": [1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0],
        }
    )

    payload = eval_model_long_df(
        model="naive-last",
        long_df=long_df,
        horizon=2,
        step=1,
        min_train_size=4,
        conformal_levels=(0.8,),
    )

    assert "coverage_80" in payload
    assert "calibration_gap_80" in payload
    assert "sharpness_80" in payload
    assert "interval_score_80" in payload
    assert "winkler_score_80" in payload
    assert payload["calibration_gap_80"] == pytest.approx(payload["coverage_80"] - 0.8)
