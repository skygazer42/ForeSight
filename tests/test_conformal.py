import numpy as np
import pandas as pd

from foresight.conformal import apply_conformal_intervals, fit_conformal_intervals


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
