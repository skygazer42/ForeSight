import importlib.util

import pandas as pd
import pytest

from foresight.cv import cross_validation_predictions_long_df
from foresight.eval_forecast import eval_model_long_df


def _small_panel_long_df() -> pd.DataFrame:
    ds = pd.date_range("2020-01-01", periods=10, freq="D")
    rows = []
    for uid, bias in [("s0", 0.0), ("s1", 1.0)]:
        for i, d in enumerate(ds):
            rows.append({"unique_id": uid, "ds": d, "y": float(bias + 0.1 * i)})
    return pd.DataFrame(rows)


def test_cv_dispatch_uses_global_interface_for_global_models():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    long_df = _small_panel_long_df()
    with pytest.raises(ImportError):
        cross_validation_predictions_long_df(
            model="torch-informer-global",
            long_df=long_df,
            horizon=2,
            step_size=2,
            min_train_size=4,
            n_windows=1,
        )


def test_eval_dispatch_uses_global_interface_for_global_models():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    long_df = _small_panel_long_df()
    with pytest.raises(ImportError):
        eval_model_long_df(
            model="torch-informer-global",
            long_df=long_df,
            horizon=2,
            step=2,
            min_train_size=4,
            max_windows=1,
        )
