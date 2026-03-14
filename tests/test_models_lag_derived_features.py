import numpy as np
import pandas as pd
import pytest

from foresight.models.global_regression import _panel_step_lag_predict_X
from foresight.models.registry import make_forecaster


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def test_lr_lag_rejects_invalid_roll_windows() -> None:
    y = 1.0 + np.sin(np.arange(60, dtype=float) / 3.0)
    f = make_forecaster(
        "lr-lag",
        lags=8,
        roll_windows=(9,),  # invalid: roll_window > lags
        roll_stats=("mean",),
    )
    with pytest.raises(ValueError, match="exceeds lags"):
        _ = f(y, 2)


def test_lr_lag_direct_rejects_invalid_roll_windows() -> None:
    y = 1.0 + np.sin(np.arange(60, dtype=float) / 3.0)
    f = make_forecaster(
        "lr-lag-direct",
        lags=8,
        roll_windows=(9,),  # invalid: roll_window > lags
        roll_stats=("mean",),
    )
    with pytest.raises(ValueError, match="exceeds lags"):
        _ = f(y, 2)


def test_ridge_lag_rejects_invalid_roll_windows_when_sklearn_installed() -> None:
    if not _has_module("sklearn"):
        pytest.skip("scikit-learn not installed; this test targets optional-dep path")

    y = 1.0 + np.sin(np.arange(80, dtype=float) / 3.0)
    f = make_forecaster(
        "ridge-lag",
        lags=8,
        alpha=1.0,
        roll_windows=(9,),  # invalid: roll_window > lags
        roll_stats=("mean",),
    )
    with pytest.raises(ValueError, match="exceeds lags"):
        _ = f(y, 2)


def test_lr_lag_supports_target_lags_via_registry() -> None:
    y = 1.0 + np.sin(np.arange(60, dtype=float) / 3.0)
    f = make_forecaster(
        "lr-lag",
        lags=8,
        target_lags=(1, 3, 6),
        roll_windows=(4,),
        roll_stats=("mean",),
    )
    with pytest.raises(ValueError, match="exceeds lags=3"):
        _ = f(y, 2)


def test_panel_step_lag_predict_x_supports_historic_and_future_x_lags() -> None:
    ds = pd.date_range("2020-01-01", periods=10, freq="D")
    g = pd.DataFrame(
        {
            "unique_id": ["s0"] * 10,
            "ds": ds,
            "y": np.arange(10, dtype=float),
            "promo": np.arange(100, 110, dtype=float),
        }
    )

    x_pred, ds_out = _panel_step_lag_predict_X(
        g,
        uid_val=0.0,
        cutoff=ds[7],
        horizon=2,
        lags=5,
        target_lags=(1, 3),
        historic_x_lags=(2,),
        future_x_lags=(1, 0),
        roll_windows=(),
        roll_stats=(),
        diff_lags=(),
        x_cols=("promo",),
        add_time_features=False,
        id_feature="none",
        step_scale="one_based",
    )

    assert list(pd.to_datetime(ds_out)) == [ds[8], ds[9]]
    assert x_pred.tolist() == [
        [5.0, 7.0, 1.0, 106.0, 107.0, 108.0],
        [5.0, 7.0, 2.0, 106.0, 108.0, 109.0],
    ]
