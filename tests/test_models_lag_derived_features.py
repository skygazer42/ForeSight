import numpy as np
import pytest

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
