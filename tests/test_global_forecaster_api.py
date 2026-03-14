import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models import BaseGlobalForecaster, make_global_forecaster_object
from foresight.models.registry import make_global_forecaster


def _small_panel_long_df() -> pd.DataFrame:
    ds = pd.date_range("2020-01-01", periods=32, freq="D")
    rows = []
    for uid, bias in [("s0", 0.0), ("s1", 1.0)]:
        for i, d in enumerate(ds):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": float(bias + 0.1 * i),
                    "promo": float(i % 7 == 0),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_global_forecaster_object_requires_fit_before_predict() -> None:
    f = make_global_forecaster_object("ridge-step-lag-global", lags=5, alpha=1.0)
    assert isinstance(f, BaseGlobalForecaster)

    with pytest.raises(RuntimeError):
        f.predict(pd.Timestamp("2020-01-20"), 2)


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_global_forecaster_object_supports_fit_then_predict() -> None:
    long_df = _small_panel_long_df()
    cutoff = pd.Timestamp("2020-01-24")

    f = make_global_forecaster_object(
        "ridge-step-lag-global",
        lags=5,
        alpha=0.5,
        x_cols=("promo",),
        add_time_features=True,
        id_feature="ordinal",
    )
    assert f.fit(long_df) is f
    assert f.model_key == "ridge-step-lag-global"
    assert f.model_params["alpha"] == pytest.approx(0.5)

    pred = f.predict(cutoff, 2)
    expected = make_global_forecaster(
        "ridge-step-lag-global",
        lags=5,
        alpha=0.5,
        x_cols=("promo",),
        add_time_features=True,
        id_feature="ordinal",
    )(long_df, cutoff, 2)

    assert list(pred.columns) == ["unique_id", "ds", "yhat"]
    assert pred.shape == expected.shape
    assert pred["unique_id"].tolist() == expected["unique_id"].tolist()
    assert pred["ds"].tolist() == expected["ds"].tolist()
    assert np.allclose(
        pred["yhat"].to_numpy(dtype=float),
        expected["yhat"].to_numpy(dtype=float),
    )
