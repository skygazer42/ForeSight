from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foresight.detect import detect_anomalies, detect_anomalies_long_df
from foresight.models import registry as registry_mod


def _install_dummy_local_xreg_model(monkeypatch: pytest.MonkeyPatch, *, key: str) -> str:
    def _factory(**_params: object):
        def _f(
            train: object,
            horizon: int,
            *,
            train_exog: object | None = None,
            future_exog: object | None = None,
        ):
            train_arr = pd.Series(train, dtype=float).to_numpy(dtype=float, copy=False)
            if train_exog is None or future_exog is None:
                raise ValueError("train_exog and future_exog are required")
            train_x = pd.DataFrame(train_exog).to_numpy(dtype=float, copy=False)
            future_x = pd.DataFrame(future_exog).to_numpy(dtype=float, copy=False)
            if train_x.shape[0] != train_arr.size:
                raise ValueError("train_exog rows must equal train length")
            if future_x.shape[0] != int(horizon):
                raise ValueError("future_exog rows must equal horizon")
            return future_x[:, 0].astype(float, copy=False)

        return _f

    spec = registry_mod.ModelSpec(
        key=key,
        description="Test-only local xreg model for detection",
        factory=_factory,
        param_help={"x_cols": "Required future covariate columns"},
        capability_overrides={"requires_future_covariates": True},
    )
    monkeypatch.setitem(registry_mod._REGISTRY, key, spec)
    return key


def test_detect_anomalies_long_df_with_forecast_residual_flags_spike_and_reversal() -> None:
    values = [10.0] * 8 + [50.0, 10.0] + [10.0] * 6
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * len(values),
            "ds": pd.date_range("2020-01-01", periods=len(values), freq="D"),
            "y": values,
        }
    )

    out = detect_anomalies_long_df(
        model="naive-last",
        long_df=long_df,
        score_method="forecast-residual",
        threshold_method="mad",
        threshold_k=3.0,
        min_train_size=5,
        step_size=1,
    )

    assert {
        "unique_id",
        "ds",
        "cutoff",
        "step",
        "y",
        "yhat",
        "residual",
        "score",
        "threshold",
        "is_anomaly",
        "score_method",
        "threshold_method",
        "window_context",
        "model",
    }.issubset(set(out.columns))
    assert (out["score"] >= 0.0).all()
    assert out["model"].eq("naive-last").all()
    assert out["score_method"].eq("forecast-residual").all()
    assert out["threshold_method"].eq("mad").all()
    flagged_ds = set(out.loc[out["is_anomaly"], "ds"].dt.strftime("%Y-%m-%d"))
    assert {"2020-01-09", "2020-01-10"}.issubset(flagged_ds)


def test_detect_anomalies_long_df_with_rolling_zscore_without_model_flags_spike() -> None:
    values = [2.0] * 9 + [20.0] + [2.0] * 5
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * len(values),
            "ds": pd.date_range("2020-01-01", periods=len(values), freq="D"),
            "y": values,
        }
    )

    out = detect_anomalies_long_df(
        long_df=long_df,
        score_method="rolling-zscore",
        threshold_method="zscore",
        threshold_k=3.0,
        window=5,
        min_history=3,
    )

    spike = out.loc[out["ds"] == pd.Timestamp("2020-01-10")].iloc[0]
    assert bool(spike["is_anomaly"])
    assert spike["model"] is None
    assert spike["score_method"] == "rolling-zscore"
    assert spike["threshold_method"] == "zscore"
    assert float(spike["threshold"]) == 3.0


def test_detect_anomalies_dataset_uses_dataset_long_df_path() -> None:
    out = detect_anomalies(
        dataset="catfish",
        y_col="Total",
        model="naive-last",
        score_method="forecast-residual",
        threshold_method="quantile",
        threshold_quantile=0.95,
        min_train_size=12,
        step_size=3,
        n_windows=4,
    )

    assert not out.empty
    assert out["unique_id"].nunique() == 1
    assert out["model"].eq("naive-last").all()
    assert out.attrs["n_series"] == 1
    assert out.attrs["score_method"] == "forecast-residual"
    assert out.attrs["threshold_method"] == "quantile"


def test_detect_anomalies_long_df_supports_local_future_x_cols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _install_dummy_local_xreg_model(
        monkeypatch,
        key="__test-local-xreg-detect__",
    )

    promo = [0.0, 1.0] * 8
    y = promo.copy()
    y[9] = 12.0
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * len(y),
            "ds": pd.date_range("2020-01-01", periods=len(y), freq="D"),
            "y": y,
            "promo": promo,
        }
    )

    out = detect_anomalies_long_df(
        model=key,
        long_df=long_df,
        score_method="forecast-residual",
        threshold_method="mad",
        min_train_size=5,
        step_size=1,
        model_params={"future_x_cols": ("promo",)},
    )

    flagged_ds = set(out.loc[out["is_anomaly"], "ds"].dt.strftime("%Y-%m-%d"))
    assert "2020-01-10" in flagged_ds
    flagged = out.loc[out["is_anomaly"]].copy()
    assert not flagged.empty
    assert flagged["score_method"].eq("forecast-residual").all()
    assert flagged["model"].eq(key).all()
