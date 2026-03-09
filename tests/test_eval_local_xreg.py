import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.eval_forecast import eval_model_long_df
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
        description="Test-only local xreg model",
        factory=_factory,
        param_help={"x_cols": "Required future covariate columns"},
        capability_overrides={"requires_future_covariates": True},
    )
    monkeypatch.setitem(registry_mod._REGISTRY, key, spec)
    return key


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
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


def test_eval_model_long_df_supports_generic_local_xreg_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _install_dummy_local_xreg_model(monkeypatch, key="__test-local-xreg-eval__")

    promo = ([0, 1, 0, 1, 0, 1] * 5)[:30]
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 30,
            "ds": pd.date_range("2020-01-01", periods=30, freq="D"),
            "y": [float(p) for p in promo],
            "promo": [float(p) for p in promo],
        }
    )

    payload = eval_model_long_df(
        model=key,
        long_df=long_df,
        horizon=3,
        step=3,
        min_train_size=12,
        model_params={"x_cols": ("promo",)},
    )

    assert payload["model"] == key
    assert payload["n_series"] == 1
    assert payload["n_series_skipped"] == 0
    assert payload["n_points"] == 18
    assert payload["mae"] < 1e-9
    assert payload["rmse"] < 1e-9


def test_eval_model_long_df_supports_generic_local_xreg_models_with_future_x_cols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _install_dummy_local_xreg_model(monkeypatch, key="__test-local-future-x-eval__")

    promo = ([0, 1, 0, 1, 0, 1] * 5)[:30]
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 30,
            "ds": pd.date_range("2020-01-01", periods=30, freq="D"),
            "y": [float(p) for p in promo],
            "promo": [float(p) for p in promo],
        }
    )

    payload = eval_model_long_df(
        model=key,
        long_df=long_df,
        horizon=3,
        step=3,
        min_train_size=12,
        model_params={"future_x_cols": ("promo",)},
    )

    assert payload["model"] == key
    assert payload["mae"] < 1e-9


def test_eval_model_long_df_rejects_historic_x_cols_for_current_local_xreg_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _install_dummy_local_xreg_model(monkeypatch, key="__test-local-historic-x-eval__")

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 20,
            "ds": pd.date_range("2020-01-01", periods=20, freq="D"),
            "y": [float(i % 2) for i in range(20)],
            "promo_hist": [float(i % 2) for i in range(20)],
            "promo": [float(i % 2) for i in range(20)],
        }
    )

    with pytest.raises(ValueError, match="historic_x_cols are not yet supported"):
        eval_model_long_df(
            model=key,
            long_df=long_df,
            horizon=2,
            step=2,
            min_train_size=8,
            model_params={"historic_x_cols": ("promo_hist",), "future_x_cols": ("promo",)},
        )


def test_eval_model_long_df_requires_x_cols_for_local_models_that_need_future_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _install_dummy_local_xreg_model(monkeypatch, key="__test-local-xreg-required-eval__")

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 20,
            "ds": pd.date_range("2020-01-01", periods=20, freq="D"),
            "y": [float(i % 2) for i in range(20)],
            "promo": [float(i % 2) for i in range(20)],
        }
    )

    with pytest.raises(ValueError, match="requires future covariates via x_cols"):
        eval_model_long_df(
            model=key,
            long_df=long_df,
            horizon=2,
            step=2,
            min_train_size=8,
            model_params={},
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_eval_model_long_df_supports_local_timexer_with_x_cols() -> None:
    promo = ([0, 1, 0, 1, 0] * 8)[:40]
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 40,
            "ds": pd.date_range("2020-01-01", periods=40, freq="D"),
            "y": [5.0 + float(i % 3) + 1.0 * float(p) for i, p in enumerate(promo)],
            "promo": [float(p) for p in promo],
        }
    )

    payload = eval_model_long_df(
        model="torch-timexer-direct",
        long_df=long_df,
        horizon=3,
        step=3,
        min_train_size=24,
        model_params={
            "x_cols": ("promo",),
            "lags": 16,
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "epochs": 2,
            "batch_size": 16,
            "device": "cpu",
            "seed": 0,
            "patience": 2,
        },
    )

    assert payload["model"] == "torch-timexer-direct"
    assert payload["n_series"] == 1
    assert payload["n_series_skipped"] == 0
    assert payload["n_points"] > 0
    assert np.isfinite(float(payload["mae"]))
    assert np.isfinite(float(payload["rmse"]))


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
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
