from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foresight.eval_forecast import eval_model_long_df
from foresight.forecast import forecast_model_long_df
from foresight.models import registry as registry_mod


def _install_dummy_global_static_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    key: str,
    supports_static_cols: bool,
) -> str:
    def _factory(**_params: object):
        def _f(long_df: pd.DataFrame, cutoff: object, horizon: int) -> pd.DataFrame:
            rows: list[dict[str, object]] = []
            cutoff_ts = pd.Timestamp(cutoff)
            for uid, group in long_df.groupby("unique_id", sort=False):
                ordered = group.sort_values("ds", kind="mergesort")
                future = ordered[ordered["ds"] > cutoff_ts].head(int(horizon)).copy()
                if len(future) != int(horizon):
                    continue
                static_value = float(ordered["store_size"].dropna().iloc[0])
                for _, row in future.iterrows():
                    rows.append(
                        {
                            "unique_id": uid,
                            "ds": row["ds"],
                            "yhat": static_value,
                        }
                    )
            return pd.DataFrame(rows)

        return _f

    param_help: dict[str, str] = {}
    if supports_static_cols:
        param_help["static_cols"] = "Static covariate columns"

    spec = registry_mod.ModelSpec(
        key=key,
        description="Test-only global static covariate model",
        factory=_factory,
        param_help=param_help,
        interface="global",
    )
    monkeypatch.setitem(registry_mod._REGISTRY, key, spec)
    return key


def _make_static_panel_history_and_future() -> tuple[pd.DataFrame, pd.DataFrame]:
    history_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []

    history_ds = pd.date_range("2020-01-01", periods=6, freq="D")
    future_ds = pd.date_range("2020-01-07", periods=2, freq="D")

    for uid, store_size in (("s0", 10.0), ("s1", 20.0)):
        for d in history_ds:
            history_rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": store_size,
                    "store_size": store_size,
                }
            )
        for d in future_ds:
            future_rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "store_size": np.nan,
                }
            )

    return pd.DataFrame(history_rows), pd.DataFrame(future_rows)


def _make_static_eval_long_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ds = pd.date_range("2020-01-01", periods=8, freq="D")
    for uid, store_size in (("s0", 10.0), ("s1", 20.0)):
        for d in ds:
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": store_size,
                    "store_size": store_size,
                }
            )
    return pd.DataFrame(rows)


def test_forecast_model_long_df_supports_global_models_with_static_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _install_dummy_global_static_model(
        monkeypatch,
        key="__test-global-static-forecast__",
        supports_static_cols=True,
    )
    history, future_df = _make_static_panel_history_and_future()

    pred = forecast_model_long_df(
        model=key,
        long_df=history,
        future_df=future_df,
        horizon=2,
        model_params={"static_cols": ("store_size",)},
    )

    assert pred["yhat"].tolist() == [10.0, 10.0, 20.0, 20.0]
    assert pred["step"].tolist() == [1, 2, 1, 2]


def test_forecast_model_long_df_rejects_static_covariates_for_unsupported_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _install_dummy_global_static_model(
        monkeypatch,
        key="__test-global-static-unsupported__",
        supports_static_cols=False,
    )
    history, future_df = _make_static_panel_history_and_future()

    with pytest.raises(ValueError, match="does not support static_cols in forecast_model_long_df"):
        forecast_model_long_df(
            model=key,
            long_df=history,
            future_df=future_df,
            horizon=2,
            model_params={"static_cols": ("store_size",)},
        )


def test_eval_model_long_df_supports_global_models_with_static_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _install_dummy_global_static_model(
        monkeypatch,
        key="__test-global-static-eval__",
        supports_static_cols=True,
    )
    long_df = _make_static_eval_long_df()

    payload = eval_model_long_df(
        model=key,
        long_df=long_df,
        horizon=2,
        step=2,
        min_train_size=4,
        model_params={"static_cols": ("store_size",)},
    )

    assert payload["model"] == key
    assert payload["n_series"] == 2
    assert payload["mae"] == pytest.approx(0.0)
