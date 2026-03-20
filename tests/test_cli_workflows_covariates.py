from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from foresight.services import cli_workflows


def test_forecast_csv_workflow_future_frame_only_uses_future_covariates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.csv"
    future_path = tmp_path / "future.csv"

    pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 3.0],
            "hist": [10.0, 11.0, 12.0],
            "futr": [20.0, 21.0, 22.0],
            "store_size": [5.0, 5.0, 5.0],
        }
    ).to_csv(history_path, index=False)
    pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-04", periods=2, freq="D"),
            "futr": [23.0, 24.0],
        }
    ).to_csv(future_path, index=False)

    captured: dict[str, object] = {}

    def _fake_get_model_spec(_model_key: str) -> SimpleNamespace:
        return SimpleNamespace(interface="global")

    def _fake_forecast_model_long_df(**kwargs: object) -> pd.DataFrame:
        long_df = kwargs["long_df"]
        future_df = kwargs["future_df"]
        assert isinstance(long_df, pd.DataFrame)
        assert isinstance(future_df, pd.DataFrame)
        captured["history_historic"] = long_df.attrs.get("historic_x_cols")
        captured["history_future"] = long_df.attrs.get("future_x_cols")
        captured["history_static"] = long_df.attrs.get("static_cols")
        captured["future_historic"] = future_df.attrs.get("historic_x_cols")
        captured["future_future"] = future_df.attrs.get("future_x_cols")
        captured["future_static"] = future_df.attrs.get("static_cols")
        captured["future_columns"] = tuple(future_df.columns)
        return pd.DataFrame(
            {
                "unique_id": ["series=0", "series=0"],
                "ds": pd.date_range("2020-01-04", periods=2, freq="D"),
                "cutoff": [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-03")],
                "step": [1, 2],
                "yhat": [4.0, 5.0],
                "model": ["demo-global", "demo-global"],
            }
        )

    monkeypatch.setattr(cli_workflows._model_execution, "get_model_spec", _fake_get_model_spec)
    monkeypatch.setattr(
        cli_workflows._forecasting,
        "forecast_model_long_df",
        _fake_forecast_model_long_df,
    )

    pred = cli_workflows.forecast_csv_workflow(
        model="demo-global",
        path=str(history_path),
        future_path=str(future_path),
        time_col="ds",
        y_col="y",
        horizon=2,
        parse_dates=True,
        model_params={
            "historic_x_cols": ("hist",),
            "future_x_cols": ("futr",),
            "static_cols": ("store_size",),
        },
    )

    assert pred["step"].tolist() == [1, 2]
    assert captured["history_historic"] == ("hist",)
    assert captured["history_future"] == ("futr",)
    assert captured["history_static"] == ("store_size",)
    assert captured["future_historic"] == ()
    assert captured["future_future"] == ("futr",)
    assert captured["future_static"] == ()
    assert "hist" not in captured["future_columns"]
    assert "store_size" not in captured["future_columns"]
    assert "futr" in captured["future_columns"]


def test_eval_csv_workflow_preserves_covariate_role_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.csv"
    pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=6, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "hist": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "futr": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            "store_size": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
    ).to_csv(history_path, index=False)

    captured: dict[str, object] = {}

    def _fake_eval_model_long_df(**kwargs: object) -> dict[str, object]:
        long_df = kwargs["long_df"]
        assert isinstance(long_df, pd.DataFrame)
        captured["historic"] = long_df.attrs.get("historic_x_cols")
        captured["future"] = long_df.attrs.get("future_x_cols")
        captured["static"] = long_df.attrs.get("static_cols")
        captured["columns"] = tuple(long_df.columns)
        return {
            "model": str(kwargs["model"]),
            "n_series": 1,
            "n_series_skipped": 0,
            "n_points": int(len(long_df)),
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "smape": 0.0,
        }

    monkeypatch.setattr(
        cli_workflows._evaluation,
        "eval_model_long_df",
        _fake_eval_model_long_df,
    )

    payload = cli_workflows.eval_csv_workflow(
        model="demo-eval",
        path=str(history_path),
        time_col="ds",
        y_col="y",
        horizon=2,
        step=1,
        min_train_size=3,
        parse_dates=True,
        model_params={
            "historic_x_cols": ("hist",),
            "future_x_cols": ("futr",),
            "static_cols": ("store_size",),
        },
    )

    assert payload["dataset"] == str(history_path)
    assert captured["historic"] == ("hist",)
    assert captured["future"] == ("futr",)
    assert captured["static"] == ("store_size",)
    assert "hist" in captured["columns"]
    assert "futr" in captured["columns"]
    assert "store_size" in captured["columns"]
