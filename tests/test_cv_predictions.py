import pandas as pd
import numpy as np
from types import SimpleNamespace

import foresight.cv as cv_mod
from foresight.cv import cross_validation_predictions
from foresight.models import registry as registry_mod


def _install_dummy_local_xreg_model(*, monkeypatch, key: str) -> str:
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
        description="Test-only local xreg model for CV",
        factory=_factory,
        param_help={"x_cols": "Required future covariate columns"},
        capability_overrides={"requires_future_covariates": True},
    )
    monkeypatch.setitem(registry_mod._REGISTRY, key, spec)
    return key


def test_cv_predictions_catfish_shapes_and_columns():
    df = cross_validation_predictions(
        model="naive-last",
        dataset="catfish",
        y_col="Total",
        horizon=3,
        step_size=3,
        min_train_size=12,
    )
    assert set(df.columns) == {"unique_id", "ds", "cutoff", "step", "y", "yhat", "model"}
    assert (df["step"].min(), df["step"].max()) == (1, 3)

    # catfish has 324 rows -> windows: train_end = 12..321 step 3 => 104 windows; each yields horizon rows
    assert len(df) == 104 * 3

    # naive-last: within each cutoff, yhat repeats the last observed y at cutoff
    grouped = df.groupby("cutoff", sort=False)
    for _cutoff, g in list(grouped)[:3]:
        assert g["yhat"].nunique() == 1


def test_cv_predictions_max_train_size_changes_mean_model():
    df_expanding = cross_validation_predictions(
        model="mean",
        dataset="catfish",
        y_col="Total",
        horizon=1,
        step_size=10,
        min_train_size=12,
        max_train_size=None,
        n_windows=5,
    )
    df_rolling = cross_validation_predictions(
        model="mean",
        dataset="catfish",
        y_col="Total",
        horizon=1,
        step_size=10,
        min_train_size=12,
        max_train_size=12,
        n_windows=5,
    )
    assert len(df_expanding) == len(df_rolling)
    # Rolling mean over last 12 points should differ from expanding mean at least sometimes
    assert np.any(np.abs(df_expanding["yhat"].to_numpy() - df_rolling["yhat"].to_numpy()) > 1e-9)


def test_global_cv_context_cache_reuses_cutoffs_for_same_parameters(monkeypatch):
    long_df = pd.DataFrame(
        {
            "unique_id": ["s1"] * 8,
            "ds": pd.date_range("2020-01-01", periods=8, freq="D"),
            "y": np.arange(8, dtype=float),
        }
    )
    sorted_df = long_df.sort_values(["unique_id", "ds"], kind="mergesort")

    calls = {"count": 0}

    def _fake_global_cv_cutoffs(
        df,
        *,
        horizon,
        step_size,
        min_train_size,
        max_train_size,
        n_windows,
    ):
        calls["count"] += 1
        return ("s1", [df["ds"].iloc[3]])

    monkeypatch.setattr(cv_mod, "_global_cv_cutoffs", _fake_global_cv_cutoffs)

    first = cv_mod._get_cached_global_cv_context(  # type: ignore[attr-defined]
        sorted_df,
        horizon=2,
        step_size=1,
        min_train_size=4,
        max_train_size=None,
        n_windows=1,
    )
    second = cv_mod._get_cached_global_cv_context(  # type: ignore[attr-defined]
        sorted_df,
        horizon=2,
        step_size=1,
        min_train_size=4,
        max_train_size=None,
        n_windows=1,
    )
    third = cv_mod._get_cached_global_cv_context(  # type: ignore[attr-defined]
        sorted_df,
        horizon=2,
        step_size=2,
        min_train_size=4,
        max_train_size=None,
        n_windows=1,
    )

    assert calls["count"] == 2
    assert first is second
    assert first is not third


def test_cross_validation_predictions_long_df_reuses_global_cv_context(monkeypatch):
    long_df = pd.DataFrame(
        {
            "unique_id": ["s1"] * 8 + ["s2"] * 8,
            "ds": list(pd.date_range("2020-01-01", periods=8, freq="D")) * 2,
            "y": np.arange(16, dtype=float),
        }
    )

    original_global_cv_cutoffs = cv_mod._global_cv_cutoffs  # type: ignore[attr-defined]
    calls = {"count": 0}

    def _counting_global_cv_cutoffs(
        df,
        *,
        horizon,
        step_size,
        min_train_size,
        max_train_size,
        n_windows,
    ):
        calls["count"] += 1
        return original_global_cv_cutoffs(
            df,
            horizon=horizon,
            step_size=step_size,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            n_windows=n_windows,
        )

    def _fake_global_forecaster(df, cutoff, horizon):
        rows = []
        for uid, group in df.groupby("unique_id", sort=False):
            future_ds = group.loc[group["ds"] > cutoff, "ds"].head(int(horizon))
            for ds in future_ds:
                rows.append({"unique_id": uid, "ds": ds, "yhat": 0.0})
        return pd.DataFrame(rows)

    monkeypatch.setattr(cv_mod, "_global_cv_cutoffs", _counting_global_cv_cutoffs)
    monkeypatch.setattr(
        cv_mod._model_execution,
        "get_model_spec",
        lambda model: SimpleNamespace(interface="global"),
    )
    monkeypatch.setattr(
        cv_mod._model_execution,
        "make_global_forecaster_runner",
        lambda model, params: _fake_global_forecaster,
    )

    first = cv_mod.cross_validation_predictions_long_df(
        model="fake-global",
        long_df=long_df,
        horizon=2,
        step_size=1,
        min_train_size=4,
        n_windows=1,
    )
    second = cv_mod.cross_validation_predictions_long_df(
        model="fake-global",
        long_df=long_df,
        horizon=2,
        step_size=1,
        min_train_size=4,
        n_windows=1,
    )

    assert calls["count"] == 1
    assert len(first) == len(second) == 4


def test_cross_validation_predictions_long_df_global_path_avoids_concat(monkeypatch):
    long_df = pd.DataFrame(
        {
            "unique_id": ["s1"] * 8 + ["s2"] * 8,
            "ds": list(pd.date_range("2020-01-01", periods=8, freq="D")) * 2,
            "y": np.arange(16, dtype=float),
        }
    )

    def _fake_global_forecaster(df, cutoff, horizon):
        rows = []
        for uid, group in df.groupby("unique_id", sort=False):
            future_ds = group.loc[group["ds"] > cutoff, "ds"].head(int(horizon))
            for ds in future_ds:
                rows.append({"unique_id": uid, "ds": ds, "yhat": 0.0})
        return pd.DataFrame(rows)

    original_concat = pd.concat
    calls = {"count": 0}

    def _counting_concat(*args, **kwargs):
        calls["count"] += 1
        return original_concat(*args, **kwargs)

    monkeypatch.setattr(pd, "concat", _counting_concat)
    monkeypatch.setattr(
        cv_mod._model_execution,
        "get_model_spec",
        lambda model: SimpleNamespace(interface="global"),
    )
    monkeypatch.setattr(
        cv_mod._model_execution,
        "make_global_forecaster_runner",
        lambda model, params: _fake_global_forecaster,
    )

    out = cv_mod.cross_validation_predictions_long_df(
        model="fake-global",
        long_df=long_df,
        horizon=2,
        step_size=1,
        min_train_size=4,
        n_windows=1,
    )

    assert len(out) == 4
    assert calls["count"] == 0


def test_cross_validation_predictions_long_df_global_path_builds_output_frame_once(monkeypatch):
    long_df = pd.DataFrame(
        {
            "unique_id": ["s1"] * 8 + ["s2"] * 8,
            "ds": list(pd.date_range("2020-01-01", periods=8, freq="D")) * 2,
            "y": np.arange(16, dtype=float),
        }
    )

    def _fake_global_forecaster(df, cutoff, horizon):
        rows = []
        for uid, group in df.groupby("unique_id", sort=False):
            future_ds = group.loc[group["ds"] > cutoff, "ds"].head(int(horizon))
            for ds in future_ds:
                rows.append({"unique_id": uid, "ds": ds, "yhat": 0.0})
        return pd.DataFrame(rows)

    original_frame_builder = cv_mod._prediction_frame_from_columns  # type: ignore[attr-defined]
    calls = {"count": 0}

    def _counting_frame_builder(*args, **kwargs):
        calls["count"] += 1
        return original_frame_builder(*args, **kwargs)

    monkeypatch.setattr(cv_mod, "_prediction_frame_from_columns", _counting_frame_builder)
    monkeypatch.setattr(
        cv_mod._model_execution,
        "get_model_spec",
        lambda model: SimpleNamespace(interface="global"),
    )
    monkeypatch.setattr(
        cv_mod._model_execution,
        "make_global_forecaster_runner",
        lambda model, params: _fake_global_forecaster,
    )

    out = cv_mod.cross_validation_predictions_long_df(
        model="fake-global",
        long_df=long_df,
        horizon=2,
        step_size=1,
        min_train_size=4,
        n_windows=1,
    )

    assert len(out) == 4
    assert calls["count"] == 1


def test_prepared_global_cv_frame_avoids_set_index_and_groupby(monkeypatch):
    pred = pd.DataFrame(
        {
            "unique_id": ["s1", "s1", "s2", "s2"],
            "ds": pd.to_datetime(["2020-01-05", "2020-01-06", "2020-01-05", "2020-01-06"]),
            "yhat": [1.0, 2.0, 3.0, 4.0],
            "yhat_p10": [0.5, 1.5, 2.5, 3.5],
        }
    )
    y_lookup = (
        pd.DataFrame(
            {
                "unique_id": ["s1", "s1", "s2", "s2"],
                "ds": pd.to_datetime(["2020-01-05", "2020-01-06", "2020-01-05", "2020-01-06"]),
                "y": [10.0, 11.0, 12.0, 13.0],
            }
        )
        .set_index(["unique_id", "ds"])["y"]
    )

    original_set_index = pd.DataFrame.set_index
    original_groupby = pd.DataFrame.groupby
    original_sort_values = pd.DataFrame.sort_values
    calls = {"set_index": 0, "groupby": 0, "sort_values": 0}

    def _counting_set_index(self, *args, **kwargs):
        calls["set_index"] += 1
        return original_set_index(self, *args, **kwargs)

    def _counting_groupby(self, *args, **kwargs):
        calls["groupby"] += 1
        return original_groupby(self, *args, **kwargs)

    def _counting_sort_values(self, *args, **kwargs):
        calls["sort_values"] += 1
        return original_sort_values(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "set_index", _counting_set_index)
    monkeypatch.setattr(pd.DataFrame, "groupby", _counting_groupby)
    monkeypatch.setattr(pd.DataFrame, "sort_values", _counting_sort_values)

    frame, skipped_here, pred_cols = cv_mod._prepared_global_cv_frame(  # type: ignore[attr-defined]
        pred,
        y_lookup=y_lookup,
        horizon=2,
        total_series=2,
        cutoff=pd.Timestamp("2020-01-04"),
        model="fake-global",
    )

    assert frame is not None
    assert skipped_here == 0
    assert pred_cols == ("yhat", "yhat_p10")
    assert list(frame.columns) == ["unique_id", "ds", "cutoff", "step", "y", "yhat", "yhat_p10", "model"]
    assert frame["step"].tolist() == [1, 2, 1, 2]
    assert calls["set_index"] == 0
    assert calls["groupby"] == 0
    assert calls["sort_values"] == 0


def test_cross_validation_predictions_long_df_supports_local_future_x_cols(monkeypatch) -> None:
    key = _install_dummy_local_xreg_model(
        monkeypatch=monkeypatch,
        key="__test-local-xreg-cv__",
    )

    promo = ([0, 1, 0, 1, 0, 1] * 5)[:30]
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 30,
            "ds": pd.date_range("2020-01-01", periods=30, freq="D"),
            "y": [float(p) for p in promo],
            "promo": [float(p) for p in promo],
        }
    )

    out = cv_mod.cross_validation_predictions_long_df(
        model=key,
        long_df=long_df,
        horizon=3,
        step_size=3,
        min_train_size=12,
        model_params={"future_x_cols": ("promo",)},
    )

    assert set(out.columns) == {"unique_id", "ds", "cutoff", "step", "y", "yhat", "model"}
    assert out["model"].eq(key).all()
    assert out["step"].min() == 1
    assert out["step"].max() == 3
    assert np.allclose(out["y"].to_numpy(dtype=float), out["yhat"].to_numpy(dtype=float))


def test_cross_validation_predictions_long_df_preserves_datetime_ds_for_local_models(
    monkeypatch,
) -> None:
    key = _install_dummy_local_xreg_model(
        monkeypatch=monkeypatch,
        key="__test-local-xreg-cv-datetime__",
    )

    promo = ([0.0, 1.0] * 15)[:30]
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 30,
            "ds": pd.date_range("2020-01-01", periods=30, freq="D"),
            "y": promo,
            "promo": promo,
        }
    )

    out = cv_mod.cross_validation_predictions_long_df(
        model=key,
        long_df=long_df,
        horizon=3,
        step_size=3,
        min_train_size=12,
        model_params={"future_x_cols": ("promo",)},
    )

    assert pd.api.types.is_datetime64_any_dtype(out["ds"])
    assert out["ds"].iloc[0] == pd.Timestamp("2020-01-13")
    assert out["ds"].iloc[-1] == pd.Timestamp("2020-01-30")


def test_cross_validation_predictions_long_df_rejects_local_historic_x_cols(monkeypatch) -> None:
    key = _install_dummy_local_xreg_model(
        monkeypatch=monkeypatch,
        key="__test-local-historic-x-cv__",
    )

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 20,
            "ds": pd.date_range("2020-01-01", periods=20, freq="D"),
            "y": [float(i % 2) for i in range(20)],
            "promo_hist": [float(i % 2) for i in range(20)],
            "promo": [float(i % 2) for i in range(20)],
        }
    )

    try:
        cv_mod.cross_validation_predictions_long_df(
            model=key,
            long_df=long_df,
            horizon=2,
            step_size=2,
            min_train_size=8,
            model_params={"historic_x_cols": ("promo_hist",), "future_x_cols": ("promo",)},
        )
    except ValueError as exc:
        assert "historic_x_cols are not yet supported" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported local historic_x_cols")
