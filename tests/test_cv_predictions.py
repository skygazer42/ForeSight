import pandas as pd
import numpy as np
from types import SimpleNamespace

import foresight.cv as cv_mod
from foresight.cv import cross_validation_predictions


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
