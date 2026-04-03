from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


def test_prepare_promotion_long_df_regularizes_weekly_grid_and_zero_fills() -> None:
    from foresight.datasets import load_dataset
    from foresight.services.model_validation import prepare_promotion_long_df

    raw = load_dataset("promotion_data")
    out = prepare_promotion_long_df(max_series=None)

    assert list(out.columns[:3]) == ["unique_id", "ds", "y"]
    assert not out.empty

    raw_uid = "store=42|dept=32"
    raw_count = int(((raw["store"] == 42) & (raw["dept"] == 32)).sum())
    group = out.loc[out["unique_id"] == raw_uid].reset_index(drop=True)

    assert len(group) > raw_count
    assert (group["ds"].diff().dropna() == pd.Timedelta(days=7)).all()
    assert float(group["y"].eq(0.0).sum()) > 0.0
    assert {
        "time_month_sin",
        "time_month_cos",
        "time_week_sin",
        "time_week_cos",
    }.issubset(set(out.columns))


def test_prepare_promotion_long_df_limits_series_deterministically() -> None:
    from foresight.services.model_validation import prepare_promotion_long_df

    full = prepare_promotion_long_df(max_series=None)
    out = prepare_promotion_long_df(max_series=16)

    expected = (
        full.groupby("unique_id", sort=False)["y"]
        .agg(
            n_obs="size",
            n_nonzero=lambda s: int((s != 0).sum()),
            n_unique="nunique",
            y_std="std",
        )
        .reset_index()
        .sort_values(
            ["n_nonzero", "n_unique", "y_std", "unique_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )["unique_id"]
        .head(16)
        .tolist()
    )

    assert out["unique_id"].nunique() == 16
    assert sorted(out["unique_id"].unique().tolist()) == sorted(expected)


def test_build_multivariate_validation_frame_selects_top_four_series_and_zero_fills() -> None:
    from foresight.services.model_validation import (
        build_promotion_multivariate_wide_df,
        prepare_promotion_long_df,
    )

    long_df = prepare_promotion_long_df()
    wide_df, target_cols = build_promotion_multivariate_wide_df(long_df)

    counts = (
        long_df.groupby("unique_id", sort=False)["y"]
        .agg(
            n_obs="size",
            n_nonzero=lambda s: int((s != 0).sum()),
            n_unique="nunique",
            y_std="std",
        )
        .reset_index()
    )
    expected = sorted(
        counts.sort_values(
            ["n_nonzero", "n_unique", "y_std", "unique_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )["unique_id"].head(4)
    )

    assert len(target_cols) == 4
    assert sorted(target_cols) == expected
    assert list(wide_df.columns) == ["ds", *target_cols]
    assert not wide_df.loc[:, target_cols].isna().any().any()


def test_limit_series_reuses_ranked_unique_ids_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    from foresight.services.model_validation import _limit_series

    long_df = pd.DataFrame(
        {
            "unique_id": ["b", "b", "a", "a", "c", "c"],
            "ds": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-01",
                    "2020-01-08",
                ]
            ),
            "y": [5.0, 6.0, 1.0, 2.0, 9.0, 10.0],
        }
    )
    long_df.attrs["_validation_ranked_unique_ids"] = ["c", "b", "a"]

    original_groupby = pd.DataFrame.groupby

    def _forbid_groupby(self, *args, **kwargs):
        raise AssertionError("ranked unique ids attr should avoid groupby")

    monkeypatch.setattr(pd.DataFrame, "groupby", _forbid_groupby)
    try:
        out = _limit_series(long_df, max_series=2)
    finally:
        monkeypatch.setattr(pd.DataFrame, "groupby", original_groupby)

    assert out["unique_id"].unique().tolist() == ["b", "c"]
    assert out.attrs["_validation_ranked_unique_ids"] == ["c", "b", "a"]


def test_build_multivariate_validation_frame_reuses_ranked_unique_ids_attr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from foresight.services.model_validation import build_promotion_multivariate_wide_df

    long_df = pd.DataFrame(
        {
            "unique_id": ["b", "b", "a", "a", "c", "c"],
            "ds": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-01",
                    "2020-01-08",
                ]
            ),
            "y": [5.0, 6.0, 1.0, 2.0, 9.0, 10.0],
        }
    )
    long_df.attrs["_validation_ranked_unique_ids"] = ["c", "b", "a"]

    original_groupby = pd.DataFrame.groupby

    def _forbid_groupby(self, *args, **kwargs):
        raise AssertionError("ranked unique ids attr should avoid groupby")

    monkeypatch.setattr(pd.DataFrame, "groupby", _forbid_groupby)
    try:
        wide_df, target_cols = build_promotion_multivariate_wide_df(long_df, n_series=2)
    finally:
        monkeypatch.setattr(pd.DataFrame, "groupby", original_groupby)

    assert target_cols == ["b", "c"]
    assert list(wide_df.columns) == ["ds", "b", "c"]


def test_build_multivariate_validation_frame_caches_wide_result_for_same_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from foresight.services import model_validation as mod

    long_df = pd.DataFrame(
        {
            "unique_id": ["b", "b", "a", "a", "c", "c"],
            "ds": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-01",
                    "2020-01-08",
                ]
            ),
            "y": [5.0, 6.0, 1.0, 2.0, 9.0, 10.0],
        }
    )
    long_df.attrs["_validation_ranked_unique_ids"] = ["c", "b", "a"]

    first_wide, first_targets = mod.build_promotion_multivariate_wide_df(long_df, n_series=2)

    def _forbid_long_to_wide(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("cached multivariate validation frame should avoid long_to_wide")

    monkeypatch.setattr(mod, "long_to_wide", _forbid_long_to_wide)

    second_wide, second_targets = mod.build_promotion_multivariate_wide_df(long_df, n_series=2)

    assert second_wide is first_wide
    assert second_targets == first_targets == ["b", "c"]


def test_promotion_validation_bundle_cache_reuses_prepared_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from foresight.services import model_validation as mod

    counts = {"prepare": 0, "wide": 0}

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s1", "s1"],
            "ds": pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-06", "2020-01-13"]),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    wide_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2020-01-06", "2020-01-13"]),
            "s0": [1.0, 2.0],
            "s1": [3.0, 4.0],
        }
    )

    def _fake_prepare_promotion_long_df(
        data_dir: str | Path | None = None,
        *,
        max_series: int | None = mod._VALIDATION_MAX_SERIES,
    ) -> pd.DataFrame:
        counts["prepare"] += 1
        return long_df

    def _fake_build_promotion_multivariate_wide_df(
        frame: pd.DataFrame,
        *,
        n_series: int = mod._MULTIVARIATE_N_SERIES,
    ) -> tuple[pd.DataFrame, list[str]]:
        assert frame is long_df
        counts["wide"] += 1
        return wide_df, ["s0", "s1"]

    cache_before = dict(getattr(mod, "_PROMOTION_VALIDATION_BUNDLE_CACHE", {}))
    if hasattr(mod, "_PROMOTION_VALIDATION_BUNDLE_CACHE"):
        mod._PROMOTION_VALIDATION_BUNDLE_CACHE.clear()
    monkeypatch.setattr(mod, "prepare_promotion_long_df", _fake_prepare_promotion_long_df)
    monkeypatch.setattr(
        mod,
        "build_promotion_multivariate_wide_df",
        _fake_build_promotion_multivariate_wide_df,
    )

    try:
        first = mod._build_promotion_validation_bundle()  # type: ignore[attr-defined]
        second = mod._build_promotion_validation_bundle()  # type: ignore[attr-defined]
    finally:
        if hasattr(mod, "_PROMOTION_VALIDATION_BUNDLE_CACHE"):
            mod._PROMOTION_VALIDATION_BUNDLE_CACHE.clear()
            mod._PROMOTION_VALIDATION_BUNDLE_CACHE.update(cache_before)

    assert counts == {"prepare": 1, "wide": 1}
    assert first["long_df"] is second["long_df"]
    assert first["wide_df"] is second["wide_df"]
    assert first["target_cols"] == second["target_cols"] == ["s0", "s1"]


def test_build_model_params_applies_lightweight_supported_overrides() -> None:
    from foresight.services.model_validation import build_model_params

    @dataclass(frozen=True)
    class _Spec:
        key: str
        interface: str = "local"
        requires: tuple[str, ...] = ("torch",)
        default_params: dict[str, Any] = field(
            default_factory=lambda: {
                "device": "cpu",
                "epochs": 50,
                "batch_size": 64,
                "seed": 123,
                "patience": 10,
                "warmup_epochs": 5,
                "ema_warmup_epochs": 4,
                "swa_start_epoch": 7,
                "min_epochs": 3,
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_iter": 1000,
                "random_state": 11,
            }
        )
        param_help: dict[str, str] = field(default_factory=dict)

    params = build_model_params(_Spec(key="demo"), device="cuda")

    assert params["device"] == "cuda"
    assert params["epochs"] == 1
    assert params["batch_size"] == 32
    assert params["seed"] == 0
    assert params["patience"] == 2
    assert params["warmup_epochs"] == 0
    assert params["ema_warmup_epochs"] == 0
    assert params["swa_start_epoch"] == -1
    assert params["min_epochs"] == 1
    assert params["n_estimators"] == 20
    assert params["learning_rate"] == 0.1
    assert params["max_iter"] == 200
    assert params["random_state"] == 0


def test_build_model_params_applies_stats_validation_overrides() -> None:
    from foresight.services.model_validation import build_model_params

    @dataclass(frozen=True)
    class _Spec:
        key: str
        interface: str = "local"
        requires: tuple[str, ...] = ("stats",)
        default_params: dict[str, Any] = field(default_factory=dict)
        param_help: dict[str, str] = field(default_factory=dict)

    autoreg = _Spec(
        key="autoreg",
        default_params={"lags": 12, "trend": "c"},
        param_help={"lags": "", "trend": ""},
    )
    ets = _Spec(
        key="ets",
        default_params={"season_length": 12, "seasonal": "add"},
        param_help={"season_length": "", "seasonal": ""},
    )
    fourier_autoreg = _Spec(
        key="fourier-autoreg",
        default_params={"periods": (12,), "orders": 2, "lags": 0},
        param_help={"periods": "", "orders": "", "lags": ""},
    )
    mstl_arima = _Spec(
        key="mstl-arima",
        default_params={"periods": (12,), "order": (1, 0, 0)},
        param_help={"periods": "", "order": ""},
    )
    crossformer = _Spec(
        key="torch-crossformer-deep-global",
        requires=("torch",),
        interface="global",
        default_params={"context_length": 8, "segment_len": 16, "stride": 16},
        param_help={"context_length": "", "segment_len": "", "stride": ""},
    )
    patchtst = _Spec(
        key="torch-patchtst-global",
        requires=("torch",),
        interface="global",
        default_params={"context_length": 8, "patch_len": 16, "stride": 16},
        param_help={"context_length": "", "patch_len": "", "stride": ""},
    )
    pyraformer = _Spec(
        key="torch-pyraformer-global",
        requires=("torch",),
        interface="global",
        default_params={"context_length": 8, "segment_len": 16, "stride": 16},
        param_help={"context_length": "", "segment_len": "", "stride": ""},
    )
    seasonal_drift = _Spec(
        key="seasonal-drift",
        requires=(),
        default_params={"season_length": 12},
        param_help={"season_length": ""},
    )
    sar_ols = _Spec(
        key="sar-ols",
        requires=(),
        default_params={"p": 1, "P": 1, "season_length": 12},
        param_help={"p": "", "P": "", "season_length": ""},
    )
    ssa = _Spec(
        key="ssa",
        requires=(),
        default_params={"window_length": 24, "rank": 5},
        param_help={"window_length": "", "rank": ""},
    )
    hw_add = _Spec(
        key="holt-winters-add",
        requires=(),
        default_params={"season_length": 12, "trend": "add", "seasonal": "add"},
        param_help={"season_length": "", "trend": "", "seasonal": ""},
    )
    knn = _Spec(
        key="knn-lag",
        requires=("ml",),
        default_params={"lags": 12, "n_neighbors": 10},
        param_help={"lags": "", "n_neighbors": ""},
    )
    hf_tst = _Spec(
        key="hf-timeseries-transformer-direct",
        requires=("transformers", "torch"),
        default_params={"context_length": 48, "lags_sequence": (1, 2, 3, 4, 5, 6, 7)},
        param_help={"context_length": "", "lags_sequence": ""},
    )
    lstnet = _Spec(
        key="torch-lstnet-direct",
        requires=("torch",),
        default_params={"highway_window": 24},
        param_help={"highway_window": ""},
    )
    segrnn = _Spec(
        key="torch-segrnn-direct",
        requires=("torch",),
        default_params={"segment_len": 12, "lags": 96},
        param_help={"segment_len": "", "lags": ""},
    )
    timexer = _Spec(
        key="torch-timexer-direct",
        requires=("torch",),
        default_params={"x_cols": (), "context_length": 96},
        param_help={"x_cols": "", "context_length": ""},
    )

    autoreg_params = build_model_params(autoreg, device="cpu")
    ets_params = build_model_params(ets, device="cpu")
    fourier_autoreg_params = build_model_params(fourier_autoreg, device="cpu")
    mstl_arima_params = build_model_params(mstl_arima, device="cpu")
    crossformer_params = build_model_params(crossformer, device="cpu")
    patchtst_params = build_model_params(patchtst, device="cpu")
    pyraformer_params = build_model_params(pyraformer, device="cpu")
    seasonal_drift_params = build_model_params(seasonal_drift, device="cpu")
    sar_ols_params = build_model_params(sar_ols, device="cpu")
    ssa_params = build_model_params(ssa, device="cpu")
    hw_add_params = build_model_params(hw_add, device="cpu")
    hf_tst_params = build_model_params(hf_tst, device="cpu")
    knn_params = build_model_params(knn, device="cpu")
    lstnet_params = build_model_params(lstnet, device="cpu")
    segrnn_params = build_model_params(segrnn, device="cpu")
    timexer_params = build_model_params(timexer, device="cpu")

    assert autoreg_params["lags"] == 4
    assert ets_params["season_length"] == 4
    assert fourier_autoreg_params["periods"] == (4,)
    assert fourier_autoreg_params["orders"] == 1
    assert fourier_autoreg_params["lags"] == 2
    assert mstl_arima_params["periods"] == (4,)
    assert crossformer_params["segment_len"] == 8
    assert crossformer_params["stride"] == 8
    assert patchtst_params["patch_len"] == 8
    assert patchtst_params["stride"] == 8
    assert pyraformer_params["segment_len"] == 8
    assert pyraformer_params["stride"] == 8
    assert seasonal_drift_params["season_length"] == 4
    assert sar_ols_params["season_length"] == 4
    assert ssa_params["window_length"] == 8
    assert hw_add_params["season_length"] == 4
    assert hf_tst_params["context_length"] == 4
    assert hf_tst_params["lags_sequence"] == (1, 2, 3)
    assert knn_params["n_neighbors"] == 2
    assert lstnet_params["highway_window"] == 8
    assert segrnn_params["segment_len"] == 8
    assert timexer_params["x_cols"] == (
        "time_month_sin",
        "time_month_cos",
        "time_week_sin",
        "time_week_cos",
    )


def test_build_model_params_configures_foundation_wrappers_for_fixture_mode(tmp_path: Path) -> None:
    from foresight.services.model_validation import build_model_params

    @dataclass(frozen=True)
    class _Spec:
        key: str = "lag-llama"
        interface: str = "local"
        requires: tuple[str, ...] = ()
        default_params: dict[str, Any] = field(
            default_factory=lambda: {
                "backend": "auto",
                "checkpoint_path": "",
                "model_source": "",
                "local_files_only": True,
                "device": "cpu",
                "seed": 123,
                "context_length": 48,
                "num_samples": 100,
            }
        )
        param_help: dict[str, str] = field(default_factory=dict)

    fixture_path = tmp_path / "foundation.json"
    fixture_path.write_text('{"bias": 1.5, "scale": 1.0, "use_trend": true}', encoding="utf-8")

    params = build_model_params(_Spec(), device="cpu", foundation_fixture_path=fixture_path)

    assert params["backend"] == "fixture-json"
    assert params["checkpoint_path"] == str(fixture_path)
    assert params["local_files_only"] is True
    assert params["device"] == "cpu"


def test_transform_validation_long_df_for_gamma_makes_targets_strictly_positive() -> None:
    from foresight.services.model_validation import transform_validation_long_df_for_model

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s1", "s1"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-01", "2020-01-08"]),
            "y": [0.0, 2.0, 0.0, 3.0],
        }
    )

    out = transform_validation_long_df_for_model(long_df, model="gamma-lag")

    assert (out["y"] > 0.0).all()
    assert out["y"].iloc[1] > out["y"].iloc[0]
    assert float(out["y"].max()) <= 1.0


def test_transform_validation_long_df_for_poisson_scales_targets_to_stable_positive_range() -> None:
    from foresight.services.model_validation import transform_validation_long_df_for_model

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s1", "s1"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-01", "2020-01-08"]),
            "y": [0.0, 2000.0, 5.0, 10.0],
        }
    )

    out = transform_validation_long_df_for_model(long_df, model="poisson-lag")

    assert (out["y"] >= 0.0).all()
    assert float(out["y"].max()) <= 1.0


def test_transform_validation_long_df_for_multiplicative_holt_winters_makes_targets_positive() -> (
    None
):
    from foresight.services.model_validation import transform_validation_long_df_for_model

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s1", "s1"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-01", "2020-01-08"]),
            "y": [0.0, 20.0, 5.0, 10.0],
        }
    )

    out = transform_validation_long_df_for_model(long_df, model="holt-winters-mul")

    assert (out["y"] > 0.0).all()
    assert float(out["y"].max()) <= 1.0


def test_transform_validation_long_df_for_logistic_scales_each_series_to_unit_interval() -> None:
    from foresight.services.model_validation import transform_validation_long_df_for_model

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s1", "s1"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-01", "2020-01-08"]),
            "y": [0.0, 20.0, 5.0, 10.0],
        }
    )

    out = transform_validation_long_df_for_model(long_df, model="xgb-logistic-lag")

    assert ((out["y"] >= 0.0) & (out["y"] <= 1.0)).all()
    assert (
        float(
            out.loc[
                (out["unique_id"] == "s0") & (out["ds"] == pd.Timestamp("2020-01-08")), "y"
            ].iloc[0]
        )
        == 1.0
    )


def test_transform_validation_long_df_for_catboost_breaks_constant_targets_per_series() -> None:
    from foresight.services.model_validation import transform_validation_long_df_for_model

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s0", "s1", "s1", "s1"],
            "ds": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-15",
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-15",
                ]
            ),
            "y": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
        }
    )

    out = transform_validation_long_df_for_model(long_df, model="catboost-lag")

    assert out.loc[out["unique_id"] == "s0", "y"].nunique() == 3
    assert out.loc[out["unique_id"] == "s1", "y"].nunique() == 3
    assert float(out["y"].min()) >= 0.0


def test_transform_validation_long_df_reuses_cached_family_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from foresight.services.model_validation import transform_validation_long_df_for_model

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s1", "s1"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-01", "2020-01-08"]),
            "y": [0.0, 20.0, 5.0, 10.0],
        }
    )

    first = transform_validation_long_df_for_model(long_df, model="xgb-logistic-lag")

    original_groupby = pd.DataFrame.groupby

    def _forbid_groupby(self, *args, **kwargs):
        raise AssertionError("cached family transform should avoid groupby")

    monkeypatch.setattr(pd.DataFrame, "groupby", _forbid_groupby)
    try:
        second = transform_validation_long_df_for_model(long_df, model="catboost-logistic-lag")
    finally:
        monkeypatch.setattr(pd.DataFrame, "groupby", original_groupby)

    assert second is first


def test_run_registry_validation_writes_outputs_and_counts_failures(
    monkeypatch, tmp_path: Path
) -> None:
    from foresight.services import model_validation as mod

    @dataclass(frozen=True)
    class _Spec:
        key: str
        interface: str
        requires: tuple[str, ...] = ()
        default_params: dict[str, Any] = field(default_factory=dict)
        param_help: dict[str, str] = field(default_factory=dict)

    raw = pd.DataFrame(
        {
            "store": [1, 1, 1, 1, 2, 2, 2, 2],
            "dept": [1, 1, 2, 2, 1, 1, 2, 2],
            "week": pd.to_datetime(
                [
                    "2020-01-06",
                    "2020-01-20",
                    "2020-01-06",
                    "2020-01-20",
                    "2020-01-06",
                    "2020-01-20",
                    "2020-01-06",
                    "2020-01-20",
                ]
            ),
            "promotion_sales": [1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0],
        }
    )

    def _fake_load_dataset(key: str, data_dir: str | None = None) -> pd.DataFrame:
        assert key == "promotion_data"
        return raw.copy()

    def _fake_list_models() -> list[str]:
        return ["ok-local", "ok-global", "bad-multivariate"]

    def _fake_get_model_spec(key: str) -> _Spec:
        mapping = {
            "ok-local": _Spec(key="ok-local", interface="local"),
            "ok-global": _Spec(key="ok-global", interface="global"),
            "bad-multivariate": _Spec(key="bad-multivariate", interface="multivariate"),
        }
        return mapping[key]

    def _fake_eval_model_long_df(
        *,
        model: str,
        long_df: pd.DataFrame,
        horizon: int,
        step: int,
        min_train_size: int,
        max_windows: int | None,
        model_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert not long_df.empty
        assert horizon == 3
        assert step == 3
        assert min_train_size == 12
        assert max_windows == 8
        return {
            "model": model,
            "n_points": 12,
            "mae": 1.0,
            "rmse": 2.0,
            "mape": 3.0,
            "smape": 4.0,
        }

    def _fake_eval_multivariate_model_df(
        *,
        model: str,
        df: pd.DataFrame,
        target_cols: list[str],
        horizon: int,
        step: int,
        min_train_size: int,
        model_params: dict[str, Any] | None = None,
        ds_col: str | None = "ds",
        max_windows: int | None = None,
        max_train_size: int | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"{model} exploded")

    monkeypatch.setattr(mod, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(mod, "list_models", _fake_list_models)
    monkeypatch.setattr(mod, "get_model_spec", _fake_get_model_spec)
    monkeypatch.setattr(mod, "eval_model_long_df", _fake_eval_model_long_df)
    monkeypatch.setattr(mod, "eval_multivariate_model_df", _fake_eval_multivariate_model_df)
    monkeypatch.setattr(mod, "resolve_runtime_device", lambda device: "cpu")

    payload = mod.run_registry_validation(output_dir=tmp_path)

    assert payload["summary"]["total_models"] == 3
    assert payload["summary"]["ok_models"] == 2
    assert payload["summary"]["failed_models"] == 1
    assert {row["status"] for row in payload["rows"]} == {"ok", "error"}
    assert (tmp_path / "rows.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()

    rows = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
    assert len(rows) == 3
    assert any(row["model"] == "bad-multivariate" and row["status"] == "error" for row in rows)


def test_run_registry_validation_marks_infinite_metrics_as_error(
    monkeypatch, tmp_path: Path
) -> None:
    from foresight.services import model_validation as mod

    @dataclass(frozen=True)
    class _Spec:
        key: str = "ok-local"
        interface: str = "local"
        requires: tuple[str, ...] = ()
        default_params: dict[str, Any] = field(default_factory=dict)
        param_help: dict[str, str] = field(default_factory=dict)

    raw = pd.DataFrame(
        {
            "store": [1, 1, 1, 1],
            "dept": [1, 1, 1, 1],
            "week": pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-20", "2020-01-27"]),
            "promotion_sales": [1.0, 2.0, 3.0, 4.0],
        }
    )

    monkeypatch.setattr(mod, "load_dataset", lambda key, data_dir=None: raw.copy())
    monkeypatch.setattr(mod, "list_models", lambda: ["ok-local"])
    monkeypatch.setattr(mod, "get_model_spec", lambda key: _Spec())
    monkeypatch.setattr(mod, "resolve_runtime_device", lambda device: "cpu")
    monkeypatch.setattr(
        mod,
        "build_promotion_multivariate_wide_df",
        lambda long_df: (
            pd.DataFrame(
                {
                    "ds": pd.date_range("2020-01-01", periods=4, freq="W-MON"),
                    "a": [1.0, 2.0, 3.0, 4.0],
                    "b": [2.0, 3.0, 4.0, 5.0],
                    "c": [3.0, 4.0, 5.0, 6.0],
                    "d": [4.0, 5.0, 6.0, 7.0],
                }
            ),
            ["a", "b", "c", "d"],
        ),
    )
    monkeypatch.setattr(
        mod,
        "eval_model_long_df",
        lambda **kwargs: {
            "model": "ok-local",
            "n_points": 12,
            "mae": 1.0,
            "rmse": float("inf"),
            "mape": 3.0,
            "smape": 4.0,
        },
    )
    monkeypatch.setattr(
        mod,
        "eval_multivariate_model_df",
        lambda **kwargs: {
            "model": "unused",
            "n_points": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "smape": 0.0,
        },
    )

    payload = mod.run_registry_validation(output_dir=tmp_path)

    assert payload["summary"]["ok_models"] == 0
    assert payload["summary"]["failed_models"] == 1
    assert payload["rows"][0]["status"] == "error"


def test_evaluate_model_multivariate_reuses_base_wide_frame_when_transform_is_noop(
    monkeypatch, tmp_path: Path
) -> None:
    from foresight.services import model_validation as mod

    @dataclass(frozen=True)
    class _Spec:
        key: str = "var"
        interface: str = "multivariate"
        requires: tuple[str, ...] = ("stats",)
        default_params: dict[str, Any] = field(default_factory=dict)
        param_help: dict[str, str] = field(default_factory=dict)

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s1", "s1"],
            "ds": pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-06", "2020-01-13"]),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    wide_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2020-01-06", "2020-01-13"]),
            "s0": [1.0, 2.0],
            "s1": [3.0, 4.0],
        }
    )

    def _forbid_rebuild(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no-op multivariate validation should reuse provided wide frame")

    captured: dict[str, Any] = {}

    def _fake_eval_multivariate_model_df(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "model": kwargs["model"],
            "n_points": 6,
            "mae": 1.0,
            "rmse": 2.0,
            "mape": 3.0,
            "smape": 4.0,
        }

    monkeypatch.setattr(mod, "build_promotion_multivariate_wide_df", _forbid_rebuild)
    monkeypatch.setattr(mod, "eval_multivariate_model_df", _fake_eval_multivariate_model_df)
    monkeypatch.setattr(mod, "build_model_params", lambda *args, **kwargs: {})

    row = mod._evaluate_model(
        model="var",
        spec=_Spec(),
        long_df=long_df,
        wide_df=wide_df,
        target_cols=["s0", "s1"],
        device="cpu",
        foundation_fixture_path=tmp_path / "fixture.json",
    )

    assert row["status"] == "ok"
    assert captured["df"] is wide_df
    assert captured["target_cols"] == ["s0", "s1"]


def test_train_multivariate_model_reuses_base_wide_frame_when_transform_is_noop(
    monkeypatch, tmp_path: Path
) -> None:
    from foresight.services import model_validation as mod

    @dataclass(frozen=True)
    class _Spec:
        key: str = "var"
        interface: str = "multivariate"
        requires: tuple[str, ...] = ("stats",)
        default_params: dict[str, Any] = field(default_factory=dict)
        param_help: dict[str, str] = field(default_factory=dict)

    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s1", "s1"],
            "ds": pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-06", "2020-01-13"]),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    wide_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2020-01-06", "2020-01-13"]),
            "s0": [1.0, 2.0],
            "s1": [3.0, 4.0],
        }
    )
    artifact_paths = mod.build_training_artifact_paths(output_dir=tmp_path, model="var")

    def _forbid_rebuild(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no-op multivariate training should reuse provided wide frame")

    captured: dict[str, Any] = {}

    def _fake_persist_var_artifact(**kwargs: Any) -> Path:
        captured.update(kwargs)
        artifact_path = kwargs["artifact_path"]
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("var", encoding="utf-8")
        return artifact_path

    monkeypatch.setattr(mod, "build_promotion_multivariate_wide_df", _forbid_rebuild)
    monkeypatch.setattr(mod, "_persist_var_artifact", _fake_persist_var_artifact)
    monkeypatch.setattr(mod, "build_training_model_params", lambda *args, **kwargs: {})

    row = mod._train_multivariate_model(
        model="var",
        spec=_Spec(),
        long_df=long_df,
        wide_df=wide_df,
        target_cols=["s0", "s1"],
        device="cpu",
        artifact_paths=artifact_paths,
        foundation_fixture_path=tmp_path / "fixture.json",
    )

    assert row["status"] == "ok"
    assert captured["wide_df"] is wide_df
    assert captured["target_cols"] == ["s0", "s1"]


def test_build_training_artifact_paths_returns_expected_layout(tmp_path: Path) -> None:
    from foresight.services.model_validation import build_training_artifact_paths

    paths = build_training_artifact_paths(output_dir=tmp_path, model="torch-mlp-direct")

    assert paths["model_dir"] == tmp_path / "models" / "torch-mlp-direct"
    assert paths["result_path"] == tmp_path / "models" / "torch-mlp-direct" / "result.json"
    assert paths["forecast_artifact_path"] == (
        tmp_path / "models" / "torch-mlp-direct" / "forecast_artifact.pkl"
    )
    assert paths["checkpoints_dir"] == tmp_path / "models" / "torch-mlp-direct" / "checkpoints"
    assert paths["best_checkpoint_path"] == (
        tmp_path / "models" / "torch-mlp-direct" / "checkpoints" / "best.pt"
    )
    assert paths["last_checkpoint_path"] == (
        tmp_path / "models" / "torch-mlp-direct" / "checkpoints" / "last.pt"
    )
    assert paths["var_artifact_path"] == tmp_path / "models" / "torch-mlp-direct" / "var.pkl"


def test_build_training_model_params_enables_torch_checkpoints(tmp_path: Path) -> None:
    from foresight.services.model_validation import (
        build_training_artifact_paths,
        build_training_model_params,
    )

    @dataclass(frozen=True)
    class _Spec:
        key: str = "torch-demo"
        interface: str = "local"
        requires: tuple[str, ...] = ("torch",)
        default_params: dict[str, Any] = field(
            default_factory=lambda: {
                "device": "cpu",
                "epochs": 99,
                "checkpoint_dir": "",
                "save_best_checkpoint": False,
                "save_last_checkpoint": False,
                "resume_checkpoint_path": "",
            }
        )
        param_help: dict[str, str] = field(
            default_factory=lambda: {
                "device": "",
                "epochs": "",
                "checkpoint_dir": "",
                "save_best_checkpoint": "",
                "save_last_checkpoint": "",
                "resume_checkpoint_path": "",
            }
        )

    artifact_paths = build_training_artifact_paths(output_dir=tmp_path, model="torch-demo")
    params = build_training_model_params(
        _Spec(),
        device="cuda",
        artifact_paths=artifact_paths,
    )

    assert params["device"] == "cuda"
    assert params["epochs"] == 1
    assert params["checkpoint_dir"] == str(artifact_paths["checkpoints_dir"])
    assert params["save_best_checkpoint"] is True
    assert params["save_last_checkpoint"] is True
    assert params["resume_checkpoint_path"] == ""


def test_write_training_progress_updates_progress_json(tmp_path: Path) -> None:
    from foresight.services.model_validation import write_training_progress

    path = write_training_progress(
        output_dir=tmp_path,
        completed_models=50,
        total_models=1265,
        ok_models=47,
        failed_models=3,
        last_model="torch-mlp-direct",
    )

    assert path == tmp_path / "progress.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["completed_models"] == 50
    assert payload["total_models"] == 1265
    assert payload["ok_models"] == 47
    assert payload["failed_models"] == 3
    assert payload["last_model"] == "torch-mlp-direct"


def test_run_registry_training_validation_writes_rows_summary_progress_and_per_model_results(
    monkeypatch, tmp_path: Path
) -> None:
    from foresight.services import model_validation as mod

    @dataclass(frozen=True)
    class _Spec:
        key: str
        interface: str
        requires: tuple[str, ...] = ()
        default_params: dict[str, Any] = field(default_factory=dict)
        param_help: dict[str, str] = field(default_factory=dict)

    raw = pd.DataFrame(
        {
            "store": [1, 1, 1, 1, 2, 2, 2, 2],
            "dept": [1, 1, 2, 2, 1, 1, 2, 2],
            "week": pd.to_datetime(
                [
                    "2020-01-06",
                    "2020-01-20",
                    "2020-01-06",
                    "2020-01-20",
                    "2020-01-06",
                    "2020-01-20",
                    "2020-01-06",
                    "2020-01-20",
                ]
            ),
            "promotion_sales": [1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0],
        }
    )

    def _fake_load_dataset(key: str, data_dir: str | None = None) -> pd.DataFrame:
        assert key == "promotion_data"
        return raw.copy()

    def _fake_list_models() -> list[str]:
        return ["ok-local", "ok-global", "bad-multivariate"]

    def _fake_get_model_spec(key: str) -> _Spec:
        mapping = {
            "ok-local": _Spec(key="ok-local", interface="local"),
            "ok-global": _Spec(key="ok-global", interface="global"),
            "bad-multivariate": _Spec(key="bad-multivariate", interface="multivariate"),
        }
        return mapping[key]

    def _fake_train_single_model(
        *,
        model: str,
        spec: Any,
        long_df: pd.DataFrame,
        wide_df: pd.DataFrame,
        target_cols: list[str],
        device: str,
        output_dir: str | Path,
        artifact_paths: dict[str, Path],
        foundation_fixture_path: str | Path | None,
    ) -> dict[str, Any]:
        artifact_paths["model_dir"].mkdir(parents=True, exist_ok=True)
        artifact_paths["result_path"].write_text(
            json.dumps({"model": model}, ensure_ascii=False),
            encoding="utf-8",
        )
        if model == "bad-multivariate":
            return {
                "model": model,
                "interface": str(spec.interface),
                "requires": [],
                "backend": "core",
                "status": "error",
                "device": device,
                "duration_seconds": 0.1,
                "artifact_kind": "none",
                "artifact_path": "",
                "error_type": "RuntimeError",
                "error_message": "boom",
            }
        return {
            "model": model,
            "interface": str(spec.interface),
            "requires": [],
            "backend": "core",
            "status": "ok",
            "device": device,
            "duration_seconds": 0.1,
            "artifact_kind": "forecast_artifact",
            "artifact_path": str(artifact_paths["forecast_artifact_path"]),
            "error_type": "",
            "error_message": "",
        }

    monkeypatch.setattr(mod, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(mod, "list_models", _fake_list_models)
    monkeypatch.setattr(mod, "get_model_spec", _fake_get_model_spec)
    monkeypatch.setattr(mod, "resolve_runtime_device", lambda device: "cuda")
    monkeypatch.setattr(mod, "_train_single_model", _fake_train_single_model)

    payload = mod.run_registry_training_validation(
        output_dir=tmp_path,
        progress_every=2,
    )

    assert payload["summary"]["total_models"] == 3
    assert payload["summary"]["ok_models"] == 2
    assert payload["summary"]["failed_models"] == 1
    assert (tmp_path / "rows.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "progress.json").exists()
    assert (tmp_path / "models" / "ok-local" / "result.json").exists()
    assert (tmp_path / "models" / "ok-global" / "result.json").exists()
    assert (tmp_path / "models" / "bad-multivariate" / "result.json").exists()


def test_validate_all_models_parser_accepts_progress_every() -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "validate_all_models.py"
    spec = importlib.util.spec_from_file_location("validate_all_models_tool", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(["--progress-every", "25"])

    assert args.progress_every == 25


def test_validate_all_models_main_uses_training_validation_runner(
    monkeypatch, tmp_path: Path
) -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "validate_all_models.py"
    spec = importlib.util.spec_from_file_location("validate_all_models_tool", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module._ensure_src_on_path(module._repo_root())

    from foresight.services import model_validation as mod

    recorded: dict[str, Any] = {}

    def _fake_runner(**kwargs: Any) -> dict[str, Any]:
        recorded.update(kwargs)
        return {
            "summary": {
                "total_models": 2,
                "ok_models": 2,
                "failed_models": 0,
                "by_interface": {"local": 2},
                "by_backend": {"core": 2},
                "duration_seconds_total": 1.0,
            }
        }

    monkeypatch.setattr(mod, "run_registry_training_validation", _fake_runner)

    exit_code = module.main(
        [
            "--output-dir",
            str(tmp_path),
            "--models",
            "theta,torch-mlp-direct",
            "--device",
            "cuda",
            "--progress-every",
            "25",
        ]
    )

    assert exit_code == 0
    assert recorded["models"] == ["theta", "torch-mlp-direct"]
    assert recorded["device"] == "cuda"
    assert recorded["output_dir"] == str(tmp_path)
    assert recorded["progress_every"] == 25
