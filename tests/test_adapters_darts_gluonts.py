from __future__ import annotations

import pandas as pd
import pytest

import foresight.adapters.darts as darts_adapter_mod
import foresight.adapters.gluonts as gluonts_adapter_mod


class _FakeTimeSeries:
    def __init__(
        self,
        data: pd.Series | pd.DataFrame,
        *,
        static_covariates: pd.DataFrame | None = None,
    ) -> None:
        if isinstance(data, pd.Series):
            self._data = data.astype(float)
        else:
            self._data = data.astype(float)
        self.static_covariates = static_covariates

    @classmethod
    def from_series(cls, series: pd.Series):
        return cls(series)

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame):
        return cls(frame.copy())

    def pd_series(self) -> pd.Series:
        if isinstance(self._data, pd.Series):
            return self._data.copy()
        if self._data.shape[1] != 1:
            raise TypeError("fake TimeSeries needs pd_dataframe() for multi-column data")
        return self._data.iloc[:, 0].copy()

    def pd_dataframe(self) -> pd.DataFrame:
        if isinstance(self._data, pd.Series):
            return self._data.to_frame()
        return self._data.copy()

    def with_static_covariates(self, static_covariates: pd.DataFrame):
        data = self._data.copy() if hasattr(self._data, "copy") else self._data
        return type(self)(data, static_covariates=static_covariates.copy())


class _FakeListDataset(list):
    def __init__(self, data_iter, *, freq: str):
        super().__init__(list(data_iter))
        self.freq = freq


def test_darts_adapter_round_trips_single_series(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        darts_adapter_mod,
        "_require_darts",
        lambda: type("_FakeDartsModule", (), {"TimeSeries": _FakeTimeSeries})(),
    )
    series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
        name="y",
    )

    ts = darts_adapter_mod.to_darts_timeseries(series)
    restored = darts_adapter_mod.from_darts_timeseries(ts)

    assert isinstance(restored, pd.Series)
    assert restored.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert restored.index.tolist() == list(series.index)


def test_darts_adapter_converts_panel_long_df_to_mapping_and_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        darts_adapter_mod,
        "_require_darts",
        lambda: type("_FakeDartsModule", (), {"TimeSeries": _FakeTimeSeries})(),
    )
    long_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "y": [1.0, 2.0, 10.0, 11.0],
        }
    )

    mapping = darts_adapter_mod.to_darts_timeseries(long_df)
    restored = darts_adapter_mod.from_darts_timeseries(mapping)

    assert sorted(mapping) == ["a", "b"]
    assert list(restored.columns) == ["unique_id", "ds", "y"]
    assert restored.to_dict("records") == [
        {"unique_id": "a", "ds": pd.Timestamp("2024-01-01"), "y": 1.0},
        {"unique_id": "a", "ds": pd.Timestamp("2024-01-02"), "y": 2.0},
        {"unique_id": "b", "ds": pd.Timestamp("2024-01-01"), "y": 10.0},
        {"unique_id": "b", "ds": pd.Timestamp("2024-01-02"), "y": 11.0},
    ]


def test_darts_bundle_round_trips_single_series_future_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        darts_adapter_mod,
        "_require_darts",
        lambda: type("_FakeDartsModule", (), {"TimeSeries": _FakeTimeSeries})(),
    )
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s0"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, 3.0],
            "promo": [0.0, 1.0, 0.0],
        }
    )
    long_df.attrs["historic_x_cols"] = ()
    long_df.attrs["future_x_cols"] = ("promo",)
    long_df.attrs["static_cols"] = ()

    bundle = darts_adapter_mod.to_darts_bundle(long_df)

    assert sorted(bundle) == ["freq", "future_covariates", "past_covariates", "target"]
    assert bundle["past_covariates"] == {}
    assert sorted(bundle["target"]) == ["s0"]
    assert sorted(bundle["future_covariates"]) == ["s0"]
    assert bundle["freq"] == {"s0": "D"}
    assert isinstance(bundle["target"]["s0"], _FakeTimeSeries)
    assert isinstance(bundle["future_covariates"]["s0"], _FakeTimeSeries)

    restored = darts_adapter_mod.from_darts_bundle(bundle)

    assert list(restored.columns) == ["unique_id", "ds", "y", "promo"]
    assert restored.attrs["historic_x_cols"] == ()
    assert restored.attrs["future_x_cols"] == ("promo",)
    assert restored.attrs["static_cols"] == ()
    assert restored["promo"].tolist() == pytest.approx([0.0, 1.0, 0.0])


def test_darts_bundle_import_accepts_legacy_single_series_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        darts_adapter_mod,
        "_require_darts",
        lambda: type("_FakeDartsModule", (), {"TimeSeries": _FakeTimeSeries})(),
    )
    target = _FakeTimeSeries.from_series(
        pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
            name="y",
        )
    )
    setattr(target, "_foresight_unique_id", "s0")
    future = _FakeTimeSeries.from_series(
        pd.Series(
            [0.0, 1.0, 0.0],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
            name="promo",
        )
    )
    setattr(future, "_foresight_unique_id", "s0")

    restored = darts_adapter_mod.from_darts_bundle(
        {
            "target": target,
            "past_covariates": None,
            "future_covariates": future,
            "freq": "D",
        }
    )

    assert list(restored.columns) == ["unique_id", "ds", "y", "promo"]
    assert restored.attrs["historic_x_cols"] == ()
    assert restored.attrs["future_x_cols"] == ("promo",)
    assert restored.attrs["static_cols"] == ()
    assert restored["unique_id"].tolist() == ["s0", "s0", "s0"]


def test_darts_bundle_round_trips_panel_covariates_and_static_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        darts_adapter_mod,
        "_require_darts",
        lambda: type("_FakeDartsModule", (), {"TimeSeries": _FakeTimeSeries})(),
    )
    long_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "y": [1.0, 2.0, 10.0, 11.0],
            "stock": [5.0, 6.0, 7.0, 8.0],
            "stock_lag2": [4.0, 5.0, 6.0, 7.0],
            "promo": [0.0, 1.0, 1.0, 0.0],
            "future_temp": [20.0, 21.0, 18.0, 19.0],
            "store_size": [100.0, 100.0, 150.0, 150.0],
        }
    )
    long_df.attrs["historic_x_cols"] = ("stock", "stock_lag2")
    long_df.attrs["future_x_cols"] = ("promo", "future_temp")
    long_df.attrs["static_cols"] = ("store_size",)

    bundle = darts_adapter_mod.to_darts_bundle(long_df)

    assert sorted(bundle) == ["freq", "future_covariates", "past_covariates", "target"]
    assert bundle["freq"] == {"a": "D", "b": "D"}
    assert sorted(bundle["target"]) == ["a", "b"]
    assert sorted(bundle["past_covariates"]) == ["a", "b"]
    assert sorted(bundle["future_covariates"]) == ["a", "b"]
    assert bundle["target"]["a"].static_covariates is not None

    restored = darts_adapter_mod.from_darts_bundle(bundle)

    assert list(restored.columns) == [
        "unique_id",
        "ds",
        "y",
        "stock",
        "stock_lag2",
        "promo",
        "future_temp",
        "store_size",
    ]
    assert restored.attrs["historic_x_cols"] == ("stock", "stock_lag2")
    assert restored.attrs["future_x_cols"] == ("promo", "future_temp")
    assert restored.attrs["static_cols"] == ("store_size",)
    assert restored["store_size"].tolist() == pytest.approx([100.0, 100.0, 150.0, 150.0])


def test_darts_bundle_rejects_non_constant_static_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        darts_adapter_mod,
        "_require_darts",
        lambda: type("_FakeDartsModule", (), {"TimeSeries": _FakeTimeSeries})(),
    )
    long_df = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "y": [1.0, 2.0],
            "store_size": [100.0, 101.0],
        }
    )
    long_df.attrs["historic_x_cols"] = ()
    long_df.attrs["future_x_cols"] = ()
    long_df.attrs["static_cols"] = ("store_size",)

    with pytest.raises(ValueError, match="static_cols column 'store_size' must be constant"):
        darts_adapter_mod.to_darts_bundle(long_df)


def test_darts_bundle_requires_two_points_per_series_to_infer_freq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        darts_adapter_mod,
        "_require_darts",
        lambda: type("_FakeDartsModule", (), {"TimeSeries": _FakeTimeSeries})(),
    )
    long_df = pd.DataFrame(
        {
            "unique_id": ["a"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "y": [1.0],
        }
    )

    with pytest.raises(ValueError, match="requires explicit freq or at least 2 timestamps per series"):
        darts_adapter_mod.to_darts_bundle(long_df)


def test_gluonts_adapter_builds_list_dataset_from_panel_long_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gluonts_adapter_mod,
        "_require_gluonts",
        lambda: type(
            "_FakeGluonTSModule",
            (),
            {
                "dataset": type(
                    "_Dataset",
                    (),
                    {"common": type("_Common", (), {"ListDataset": _FakeListDataset})},
                )
            },
        )(),
    )
    long_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "y": [1.0, 2.0, 10.0, 11.0],
        }
    )

    dataset = gluonts_adapter_mod.to_gluonts_list_dataset(long_df)

    assert isinstance(dataset, _FakeListDataset)
    assert dataset.freq == "D"
    assert dataset == [
        {"start": pd.Timestamp("2024-01-01"), "target": [1.0, 2.0], "item_id": "a"},
        {"start": pd.Timestamp("2024-01-01"), "target": [10.0, 11.0], "item_id": "b"},
    ]


def test_gluonts_bundle_exports_panel_covariates_and_static_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gluonts_adapter_mod,
        "_require_gluonts",
        lambda: type(
            "_FakeGluonTSModule",
            (),
            {
                "dataset": type(
                    "_Dataset",
                    (),
                    {"common": type("_Common", (), {"ListDataset": _FakeListDataset})},
                )
            },
        )(),
    )
    long_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "y": [1.0, 2.0, 10.0, 11.0],
            "stock": [5.0, 6.0, 7.0, 8.0],
            "promo": [0.0, 1.0, 1.0, 0.0],
            "store_size": [100.0, 100.0, 150.0, 150.0],
        }
    )
    long_df.attrs["historic_x_cols"] = ("stock",)
    long_df.attrs["future_x_cols"] = ("promo",)
    long_df.attrs["static_cols"] = ("store_size",)

    bundle = gluonts_adapter_mod.to_gluonts_bundle(long_df)

    assert sorted(bundle) == [
        "feat_dynamic_real",
        "feat_static_real",
        "feature_names",
        "freq",
        "past_feat_dynamic_real",
        "target",
    ]
    assert sorted(bundle["target"]) == ["a", "b"]
    assert bundle["freq"] == {"a": "D", "b": "D"}
    assert bundle["past_feat_dynamic_real"]["a"] == [[5.0, 6.0]]
    assert bundle["feat_dynamic_real"]["a"] == [[0.0, 1.0]]
    assert bundle["feat_static_real"]["a"] == [100.0]
    assert bundle["feature_names"] == {
        "historic_x_cols": ("stock",),
        "future_x_cols": ("promo",),
        "static_cols": ("store_size",),
    }


def test_gluonts_bundle_round_trips_back_to_long_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gluonts_adapter_mod,
        "_require_gluonts",
        lambda: type(
            "_FakeGluonTSModule",
            (),
            {
                "dataset": type(
                    "_Dataset",
                    (),
                    {"common": type("_Common", (), {"ListDataset": _FakeListDataset})},
                )
            },
        )(),
    )
    bundle = {
        "target": {
            "a": {
                "start": pd.Timestamp("2024-01-01"),
                "target": [1.0, 2.0],
                "item_id": "a",
            }
        },
        "past_feat_dynamic_real": {"a": [[5.0, 6.0]]},
        "feat_dynamic_real": {"a": [[0.0, 1.0]]},
        "feat_static_real": {"a": [100.0]},
        "feature_names": {
            "historic_x_cols": ("stock",),
            "future_x_cols": ("promo",),
            "static_cols": ("store_size",),
        },
        "freq": {"a": "D"},
    }

    restored = gluonts_adapter_mod.from_gluonts_bundle(bundle)

    assert list(restored.columns) == ["unique_id", "ds", "y", "stock", "promo", "store_size"]
    assert restored.attrs["historic_x_cols"] == ("stock",)
    assert restored.attrs["future_x_cols"] == ("promo",)
    assert restored.attrs["static_cols"] == ("store_size",)
    assert restored.to_dict("records") == [
        {
            "unique_id": "a",
            "ds": pd.Timestamp("2024-01-01"),
            "y": 1.0,
            "stock": 5.0,
            "promo": 0.0,
            "store_size": 100.0,
        },
        {
            "unique_id": "a",
            "ds": pd.Timestamp("2024-01-02"),
            "y": 2.0,
            "stock": 6.0,
            "promo": 1.0,
            "store_size": 100.0,
        },
    ]


def test_shared_beta_bundle_exports_panel_covariates_and_static_columns() -> None:
    import foresight.adapters as adapters_mod

    long_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "y": [1.0, 2.0, 10.0, 11.0],
            "stock": [5.0, 6.0, 7.0, 8.0],
            "promo": [0.0, 1.0, 1.0, 0.0],
            "store_size": [100.0, 100.0, 150.0, 150.0],
        }
    )
    long_df.attrs["historic_x_cols"] = ("stock",)
    long_df.attrs["future_x_cols"] = ("promo",)
    long_df.attrs["static_cols"] = ("store_size",)

    bundle = adapters_mod.to_beta_bundle(long_df)

    assert sorted(bundle) == [
        "freq",
        "future_covariates",
        "historic_covariates",
        "static_covariates",
        "target",
    ]
    assert bundle["freq"] == {"a": "D", "b": "D"}
    assert sorted(bundle["target"]) == ["a", "b"]
    assert list(bundle["target"]["a"].columns) == ["ds", "y"]
    assert list(bundle["historic_covariates"]["a"].columns) == ["ds", "stock"]
    assert list(bundle["future_covariates"]["a"].columns) == ["ds", "promo"]
    assert list(bundle["static_covariates"]["a"].columns) == ["store_size"]


def test_shared_beta_bundle_round_trips_back_to_canonical_long_df() -> None:
    import foresight.adapters as adapters_mod

    bundle = {
        "target": {
            "a": pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                    "y": [1.0, 2.0],
                }
            )
        },
        "historic_covariates": {
            "a": pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                    "stock": [5.0, 6.0],
                }
            )
        },
        "future_covariates": {
            "a": pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                    "promo": [0.0, 1.0],
                }
            )
        },
        "static_covariates": {
            "a": pd.DataFrame([{"store_size": 100.0}])
        },
        "freq": {"a": "D"},
    }

    restored = adapters_mod.from_beta_bundle(bundle)

    assert list(restored.columns) == ["unique_id", "ds", "y", "stock", "promo", "store_size"]
    assert restored.attrs["historic_x_cols"] == ("stock",)
    assert restored.attrs["future_x_cols"] == ("promo",)
    assert restored.attrs["static_cols"] == ("store_size",)
    assert restored.to_dict("records") == [
        {
            "unique_id": "a",
            "ds": pd.Timestamp("2024-01-01"),
            "y": 1.0,
            "stock": 5.0,
            "promo": 0.0,
            "store_size": 100.0,
        },
        {
            "unique_id": "a",
            "ds": pd.Timestamp("2024-01-02"),
            "y": 2.0,
            "stock": 6.0,
            "promo": 1.0,
            "store_size": 100.0,
        },
    ]


def test_gluonts_adapter_normalizes_two_point_daily_frequency_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pandas3DailyOffset:
        freqstr = "24h"

    monkeypatch.setattr(
        gluonts_adapter_mod.pd.tseries.frequencies,
        "to_offset",
        lambda _delta: _Pandas3DailyOffset(),
    )

    freq = gluonts_adapter_mod._infer_series_frequency(
        pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    )

    assert freq == "D"


def test_darts_adapter_missing_dependency_uses_darts_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        darts_adapter_mod,
        "_require_darts",
        lambda: (_ for _ in ()).throw(
            ImportError(
                "darts adapter requires darts. Install with: "
                'pip install "foresight-ts[darts]" or pip install -e ".[darts]"'
            )
        ),
    )

    with pytest.raises(ImportError, match='pip install "foresight-ts\\[darts\\]"'):
        darts_adapter_mod.to_darts_timeseries(pd.Series([1.0, 2.0, 3.0]))


def test_gluonts_adapter_missing_dependency_uses_gluonts_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gluonts_adapter_mod,
        "_require_gluonts",
        lambda: (_ for _ in ()).throw(
            ImportError(
                "gluonts adapter requires gluonts. Install with: "
                'pip install "foresight-ts[gluonts]" or pip install -e ".[gluonts]"'
            )
        ),
    )

    with pytest.raises(ImportError, match='pip install "foresight-ts\\[gluonts\\]"'):
        gluonts_adapter_mod.to_gluonts_list_dataset(
            pd.DataFrame(
                {
                    "unique_id": ["a"],
                    "ds": pd.to_datetime(["2024-01-01"]),
                    "y": [1.0],
                }
            )
        )
