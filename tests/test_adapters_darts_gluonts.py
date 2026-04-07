from __future__ import annotations

import pandas as pd
import pytest

import foresight.adapters.darts as darts_adapter_mod
import foresight.adapters.gluonts as gluonts_adapter_mod


class _FakeTimeSeries:
    def __init__(
        self,
        series: pd.Series,
        *,
        static_covariates: pd.DataFrame | None = None,
    ) -> None:
        self._series = series.astype(float)
        self.static_covariates = static_covariates

    @classmethod
    def from_series(cls, series: pd.Series):
        return cls(series)

    def pd_series(self) -> pd.Series:
        return self._series.copy()

    def with_static_covariates(self, static_covariates: pd.DataFrame):
        return type(self)(self._series.copy(), static_covariates=static_covariates.copy())


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
    assert bundle["past_covariates"] is None
    assert isinstance(bundle["target"], _FakeTimeSeries)
    assert isinstance(bundle["future_covariates"], _FakeTimeSeries)
    assert bundle["freq"] == "D"

    restored = darts_adapter_mod.from_darts_bundle(bundle)

    assert list(restored.columns) == ["unique_id", "ds", "y", "promo"]
    assert restored.attrs["historic_x_cols"] == ()
    assert restored.attrs["future_x_cols"] == ("promo",)
    assert restored.attrs["static_cols"] == ()
    assert restored["promo"].tolist() == pytest.approx([0.0, 1.0, 0.0])


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
            "promo": [0.0, 1.0, 1.0, 0.0],
            "store_size": [100.0, 100.0, 150.0, 150.0],
        }
    )
    long_df.attrs["historic_x_cols"] = ("stock",)
    long_df.attrs["future_x_cols"] = ("promo",)
    long_df.attrs["static_cols"] = ("store_size",)

    bundle = darts_adapter_mod.to_darts_bundle(long_df)

    assert sorted(bundle) == ["freq", "future_covariates", "past_covariates", "target"]
    assert sorted(bundle["target"]) == ["a", "b"]
    assert sorted(bundle["past_covariates"]) == ["a", "b"]
    assert sorted(bundle["future_covariates"]) == ["a", "b"]
    assert bundle["target"]["a"].static_covariates is not None

    restored = darts_adapter_mod.from_darts_bundle(bundle)

    assert list(restored.columns) == ["unique_id", "ds", "y", "stock", "promo", "store_size"]
    assert restored.attrs["historic_x_cols"] == ("stock",)
    assert restored.attrs["future_x_cols"] == ("promo",)
    assert restored.attrs["static_cols"] == ("store_size",)
    assert restored["store_size"].tolist() == pytest.approx([100.0, 100.0, 150.0, 150.0])


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
