import numpy as np
import pandas as pd

from foresight.data import (
    align_long_df,
    clip_long_df_outliers,
    enrich_long_df_calendar,
    fit_long_df_scaler,
    inverse_transform_long_df_with_scaler,
    make_supervised_frame,
    prepare_long_df,
    split_long_df,
    transform_long_df_with_scaler,
)


def test_align_long_df_regularizes_frequency_and_aggregates_duplicate_timestamps() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s0", "s1", "s1"],
            "ds": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-03", "2020-01-01", "2020-01-03"]
            ),
            "y": [1.0, 2.0, 4.0, 10.0, 12.0],
        }
    )

    out = align_long_df(long_df, freq="D", agg="last")

    s0 = out.loc[out["unique_id"] == "s0"].reset_index(drop=True)
    s1 = out.loc[out["unique_id"] == "s1"].reset_index(drop=True)

    assert s0["ds"].tolist() == list(pd.date_range("2020-01-01", periods=3, freq="D"))
    assert s1["ds"].tolist() == list(pd.date_range("2020-01-01", periods=3, freq="D"))
    assert s0.loc[[0, 2], "y"].tolist() == [2.0, 4.0]
    assert s1.loc[[0, 2], "y"].tolist() == [10.0, 12.0]
    assert np.isnan(float(s0.loc[1, "y"]))
    assert np.isnan(float(s1.loc[1, "y"]))


def test_clip_long_df_outliers_clips_each_series_independently() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["stable"] * 4 + ["spiky"] * 4,
            "ds": pd.date_range("2020-01-01", periods=8, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 100.0],
        }
    )

    out = clip_long_df_outliers(long_df, method="iqr", columns=("y",), iqr_k=1.5)

    stable = out.loc[out["unique_id"] == "stable", "y"].tolist()
    spiky = out.loc[out["unique_id"] == "spiky", "y"].tolist()

    assert stable == [1.0, 2.0, 3.0, 4.0]
    assert spiky[-1] < 100.0
    assert spiky[-1] == 65.5


def test_enrich_long_df_calendar_appends_prefixed_time_features() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s0"],
            "ds": pd.date_range("2020-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 3.0],
        }
    )

    out = enrich_long_df_calendar(long_df, prefix="cal_")

    assert len(out) == 3
    assert {"cal_time_idx", "cal_dow_sin", "cal_dow_cos"} <= set(out.columns)
    assert np.all(np.isfinite(out.loc[:, ["cal_time_idx", "cal_dow_sin", "cal_dow_cos"]].to_numpy()))


def test_make_supervised_frame_long_single_step_schema_and_values() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 6,
            "ds": pd.date_range("2020-01-01", periods=6, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "promo": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )

    out = make_supervised_frame(long_df, lags=3, horizon=1)

    assert len(out) == 3
    assert {
        "unique_id",
        "ds",
        "target_t",
        "feat_y_lag3",
        "feat_y_lag2",
        "feat_y_lag1",
        "feat_x_promo",
        "y_target",
    } <= set(out.columns)
    assert out.loc[0, ["feat_y_lag3", "feat_y_lag2", "feat_y_lag1"]].tolist() == [1.0, 2.0, 3.0]
    assert float(out.loc[0, "feat_x_promo"]) == 13.0
    assert float(out.loc[0, "y_target"]) == 4.0


def test_make_supervised_frame_long_multi_step_outputs_direct_targets() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 6,
            "ds": pd.date_range("2020-01-01", periods=6, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    out = make_supervised_frame(long_df, lags=3, horizon=2)

    assert len(out) == 2
    assert {"y_t+1", "y_t+2"} <= set(out.columns)
    assert float(out.loc[0, "y_t+1"]) == 4.0
    assert float(out.loc[0, "y_t+2"]) == 5.0


def test_balanced_workflow_smoke_from_prepare_to_supervised() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s0", "s0", "s0"],
            "ds": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05", "2020-01-06"]
            ),
            "y": [1.0, 2.0, 100.0, 5.0, 6.0],
        }
    )

    prepared = prepare_long_df(long_df, freq="D", y_missing="interpolate")
    aligned = align_long_df(prepared, freq="D", agg="last")
    clipped = clip_long_df_outliers(aligned, method="iqr")
    enriched = enrich_long_df_calendar(clipped, prefix="cal_")
    frame = make_supervised_frame(enriched, lags=3, horizon=2)

    assert not frame.empty
    assert {"feat_x_cal_time_idx", "feat_x_cal_dow_sin", "y_t+1", "y_t+2"} <= set(frame.columns)
    assert np.all(np.isfinite(frame.loc[:, ["feat_x_cal_time_idx", "feat_x_cal_dow_sin"]].to_numpy()))


def test_split_long_df_sizes_respects_per_series_order() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 5 + ["s1"] * 5,
            "ds": list(pd.date_range("2020-01-01", periods=5, freq="D")) * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )

    parts = split_long_df(long_df, valid_size=1, test_size=1)

    assert set(parts) == {"train", "valid", "test"}
    assert len(parts["train"]) == 6
    assert len(parts["valid"]) == 2
    assert len(parts["test"]) == 2
    assert parts["train"].loc[parts["train"]["unique_id"] == "s0", "y"].tolist() == [1.0, 2.0, 3.0]
    assert parts["valid"].loc[parts["valid"]["unique_id"] == "s0", "y"].tolist() == [4.0]
    assert parts["test"].loc[parts["test"]["unique_id"] == "s0", "y"].tolist() == [5.0]


def test_split_long_df_gap_reserves_rows_between_train_and_test() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 6,
            "ds": pd.date_range("2020-01-01", periods=6, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    parts = split_long_df(long_df, test_size=1, gap=1)

    assert parts["train"]["y"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert parts["test"]["y"].tolist() == [6.0]
    assert parts["valid"].empty


def test_per_series_scaler_round_trip_restores_original_values() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 4 + ["s1"] * 4,
            "ds": list(pd.date_range("2020-01-01", periods=4, freq="D")) * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "promo": [0.0, 1.0, 0.0, 1.0, 10.0, 20.0, 10.0, 20.0],
        }
    )

    scaler = fit_long_df_scaler(long_df, method="standard", scope="per_series", columns=("y", "promo"))
    scaled = transform_long_df_with_scaler(long_df, scaler, columns=("y", "promo"))
    restored = inverse_transform_long_df_with_scaler(scaled, scaler, columns=("y", "promo"))

    assert {"scope", "unique_id", "column", "method", "center", "scale"} <= set(scaler.columns)
    assert np.allclose(restored["y"].to_numpy(dtype=float), long_df["y"].to_numpy(dtype=float))
    assert np.allclose(restored["promo"].to_numpy(dtype=float), long_df["promo"].to_numpy(dtype=float))


def test_global_scaler_uses_single_stats_row_per_column() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 3 + ["s1"] * 3,
            "ds": list(pd.date_range("2020-01-01", periods=3, freq="D")) * 2,
            "y": [1.0, 2.0, 4.0, 10.0, 20.0, 40.0],
        }
    )

    scaler = fit_long_df_scaler(long_df, method="maxabs", scope="global", columns=("y",))
    scaled = transform_long_df_with_scaler(long_df, scaler, columns=("y",))

    assert len(scaler) == 1
    assert scaler.loc[0, "unique_id"] == "__global__"
    assert scaler.loc[0, "method"] == "maxabs"
    assert float(np.abs(scaled["y"]).max()) == 1.0
