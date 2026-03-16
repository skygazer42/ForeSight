import numpy as np
import pandas as pd

from foresight.data import (
    align_long_df,
    clip_long_df_outliers,
    enrich_long_df_calendar,
    make_supervised_frame,
    prepare_long_df,
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
