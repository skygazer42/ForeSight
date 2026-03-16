import numpy as np
import pandas as pd
import pytest

from foresight.data import (
    align_long_df,
    clip_long_df_outliers,
    enrich_long_df_calendar,
    fit_long_df_scaler,
    inverse_transform_long_df_with_scaler,
    make_panel_sequence_blocks,
    make_panel_sequence_tensors,
    make_panel_window_arrays,
    make_panel_window_frame,
    make_supervised_arrays,
    make_supervised_frame,
    prepare_long_df,
    split_supervised_arrays,
    split_panel_window_arrays,
    split_panel_window_frame,
    split_panel_sequence_blocks,
    split_panel_sequence_tensors,
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


def test_make_supervised_arrays_matches_frame_output() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 6,
            "ds": pd.date_range("2020-01-01", periods=6, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "promo": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )

    frame = make_supervised_frame(long_df, lags=3, horizon=1, x_cols=("promo",))
    bundle = make_supervised_arrays(long_df, lags=3, horizon=1, x_cols=("promo",))

    feature_cols = tuple(frame.columns[3:-1])
    assert bundle["X"].shape == (3, len(feature_cols))
    assert bundle["y"].shape == (3,)
    assert bundle["feature_names"] == feature_cols
    assert bundle["target_names"] == ("y_target",)
    assert bundle["index"].equals(frame.loc[:, ["unique_id", "ds", "target_t"]])
    assert np.allclose(bundle["X"], frame.loc[:, list(feature_cols)].to_numpy(dtype=float))
    assert np.allclose(bundle["y"], frame["y_target"].to_numpy(dtype=float))
    assert bundle["metadata"]["horizon"] == 1
    assert bundle["metadata"]["n_rows"] == 3
    assert bundle["metadata"]["n_features"] == len(feature_cols)
    assert bundle["metadata"]["n_targets"] == 1


def test_make_supervised_arrays_multistep_target_is_2d() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 6,
            "ds": pd.date_range("2020-01-01", periods=6, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    frame = make_supervised_frame(long_df, lags=3, horizon=2)
    bundle = make_supervised_arrays(long_df, lags=3, horizon=2)

    target_cols = ("y_t+1", "y_t+2")
    assert bundle["X"].shape == (2, len(frame.columns) - 3 - len(target_cols))
    assert bundle["y"].shape == (2, 2)
    assert bundle["target_names"] == target_cols
    assert np.allclose(bundle["y"], frame.loc[:, list(target_cols)].to_numpy(dtype=float))


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


def test_split_supervised_arrays_respects_per_series_order() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 8 + ["s1"] * 8,
            "ds": list(pd.date_range("2020-01-01", periods=8, freq="D")) * 2,
            "y": [float(i) for i in range(1, 9)] + [float(i) for i in range(101, 109)],
        }
    )

    bundle = make_supervised_arrays(long_df, lags=2, horizon=1)
    parts = split_supervised_arrays(bundle, valid_size=1, test_size=1)

    assert set(parts) == {"train", "valid", "test"}
    assert parts["train"]["X"].shape[0] == 8
    assert parts["valid"]["X"].shape[0] == 2
    assert parts["test"]["X"].shape[0] == 2
    assert (
        parts["train"]["index"]
        .loc[parts["train"]["index"]["unique_id"] == "s0", "ds"]
        .tolist()
        == [
            pd.Timestamp("2020-01-03"),
            pd.Timestamp("2020-01-04"),
            pd.Timestamp("2020-01-05"),
            pd.Timestamp("2020-01-06"),
        ]
    )
    assert (
        parts["valid"]["index"]
        .loc[parts["valid"]["index"]["unique_id"] == "s0", "ds"]
        .tolist()
        == [pd.Timestamp("2020-01-07")]
    )
    assert (
        parts["test"]["index"]
        .loc[parts["test"]["index"]["unique_id"] == "s0", "ds"]
        .tolist()
        == [pd.Timestamp("2020-01-08")]
    )
    assert parts["train"]["metadata"]["n_rows"] == 8
    assert parts["valid"]["metadata"]["n_rows"] == 2
    assert parts["test"]["metadata"]["n_rows"] == 2


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


def test_make_panel_window_frame_multi_series_multi_step_schema_and_values() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 6 + ["s1"] * 6,
            "ds": list(pd.date_range("2020-01-01", periods=6, freq="D")) * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "promo": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )

    out = make_panel_window_frame(
        long_df,
        horizon=2,
        target_lags=(1, 2),
        seasonal_lags=(3,),
        historic_x_lags=(1,),
        future_x_lags=(0, 1),
        x_cols=("promo",),
    )

    assert len(out) == 8
    assert list(out.columns[:5]) == ["unique_id", "cutoff_ds", "target_ds", "step", "y"]
    assert {
        "y_lag_1",
        "y_lag_2",
        "y_seasonal_lag_3",
        "historic_x__promo_lag_1",
        "future_x__promo_lag_0",
        "future_x__promo_lag_1",
    } <= set(out.columns)

    first = out.iloc[0]
    second = out.iloc[1]

    assert first["unique_id"] == "s0"
    assert first["cutoff_ds"] == pd.Timestamp("2020-01-03")
    assert first["target_ds"] == pd.Timestamp("2020-01-04")
    assert int(first["step"]) == 1
    assert float(first["y"]) == 4.0
    assert float(first["y_lag_1"]) == 3.0
    assert float(first["y_lag_2"]) == 2.0
    assert float(first["y_seasonal_lag_3"]) == 1.0
    assert float(first["historic_x__promo_lag_1"]) == 12.0
    assert float(first["future_x__promo_lag_0"]) == 13.0
    assert float(first["future_x__promo_lag_1"]) == 12.0

    assert second["cutoff_ds"] == pd.Timestamp("2020-01-03")
    assert second["target_ds"] == pd.Timestamp("2020-01-05")
    assert int(second["step"]) == 2
    assert float(second["y"]) == 5.0
    assert float(second["future_x__promo_lag_0"]) == 14.0
    assert float(second["future_x__promo_lag_1"]) == 13.0


def test_make_panel_window_arrays_matches_frame_output() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 6,
            "ds": pd.date_range("2020-01-01", periods=6, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "promo": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )

    frame = make_panel_window_frame(
        long_df,
        horizon=2,
        target_lags=(1, 2),
        future_x_lags=(0, 1),
        x_cols=("promo",),
    )
    arrays = make_panel_window_arrays(
        long_df,
        horizon=2,
        target_lags=(1, 2),
        future_x_lags=(0, 1),
        x_cols=("promo",),
    )

    assert arrays["X"].shape == (6, len(frame.columns) - 5)
    assert arrays["y"].shape == (6,)
    assert tuple(arrays["feature_names"]) == tuple(frame.columns[5:])
    assert list(arrays["index"].columns) == ["unique_id", "cutoff_ds", "target_ds", "step"]
    assert np.allclose(arrays["X"], frame.loc[:, frame.columns[5:]].to_numpy(dtype=float))
    assert np.allclose(arrays["y"], frame["y"].to_numpy(dtype=float))
    assert arrays["metadata"]["horizon"] == 2
    assert arrays["metadata"]["target_lags"] == (2, 1)
    assert arrays["metadata"]["future_x_lags"] == (1, 0)
    assert arrays["metadata"]["x_cols"] == ("promo",)
    assert arrays["metadata"]["n_series"] == 1
    assert arrays["metadata"]["n_windows"] == 3
    assert arrays["metadata"]["n_rows"] == 6


def test_make_panel_window_frame_rejects_duplicate_timestamps() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s0", "s0"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"]),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="align_long_df"):
        make_panel_window_frame(long_df, horizon=1, target_lags=(1,))


def test_make_panel_window_frame_raises_when_no_windows_can_be_built() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 3,
            "ds": pd.date_range("2020-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="enough history"):
        make_panel_window_frame(long_df, horizon=2, target_lags=(1, 2, 3))


def test_split_panel_window_frame_respects_per_series_window_order() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 8 + ["s1"] * 8,
            "ds": list(pd.date_range("2020-01-01", periods=8, freq="D")) * 2,
            "y": [float(i) for i in range(1, 9)] + [float(i) for i in range(101, 109)],
            "promo": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 2,
        }
    )

    frame = make_panel_window_frame(
        long_df,
        horizon=2,
        target_lags=(1, 2),
        future_x_lags=(0,),
        x_cols=("promo",),
    )
    parts = split_panel_window_frame(frame, valid_size=1, test_size=1)

    assert set(parts) == {"train", "valid", "test"}
    assert len(parts["train"]) == 12
    assert len(parts["valid"]) == 4
    assert len(parts["test"]) == 4
    assert (
        parts["train"]
        .loc[parts["train"]["unique_id"] == "s0", "cutoff_ds"]
        .drop_duplicates()
        .tolist()
        == [
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-03"),
            pd.Timestamp("2020-01-04"),
        ]
    )
    assert (
        parts["valid"]
        .loc[parts["valid"]["unique_id"] == "s0", "cutoff_ds"]
        .drop_duplicates()
        .tolist()
        == [pd.Timestamp("2020-01-05")]
    )
    assert (
        parts["test"]
        .loc[parts["test"]["unique_id"] == "s0", "cutoff_ds"]
        .drop_duplicates()
        .tolist()
        == [pd.Timestamp("2020-01-06")]
    )
    assert parts["test"].groupby(["unique_id", "cutoff_ds"]).size().tolist() == [2, 2]


def test_split_panel_window_arrays_matches_frame_partitions() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 8 + ["s1"] * 8,
            "ds": list(pd.date_range("2020-01-01", periods=8, freq="D")) * 2,
            "y": [float(i) for i in range(1, 9)] + [float(i) for i in range(101, 109)],
            "promo": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0] * 2,
        }
    )

    frame = make_panel_window_frame(
        long_df,
        horizon=2,
        target_lags=(1, 2),
        future_x_lags=(0, 1),
        x_cols=("promo",),
    )
    arrays = make_panel_window_arrays(
        long_df,
        horizon=2,
        target_lags=(1, 2),
        future_x_lags=(0, 1),
        x_cols=("promo",),
    )

    frame_parts = split_panel_window_frame(frame, valid_size=1, test_size=1)
    array_parts = split_panel_window_arrays(arrays, valid_size=1, test_size=1)

    assert set(array_parts) == {"train", "valid", "test"}
    for name in ("train", "valid", "test"):
        expected = frame_parts[name]
        actual = array_parts[name]
        assert actual["X"].shape == (len(expected), len(expected.columns) - 5)
        assert actual["y"].shape == (len(expected),)
        assert tuple(actual["feature_names"]) == tuple(expected.columns[5:])
        assert actual["index"].reset_index(drop=True).equals(
            expected.loc[:, ["unique_id", "cutoff_ds", "target_ds", "step"]].reset_index(drop=True)
        )
        assert np.allclose(actual["X"], expected.loc[:, expected.columns[5:]].to_numpy(dtype=float))
        assert np.allclose(actual["y"], expected["y"].to_numpy(dtype=float))
    assert array_parts["train"]["metadata"]["n_windows"] == 6
    assert array_parts["valid"]["metadata"]["n_windows"] == 2
    assert array_parts["test"]["metadata"]["n_windows"] == 2


def test_split_panel_window_arrays_rejects_misaligned_bundle() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 6,
            "ds": pd.date_range("2020-01-01", periods=6, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    arrays = make_panel_window_arrays(long_df, horizon=2, target_lags=(1, 2))
    broken = dict(arrays)
    broken["X"] = arrays["X"][:-1]

    with pytest.raises(ValueError, match="same number of rows"):
        split_panel_window_arrays(broken, valid_size=1, test_size=1)


def test_make_panel_sequence_tensors_shapes_channel_order_and_predict_block() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 8 + ["s1"] * 8,
            "ds": list(pd.date_range("2020-01-01", periods=8, freq="D")) * 2,
            "y": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
            ],
            "promo": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 2,
        }
    )

    bundle = make_panel_sequence_tensors(
        long_df,
        cutoff=pd.Timestamp("2020-01-06"),
        horizon=2,
        context_length=3,
        x_cols=("promo",),
        normalize=False,
        add_time_features=True,
    )

    assert set(bundle) == {"train", "predict", "metadata"}
    assert bundle["train"]["X"].shape == (4, 5, 11)
    assert bundle["train"]["y"].shape == (4, 2)
    assert bundle["train"]["series_id"].shape == (4,)
    assert bundle["predict"]["X"].shape == (2, 5, 11)
    assert bundle["predict"]["series_id"].shape == (2,)
    assert list(bundle["train"]["window_index"].columns) == [
        "unique_id",
        "cutoff_ds",
        "target_start_ds",
        "target_end_ds",
    ]
    assert list(bundle["predict"]["index"].columns) == [
        "unique_id",
        "cutoff_ds",
        "target_start_ds",
        "target_end_ds",
    ]
    assert bundle["metadata"]["channel_names"] == (
        "y",
        "promo",
        "time_idx",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        "hour_sin",
        "hour_cos",
    )
    assert bundle["metadata"]["time_feature_names"] == (
        "time_idx",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        "hour_sin",
        "hour_cos",
    )
    assert bundle["metadata"]["n_series"] == 2
    assert bundle["metadata"]["n_train_windows"] == 4
    assert bundle["metadata"]["n_predict_windows"] == 2
    assert bundle["metadata"]["input_dim"] == 11
    assert np.allclose(bundle["train"]["X"][0, :, 0], np.asarray([1.0, 2.0, 3.0, 0.0, 0.0]))
    assert np.allclose(bundle["train"]["y"][0], np.asarray([4.0, 5.0]))
    assert np.allclose(bundle["predict"]["X"][0, :, 0], np.asarray([4.0, 5.0, 6.0, 0.0, 0.0]))


def test_make_panel_sequence_tensors_returns_target_norm_stats() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 8,
            "ds": pd.date_range("2020-01-01", periods=8, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "promo": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )

    bundle = make_panel_sequence_tensors(
        long_df,
        cutoff=pd.Timestamp("2020-01-06"),
        horizon=2,
        context_length=3,
        x_cols=("promo",),
        normalize=True,
        add_time_features=False,
    )

    expected_mean = float(np.mean([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    expected_std = float(np.std([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    assert np.isclose(float(bundle["predict"]["target_mean"][0]), expected_mean)
    assert np.isclose(float(bundle["predict"]["target_std"][0]), expected_std)
    assert np.allclose(
        bundle["train"]["y"][0],
        (np.asarray([4.0, 5.0]) - expected_mean) / expected_std,
    )


def test_split_panel_sequence_tensors_respects_per_series_order() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 10 + ["s1"] * 10,
            "ds": list(pd.date_range("2020-01-01", periods=10, freq="D")) * 2,
            "y": [float(i) for i in range(1, 11)] + [float(i) for i in range(101, 111)],
        }
    )

    bundle = make_panel_sequence_tensors(
        long_df,
        cutoff=pd.Timestamp("2020-01-08"),
        horizon=2,
        context_length=3,
        normalize=False,
        add_time_features=False,
    )
    parts = split_panel_sequence_tensors(bundle, valid_size=1, test_size=1)

    assert set(parts) == {"train", "valid", "test"}
    assert parts["train"]["X"].shape[0] == 4
    assert parts["valid"]["X"].shape[0] == 2
    assert parts["test"]["X"].shape[0] == 2
    assert (
        parts["train"]["window_index"]
        .loc[parts["train"]["window_index"]["unique_id"] == "s0", "cutoff_ds"]
        .tolist()
        == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-04")]
    )
    assert (
        parts["valid"]["window_index"]
        .loc[parts["valid"]["window_index"]["unique_id"] == "s0", "cutoff_ds"]
        .tolist()
        == [pd.Timestamp("2020-01-05")]
    )
    assert (
        parts["test"]["window_index"]
        .loc[parts["test"]["window_index"]["unique_id"] == "s0", "cutoff_ds"]
        .tolist()
        == [pd.Timestamp("2020-01-06")]
    )


def test_make_panel_sequence_tensors_rejects_duplicate_timestamps() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s0", "s0", "s0"],
            "ds": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    with pytest.raises(ValueError, match="align_long_df"):
        make_panel_sequence_tensors(
            long_df,
            cutoff=pd.Timestamp("2020-01-03"),
            horizon=1,
            context_length=2,
        )


def test_make_panel_sequence_blocks_matches_packed_sequence_values() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 8 + ["s1"] * 8,
            "ds": list(pd.date_range("2020-01-01", periods=8, freq="D")) * 2,
            "y": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
            ],
            "promo": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 2,
        }
    )

    packed = make_panel_sequence_tensors(
        long_df,
        cutoff=pd.Timestamp("2020-01-06"),
        horizon=2,
        context_length=3,
        x_cols=("promo",),
        normalize=False,
        add_time_features=True,
    )
    blocks = make_panel_sequence_blocks(
        long_df,
        cutoff=pd.Timestamp("2020-01-06"),
        horizon=2,
        context_length=3,
        x_cols=("promo",),
        normalize=False,
        add_time_features=True,
    )

    assert set(blocks) == {"train", "predict", "metadata"}
    assert blocks["train"]["past_y"].shape == (4, 3, 1)
    assert blocks["train"]["future_y_seed"].shape == (4, 2, 1)
    assert blocks["train"]["past_x"].shape == (4, 3, 1)
    assert blocks["train"]["future_x"].shape == (4, 2, 1)
    assert blocks["train"]["past_time"].shape == (4, 3, 9)
    assert blocks["train"]["future_time"].shape == (4, 2, 9)
    assert blocks["train"]["target_y"].shape == (4, 2)
    assert np.allclose(blocks["train"]["past_y"], packed["train"]["X"][:, :3, 0:1])
    assert np.allclose(blocks["train"]["future_y_seed"], packed["train"]["X"][:, 3:, 0:1])
    assert np.allclose(blocks["train"]["past_x"], packed["train"]["X"][:, :3, 1:2])
    assert np.allclose(blocks["train"]["future_x"], packed["train"]["X"][:, 3:, 1:2])
    assert np.allclose(blocks["train"]["past_time"], packed["train"]["X"][:, :3, 2:])
    assert np.allclose(blocks["train"]["future_time"], packed["train"]["X"][:, 3:, 2:])
    assert np.allclose(blocks["train"]["target_y"], packed["train"]["y"])
    assert np.allclose(blocks["predict"]["past_y"], packed["predict"]["X"][:, :3, 0:1])
    assert np.allclose(blocks["predict"]["future_y_seed"], packed["predict"]["X"][:, 3:, 0:1])
    assert np.allclose(blocks["predict"]["past_x"], packed["predict"]["X"][:, :3, 1:2])
    assert np.allclose(blocks["predict"]["future_x"], packed["predict"]["X"][:, 3:, 1:2])
    assert np.allclose(blocks["predict"]["past_time"], packed["predict"]["X"][:, :3, 2:])
    assert np.allclose(blocks["predict"]["future_time"], packed["predict"]["X"][:, 3:, 2:])
    assert np.allclose(blocks["predict"]["target_mean"], packed["predict"]["target_mean"])
    assert np.allclose(blocks["predict"]["target_std"], packed["predict"]["target_std"])


def test_make_panel_sequence_blocks_keeps_zero_width_optional_blocks() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 8,
            "ds": pd.date_range("2020-01-01", periods=8, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )

    blocks = make_panel_sequence_blocks(
        long_df,
        cutoff=pd.Timestamp("2020-01-06"),
        horizon=2,
        context_length=3,
        normalize=False,
        x_cols=(),
        add_time_features=False,
    )

    assert blocks["train"]["past_x"].shape == (2, 3, 0)
    assert blocks["train"]["future_x"].shape == (2, 2, 0)
    assert blocks["train"]["past_time"].shape == (2, 3, 0)
    assert blocks["train"]["future_time"].shape == (2, 2, 0)
    assert blocks["predict"]["past_x"].shape == (1, 3, 0)
    assert blocks["predict"]["future_x"].shape == (1, 2, 0)
    assert blocks["predict"]["past_time"].shape == (1, 3, 0)
    assert blocks["predict"]["future_time"].shape == (1, 2, 0)


def test_split_panel_sequence_blocks_respects_per_series_order() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 10 + ["s1"] * 10,
            "ds": list(pd.date_range("2020-01-01", periods=10, freq="D")) * 2,
            "y": [float(i) for i in range(1, 11)] + [float(i) for i in range(101, 111)],
        }
    )

    bundle = make_panel_sequence_blocks(
        long_df,
        cutoff=pd.Timestamp("2020-01-08"),
        horizon=2,
        context_length=3,
        normalize=False,
        add_time_features=False,
    )
    parts = split_panel_sequence_blocks(bundle, valid_size=1, test_size=1)

    assert set(parts) == {"train", "valid", "test"}
    assert parts["train"]["past_y"].shape == (4, 3, 1)
    assert parts["valid"]["past_y"].shape == (2, 3, 1)
    assert parts["test"]["past_y"].shape == (2, 3, 1)
    assert (
        parts["train"]["window_index"]
        .loc[parts["train"]["window_index"]["unique_id"] == "s0", "cutoff_ds"]
        .tolist()
        == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-04")]
    )
    assert (
        parts["valid"]["window_index"]
        .loc[parts["valid"]["window_index"]["unique_id"] == "s0", "cutoff_ds"]
        .tolist()
        == [pd.Timestamp("2020-01-05")]
    )
    assert (
        parts["test"]["window_index"]
        .loc[parts["test"]["window_index"]["unique_id"] == "s0", "cutoff_ds"]
        .tolist()
        == [pd.Timestamp("2020-01-06")]
    )
