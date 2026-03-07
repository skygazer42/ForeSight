import pandas as pd
import pytest

from foresight.data.prep import infer_series_frequency, prepare_long_df


def test_infer_series_frequency_detects_daily_series() -> None:
    ds = pd.date_range("2020-01-01", periods=4, freq="D")
    assert infer_series_frequency(ds) == "D"


def test_prepare_long_df_inserts_missing_timestamps_and_zero_fills_target() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "y": [10.0, 12.0],
        }
    )

    out = prepare_long_df(long_df, freq="D", y_missing="zero")

    assert out["ds"].tolist() == list(pd.date_range("2020-01-01", periods=3, freq="D"))
    assert out["y"].tolist() == [10.0, 0.0, 12.0]


def test_prepare_long_df_supports_separate_y_and_x_missing_policies() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "y": [10.0, 12.0],
            "promo": [1.0, None],
        }
    )

    out = prepare_long_df(long_df, freq="D", y_missing="interpolate", x_missing="ffill")

    assert out["ds"].tolist() == list(pd.date_range("2020-01-01", periods=3, freq="D"))
    assert out["y"].tolist() == [10.0, 11.0, 12.0]
    assert out["promo"].tolist() == [1.0, 1.0, 1.0]


def test_prepare_long_df_rejects_mixed_frequencies_in_strict_mode() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["daily", "daily", "weekly", "weekly"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-08"]),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="Mixed frequencies"):
        prepare_long_df(long_df, strict_freq=True)


def test_infer_series_frequency_rejects_irregular_series_in_strict_mode() -> None:
    ds = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06"])

    with pytest.raises(ValueError, match="regular frequency"):
        infer_series_frequency(ds, strict=True)
