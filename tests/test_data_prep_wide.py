import pandas as pd
import pytest

from foresight.data.prep import prepare_wide_df


def test_prepare_wide_df_inserts_missing_timestamps_and_fills() -> None:
    wide_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "a": [1.0, 3.0],
            "b": [10.0, 30.0],
        }
    )

    out = prepare_wide_df(wide_df, ds_col="ds", freq="D", missing="zero")

    assert out["ds"].tolist() == list(pd.date_range("2020-01-01", periods=3, freq="D"))
    assert out["a"].tolist() == [1.0, 0.0, 3.0]
    assert out["b"].tolist() == [10.0, 0.0, 30.0]


def test_prepare_wide_df_strict_freq_rejects_irregular() -> None:
    wide_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06"]),
            "a": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="regular frequency"):
        prepare_wide_df(wide_df, ds_col="ds", strict_freq=True)

