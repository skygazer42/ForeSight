import pandas as pd

from foresight.data.format import long_to_wide


def test_long_to_wide_pivots_expected_shape_and_columns() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
            "y": [1.0, 2.0, 10.0, 20.0],
        }
    )

    out = long_to_wide(long_df)

    assert list(out.columns) == ["ds", "a", "b"]
    assert out["ds"].tolist() == list(pd.date_range("2020-01-01", periods=2, freq="D"))
    assert out["a"].tolist() == [1.0, 2.0]
    assert out["b"].tolist() == [10.0, 20.0]


def test_long_to_wide_can_fill_missing_values_with_zero() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a", "b", "b"],
            "ds": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01", "2020-01-03"]
            ),
            "y": [1.0, 2.0, 3.0, 10.0, 30.0],
        }
    )

    out = long_to_wide(long_df, freq="D", missing="zero")

    assert out["ds"].tolist() == list(pd.date_range("2020-01-01", periods=3, freq="D"))
    assert out["a"].tolist() == [1.0, 2.0, 3.0]
    assert out["b"].tolist() == [10.0, 0.0, 30.0]
