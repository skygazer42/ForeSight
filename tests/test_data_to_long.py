import pandas as pd

from foresight.data.format import to_long


def test_to_long_builds_unique_id_and_standard_columns():
    df = pd.DataFrame(
        {
            "store": [1, 1],
            "dept": [2, 2],
            "week": pd.to_datetime(["2020-01-01", "2020-01-08"]),
            "sales": [10.0, 11.0],
        }
    )
    out = to_long(df, time_col="week", y_col="sales", id_cols=("store", "dept"))
    assert list(out.columns) == ["unique_id", "ds", "y"]
    assert out["unique_id"].tolist() == ["store=1|dept=2", "store=1|dept=2"]
    assert out["y"].tolist() == [10.0, 11.0]
