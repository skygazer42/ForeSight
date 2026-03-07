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


def test_to_long_keeps_covariates_when_x_cols_provided():
    df = pd.DataFrame(
        {
            "store": [1, 1, 1],
            "week": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
            "sales": [10.0, 11.0, 12.0],
            "promo": [0.0, 1.0, 0.0],
        }
    )
    out = to_long(df, time_col="week", y_col="sales", id_cols=("store",), x_cols=("promo",))
    assert list(out.columns) == ["unique_id", "ds", "y", "promo"]
    assert out["promo"].tolist() == [0.0, 1.0, 0.0]


def test_to_long_can_prepare_regularized_output_when_requested():
    df = pd.DataFrame(
        {
            "store": [1, 1],
            "week": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "sales": [10.0, 12.0],
            "promo": [1.0, 1.0],
        }
    )
    out = to_long(
        df,
        time_col="week",
        y_col="sales",
        id_cols=("store",),
        x_cols=("promo",),
        prepare=True,
        freq="D",
        y_missing="zero",
        x_missing="ffill",
    )

    assert out["ds"].tolist() == list(pd.date_range("2020-01-01", periods=3, freq="D"))
    assert out["y"].tolist() == [10.0, 0.0, 12.0]
    assert out["promo"].tolist() == [1.0, 1.0, 1.0]
