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
    assert out.attrs["historic_x_cols"] == ()
    assert out.attrs["future_x_cols"] == ("promo",)


def test_to_long_supports_historic_and_future_covariate_roles():
    df = pd.DataFrame(
        {
            "store": [1, 1, 1],
            "week": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
            "sales": [10.0, 11.0, 12.0],
            "promo_hist": [0.0, 1.0, 0.0],
            "promo_futr": [1.0, 1.0, 0.0],
        }
    )
    out = to_long(
        df,
        time_col="week",
        y_col="sales",
        id_cols=("store",),
        historic_x_cols=("promo_hist",),
        future_x_cols=("promo_futr",),
    )
    assert list(out.columns) == ["unique_id", "ds", "y", "promo_hist", "promo_futr"]
    assert out.attrs["historic_x_cols"] == ("promo_hist",)
    assert out.attrs["future_x_cols"] == ("promo_futr",)


def test_to_long_preserves_static_covariates_and_attrs():
    df = pd.DataFrame(
        {
            "store": [1, 1, 1],
            "week": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
            "sales": [10.0, 11.0, 12.0],
            "store_size": [100.0, 100.0, 100.0],
        }
    )
    out = to_long(
        df,
        time_col="week",
        y_col="sales",
        id_cols=("store",),
        static_cols=("store_size",),
    )

    assert list(out.columns) == ["unique_id", "ds", "y", "store_size"]
    assert out["store_size"].tolist() == [100.0, 100.0, 100.0]
    assert out.attrs["static_cols"] == ("store_size",)


def test_to_long_x_cols_alias_merges_into_future_covariates():
    df = pd.DataFrame(
        {
            "store": [1, 1],
            "week": pd.to_datetime(["2020-01-01", "2020-01-08"]),
            "sales": [10.0, 11.0],
            "promo": [0.0, 1.0],
            "future_temp": [20.0, 21.0],
        }
    )
    out = to_long(
        df,
        time_col="week",
        y_col="sales",
        id_cols=("store",),
        x_cols=("promo",),
        future_x_cols=("future_temp",),
    )
    assert out.attrs["historic_x_cols"] == ()
    assert out.attrs["future_x_cols"] == ("future_temp", "promo")


def test_to_long_can_prepare_regularized_output_with_static_covariates():
    df = pd.DataFrame(
        {
            "store": [1, 1],
            "week": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "sales": [10.0, 12.0],
            "promo": [1.0, 1.0],
            "store_size": [100.0, 100.0],
        }
    )
    out = to_long(
        df,
        time_col="week",
        y_col="sales",
        id_cols=("store",),
        x_cols=("promo",),
        static_cols=("store_size",),
        prepare=True,
        freq="D",
        y_missing="zero",
        x_missing="ffill",
    )

    assert out["ds"].tolist() == list(pd.date_range("2020-01-01", periods=3, freq="D"))
    assert out["store_size"].tolist() == [100.0, 100.0, 100.0]
    assert out.attrs["static_cols"] == ("store_size",)


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
