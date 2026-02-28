import pandas as pd

from foresight.datasets.loaders import load_cashflow_data


def test_cashflow_date_is_parsed_as_datetime():
    df = load_cashflow_data(nrows=20)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])

