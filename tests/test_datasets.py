from foresight.datasets.registry import list_datasets
from foresight.datasets.loaders import load_store_sales


def test_list_datasets_contains_store_sales():
    assert "store_sales" in list_datasets()


def test_load_store_sales_smoke():
    df = load_store_sales(nrows=500)
    assert {"store", "dept", "week", "sales"}.issubset(df.columns)
    assert len(df) > 0
