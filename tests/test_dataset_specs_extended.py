from foresight.datasets.registry import get_dataset_spec


def test_dataset_spec_includes_time_group_and_default_y():
    spec = get_dataset_spec("store_sales")
    assert spec.time_col == "week"
    assert spec.default_y == "sales"
    assert spec.group_cols == ("store", "dept")
