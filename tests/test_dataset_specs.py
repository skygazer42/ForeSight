from foresight.datasets.registry import get_dataset_spec


def test_dataset_spec_contains_expected_fields():
    spec = get_dataset_spec("store_sales")
    assert spec.key == "store_sales"
    assert "sales" in spec.expected_columns
    assert str(spec.rel_path).endswith("data/store_sales.csv")

