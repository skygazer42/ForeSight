import pytest

from foresight.datasets.loaders import load_dataset


def test_load_dataset_unknown_key_mentions_available_keys():
    with pytest.raises(KeyError) as ei:
        load_dataset("no_such_dataset")
    msg = str(ei.value)
    assert "Unknown dataset key" in msg
    assert "store_sales" in msg

