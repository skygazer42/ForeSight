from __future__ import annotations

from pathlib import Path

import pytest

from foresight.datasets.loaders import load_dataset
from foresight.datasets.registry import resolve_dataset_path


@pytest.mark.parametrize(
    "key,expected_suffix",
    [("catfish", "catfish.csv"), ("ice_cream_interest", "ice_cream_interest.csv")],
)
def test_packaged_demo_datasets_resolve_without_data_dir(
    key: str, expected_suffix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FORESIGHT_DATA_DIR", raising=False)

    p = resolve_dataset_path(key)
    assert isinstance(p, Path)
    assert p.exists()
    assert p.as_posix().endswith(f"/foresight/data/{expected_suffix}")

    df = load_dataset(key, nrows=5)
    assert len(df) > 0
