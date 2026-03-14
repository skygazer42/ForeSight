from pathlib import Path

import pytest

from foresight.datasets.loaders import load_store_sales


def test_load_store_sales_supports_env_data_dir(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    # Write a tiny dataset with a distinctive value so we can assert the loader
    # actually honors the env var (instead of accidentally reading the repo file).
    (root / "data" / "store_sales.csv").write_text(
        "store,dept,week,sales\n1,1,2010-02-01,999.0\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("FORESIGHT_DATA_DIR", str(root))
    df = load_store_sales(nrows=5)
    assert len(df) == 1
    assert float(df["sales"].iloc[0]) == pytest.approx(999.0)
