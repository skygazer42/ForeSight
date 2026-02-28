from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    description: str
    rel_path: Path
    expected_columns: set[str]
    parse_dates: list[str]


_SPECS: dict[str, DatasetSpec] = {
    "store_sales": DatasetSpec(
        key="store_sales",
        description="Retail weekly store/department sales",
        rel_path=Path("data/store_sales.csv"),
        expected_columns={"store", "dept", "week", "sales"},
        parse_dates=["week"],
    ),
    "promotion_data": DatasetSpec(
        key="promotion_data",
        description="Retail weekly promotions",
        rel_path=Path("data/promotion_data.csv"),
        expected_columns={"store", "dept", "week", "promotion_sales"},
        parse_dates=["week"],
    ),
    "cashflow_data": DatasetSpec(
        key="cashflow_data",
        description="Cashflow sample data",
        rel_path=Path("data/cashflow_data.csv"),
        expected_columns={
            "date",
            "cashflow_category",
            "cashflow_subcategory",
            "cashflow",
            "branch_id",
        },
        parse_dates=["date"],
    ),
    "catfish": DatasetSpec(
        key="catfish",
        description="Catfish sales",
        rel_path=Path("statistics time series/catfish.csv"),
        expected_columns={"Date", "Total"},
        parse_dates=["Date"],
    ),
    "ice_cream_interest": DatasetSpec(
        key="ice_cream_interest",
        description="Ice cream interest",
        rel_path=Path("statistics time series/ice_cream_interest.csv"),
        expected_columns={"month", "interest"},
        parse_dates=["month"],
    ),
}


def list_datasets() -> list[str]:
    return sorted(_SPECS.keys())


def get_dataset_spec(key: str) -> DatasetSpec:
    try:
        return _SPECS[key]
    except KeyError as e:
        raise KeyError(f"Unknown dataset key: {key!r}. Try one of: {', '.join(list_datasets())}") from e


def describe_dataset(key: str) -> str:
    return get_dataset_spec(key).description
