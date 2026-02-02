from __future__ import annotations


_DATASETS: dict[str, str] = {
    "store_sales": "Retail weekly store/department sales (data/store_sales.csv)",
    "promotion_data": "Retail weekly promotions (data/promotion_data.csv)",
    "cashflow_data": "Cashflow sample data (data/cashflow_data.csv)",
    "catfish": "Catfish sales (statistics time series/catfish.csv)",
    "ice_cream_interest": "Ice cream interest (statistics time series/ice_cream_interest.csv)",
}


def list_datasets() -> list[str]:
    return sorted(_DATASETS.keys())


def describe_dataset(key: str) -> str:
    try:
        return _DATASETS[key]
    except KeyError as e:
        raise KeyError(f"Unknown dataset key: {key!r}. Try one of: {', '.join(list_datasets())}") from e

