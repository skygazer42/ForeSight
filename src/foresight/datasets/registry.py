from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    description: str
    rel_path: Path
    expected_columns: set[str]
    parse_dates: list[str]
    time_col: str
    default_y: str
    group_cols: tuple[str, ...] = ()
    package_rel_path: Path | None = None


_SPECS: dict[str, DatasetSpec] = {
    "store_sales": DatasetSpec(
        key="store_sales",
        description="Retail weekly store/department sales",
        rel_path=Path("data/store_sales.csv"),
        expected_columns={"store", "dept", "week", "sales"},
        parse_dates=["week"],
        time_col="week",
        default_y="sales",
        group_cols=("store", "dept"),
    ),
    "promotion_data": DatasetSpec(
        key="promotion_data",
        description="Retail weekly promotions",
        rel_path=Path("data/promotion_data.csv"),
        expected_columns={"store", "dept", "week", "promotion_sales"},
        parse_dates=["week"],
        time_col="week",
        default_y="promotion_sales",
        group_cols=("store", "dept"),
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
        time_col="date",
        default_y="cashflow",
        group_cols=("branch_id", "cashflow_category", "cashflow_subcategory"),
    ),
    "catfish": DatasetSpec(
        key="catfish",
        description="Catfish sales",
        rel_path=Path("statistics time series/catfish.csv"),
        package_rel_path=Path("data/catfish.csv"),
        expected_columns={"Date", "Total"},
        parse_dates=["Date"],
        time_col="Date",
        default_y="Total",
        group_cols=(),
    ),
    "ice_cream_interest": DatasetSpec(
        key="ice_cream_interest",
        description="Ice cream interest",
        rel_path=Path("statistics time series/ice_cream_interest.csv"),
        package_rel_path=Path("data/ice_cream_interest.csv"),
        expected_columns={"month", "interest"},
        parse_dates=["month"],
        time_col="month",
        default_y="interest",
        group_cols=(),
    ),
}


def list_datasets() -> list[str]:
    return sorted(_SPECS.keys())


def list_packaged_datasets() -> list[str]:
    return sorted(key for key, spec in _SPECS.items() if spec.package_rel_path is not None)


def get_dataset_spec(key: str) -> DatasetSpec:
    try:
        return _SPECS[key]
    except KeyError as e:
        raise KeyError(
            f"Unknown dataset key: {key!r}. Try one of: {', '.join(list_datasets())}"
        ) from e


def describe_dataset(key: str) -> str:
    return get_dataset_spec(key).description


def resolve_dataset_path_info(
    key: str,
    *,
    data_dir: str | Path | None = None,
) -> tuple[Path, str]:
    spec = get_dataset_spec(key)

    if isinstance(data_dir, str) and not data_dir.strip():
        data_dir = None
    if data_dir is not None:
        base = Path(data_dir).expanduser()
        return (base / spec.rel_path).resolve(), "data_dir"

    env_dir = os.environ.get("FORESIGHT_DATA_DIR", "").strip()
    if env_dir:
        base = Path(env_dir).expanduser()
        return (base / spec.rel_path).resolve(), "env"

    # Installed-package fallback: for a small subset of datasets we ship CSVs
    # under `foresight/data/` to support `pip install` demos.
    if spec.package_rel_path is not None:
        pkg_root = Path(__file__).resolve().parents[1]  # foresight/
        pkg_path = (pkg_root / spec.package_rel_path).resolve()
        if pkg_path.exists():
            return pkg_path, "package"

    # Dev fallback: when running from the repo, `parents[3]` resolves repo root.
    repo_root = Path(__file__).resolve().parents[3]
    repo_path = (repo_root / spec.rel_path).resolve()
    if repo_path.exists():
        return repo_path, "repo"

    raise FileNotFoundError(
        f"Dataset file not found for key={key!r}. "
        "Provide `--data-dir ...` or set FORESIGHT_DATA_DIR."
    )


def resolve_dataset_path(key: str, *, data_dir: str | Path | None = None) -> Path:
    path, _source = resolve_dataset_path_info(key, data_dir=data_dir)
    return path


def preview_dataset_path_info(
    key: str,
    *,
    data_dir: str | Path | None = None,
) -> tuple[Path, str, bool]:
    spec = get_dataset_spec(key)

    if isinstance(data_dir, str) and not data_dir.strip():
        data_dir = None

    candidates: list[tuple[Path, str]] = []
    if data_dir is not None:
        base = Path(data_dir).expanduser()
        candidates.append(((base / spec.rel_path).resolve(), "data_dir"))
    else:
        env_dir = os.environ.get("FORESIGHT_DATA_DIR", "").strip()
        if env_dir:
            base = Path(env_dir).expanduser()
            candidates.append(((base / spec.rel_path).resolve(), "env"))
        else:
            if spec.package_rel_path is not None:
                pkg_root = Path(__file__).resolve().parents[1]
                candidates.append(((pkg_root / spec.package_rel_path).resolve(), "package"))

            repo_root = Path(__file__).resolve().parents[3]
            candidates.append(((repo_root / spec.rel_path).resolve(), "repo"))

    for path, source in candidates:
        if path.exists():
            return path, source, True

    if not candidates:
        raise RuntimeError(f"No dataset resolution candidates available for key={key!r}")
    path, source = candidates[0]
    return path, source, False
