from __future__ import annotations

import pandas as pd

from foresight.adapters import from_gluonts_bundle, to_gluonts_bundle


def _build_long_df() -> pd.DataFrame:
    long_df = pd.DataFrame(
        {
            "unique_id": ["store_a", "store_a", "store_b", "store_b"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "y": [10.0, 11.0, 20.0, 21.0],
            "stock": [4.0, 5.0, 7.0, 8.0],
            "promo": [0.0, 1.0, 1.0, 0.0],
            "store_size": [100.0, 100.0, 150.0, 150.0],
        }
    )
    long_df.attrs["historic_x_cols"] = ("stock",)
    long_df.attrs["future_x_cols"] = ("promo",)
    long_df.attrs["static_cols"] = ("store_size",)
    return long_df


def main() -> None:
    """
    GluonTS richer bundle example.

    Run after installing:
        pip install "foresight-ts[gluonts]"
    """

    long_df = _build_long_df()
    bundle = to_gluonts_bundle(long_df)
    restored = from_gluonts_bundle(bundle)

    print(sorted(bundle))
    print(restored.to_string(index=False))
    print(restored.attrs)


if __name__ == "__main__":
    main()
