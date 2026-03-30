import pandas as pd
import pytest

from foresight.long_df_cache import sorted_long_df


def test_sorted_long_df_fast_path_skips_sort_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0", "s0", "s1", "s1"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    original_sort_values = pd.DataFrame.sort_values
    calls = {"count": 0}

    def _counting_sort_values(self: pd.DataFrame, *args: object, **kwargs: object) -> pd.DataFrame:
        calls["count"] += 1
        return original_sort_values(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sort_values", _counting_sort_values)

    out = sorted_long_df(long_df, reset_index=False)

    assert calls["count"] == 0
    assert out is long_df


def test_sorted_long_df_sorts_unsorted_rows_and_resets_index() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s1", "s0", "s1", "s0"],
            "ds": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-01", "2020-01-01"]),
            "y": [4.0, 2.0, 3.0, 1.0],
        },
        index=[10, 11, 12, 13],
    )

    out = sorted_long_df(long_df, reset_index=True)

    assert out["unique_id"].tolist() == ["s0", "s0", "s1", "s1"]
    assert out["y"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out.index.tolist() == [0, 1, 2, 3]
