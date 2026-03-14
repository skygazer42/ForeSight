from __future__ import annotations

import pandas as pd

from foresight.services.forecasting import _as_datetime_index


def test_as_datetime_index_preserves_datetime_index_inputs() -> None:
    ds = pd.date_range("2024-01-01", periods=3, freq="D")

    out = _as_datetime_index(ds)

    assert out is ds
