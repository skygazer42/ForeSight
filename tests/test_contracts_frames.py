from __future__ import annotations

import pandas as pd
import pytest

from foresight.contracts.frames import require_future_df, require_long_df


def test_require_long_df_rejects_missing_y() -> None:
    bad = pd.DataFrame({"unique_id": ["a"], "ds": [1]})

    with pytest.raises(KeyError, match=r"long_df missing required columns: \['y'\]"):
        require_long_df(bad)


def test_require_future_df_fills_nan_y_column() -> None:
    out = require_future_df(pd.DataFrame({"unique_id": ["a"], "ds": [1]}))

    assert "y" in out.columns
    assert out["y"].isna().all()
