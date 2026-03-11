from __future__ import annotations

import pandas as pd
import pytest

from foresight.eval_forecast import eval_model_long_df
from foresight.forecast import forecast_model_long_df


def test_forecast_and_eval_share_long_df_error_message() -> None:
    bad = pd.DataFrame({"unique_id": ["a"], "ds": [1]})

    with pytest.raises(KeyError, match=r"long_df missing required columns: \['y'\]"):
        forecast_model_long_df(model="naive-last", long_df=bad, horizon=1)

    with pytest.raises(KeyError, match=r"long_df missing required columns: \['y'\]"):
        eval_model_long_df(
            model="naive-last",
            long_df=bad,
            horizon=1,
            step=1,
            min_train_size=1,
        )
