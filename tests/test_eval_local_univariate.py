import pandas as pd

from foresight.eval_forecast import eval_model_long_df


def test_eval_model_long_df_local_univariate_respects_max_windows() -> None:
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 30,
            "ds": pd.date_range("2020-01-01", periods=30, freq="D"),
            "y": [float(i) for i in range(30)],
        }
    )

    payload = eval_model_long_df(
        model="naive-last",
        long_df=long_df,
        horizon=3,
        step=3,
        min_train_size=12,
        max_windows=2,
    )

    assert payload["model"] == "naive-last"
    assert payload["n_series"] == 1
    assert payload["n_series_skipped"] == 0
    assert payload["n_windows"] == 2
    assert payload["n_points"] == 6
