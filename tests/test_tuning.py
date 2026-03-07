import pandas as pd

from foresight.tuning import tune_model, tune_model_long_df


def _trend_long_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["s0"] * 10,
            "ds": pd.date_range("2020-01-01", periods=10, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )


def test_tune_model_long_df_selects_best_config_by_mae() -> None:
    result = tune_model_long_df(
        model="moving-average",
        long_df=_trend_long_df(),
        horizon=1,
        step=1,
        min_train_size=4,
        search_space={"window": (1, 3)},
        metric="mae",
        max_windows=3,
    )

    assert result["model"] == "moving-average"
    assert result["metric"] == "mae"
    assert result["n_trials"] == 2
    assert result["best_params"] == {"window": 1}
    assert result["best_score"] == result["trials"][0]["score"]
    assert result["trials"][0]["params"] == {"window": 1}
    assert result["trials"][0]["score"] <= result["trials"][1]["score"]


def test_tune_model_dataset_wrapper_returns_best_config(tmp_path) -> None:
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)

    weeks = pd.date_range("2020-01-01", periods=10, freq="W-WED")
    df = pd.DataFrame(
        {
            "store": [1] * 10,
            "dept": [1] * 10,
            "week": [d.date().isoformat() for d in weeks],
            "sales": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    (root / "data" / "store_sales.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    result = tune_model(
        model="moving-average",
        dataset="store_sales",
        y_col="sales",
        horizon=1,
        step=1,
        min_train_size=4,
        search_space={"window": (1, 3)},
        metric="mae",
        max_windows=3,
        data_dir=str(root),
    )

    assert result["dataset"] == "store_sales"
    assert result["best_params"] == {"window": 1}
    assert result["best_score"] == result["trials"][0]["score"]
