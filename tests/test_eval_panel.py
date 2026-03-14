import pandas as pd
import pytest

from foresight.eval_forecast import eval_model


def test_eval_model_on_panel_dataset_aggregates_series(tmp_path):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)

    weeks = pd.date_range("2020-01-01", periods=10, freq="W-WED")
    rows = []
    for store, dept, value in [(1, 1, 1.0), (1, 2, 2.0)]:
        for w in weeks:
            rows.append(
                {"store": store, "dept": dept, "week": w.date().isoformat(), "sales": value}
            )
    df = pd.DataFrame(rows)
    (root / "data" / "store_sales.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    out = eval_model(
        model="naive-last",
        dataset="store_sales",
        y_col="sales",
        horizon=2,
        step=2,
        min_train_size=4,
        max_windows=2,
        data_dir=str(root),
    )
    assert out["n_series"] == 2
    assert out["mae"] == pytest.approx(0.0)
