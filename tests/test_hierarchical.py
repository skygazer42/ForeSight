import pandas as pd
import pytest

from foresight.data.format import build_hierarchy_spec, to_long
from foresight.eval_forecast import eval_hierarchical_forecast_df
from foresight.hierarchical import check_hierarchical_consistency, reconcile_hierarchical_forecasts


def _make_history_raw() -> pd.DataFrame:
    rows = []
    for ds in pd.date_range("2020-01-01", periods=2, freq="D"):
        rows.extend(
            [
                {"ds": ds, "region": "north", "store": "a", "sales": 30.0},
                {"ds": ds, "region": "north", "store": "b", "sales": 10.0},
                {"ds": ds, "region": "south", "store": "c", "sales": 60.0},
            ]
        )
    return pd.DataFrame(rows)


def _yhat_at(df: pd.DataFrame, unique_id: str, ds: str) -> float:
    row = df.loc[(df["unique_id"] == unique_id) & (df["ds"] == pd.Timestamp(ds)), "yhat"]
    assert len(row) == 1
    return float(row.iloc[0])


def test_build_hierarchy_spec_from_id_columns_returns_parent_child_edges():
    raw = _make_history_raw()

    hierarchy = build_hierarchy_spec(raw, id_cols=("region", "store"), root="total")

    assert hierarchy["total"] == ("region=north", "region=south")
    assert hierarchy["region=north"] == ("region=north|store=a", "region=north|store=b")
    assert hierarchy["region=south"] == ("region=south|store=c",)


def test_bottom_up_reconciliation_rolls_leaf_forecasts_to_parents():
    raw = _make_history_raw()
    hierarchy = build_hierarchy_spec(raw, id_cols=("region", "store"), root="total")

    forecast_df = pd.DataFrame(
        {
            "unique_id": [
                "region=north|store=a",
                "region=north|store=b",
                "region=south|store=c",
                "region=north",
                "region=south",
                "total",
            ],
            "ds": [pd.Timestamp("2020-01-03")] * 6,
            "yhat": [12.0, 8.0, 30.0, 999.0, 999.0, 999.0],
        }
    )

    out = reconcile_hierarchical_forecasts(
        forecast_df=forecast_df,
        hierarchy=hierarchy,
        method="bottom_up",
    )

    assert _yhat_at(out, "region=north|store=a", "2020-01-03") == pytest.approx(12.0)
    assert _yhat_at(out, "region=north|store=b", "2020-01-03") == pytest.approx(8.0)
    assert _yhat_at(out, "region=south|store=c", "2020-01-03") == pytest.approx(30.0)
    assert _yhat_at(out, "region=north", "2020-01-03") == pytest.approx(20.0)
    assert _yhat_at(out, "region=south", "2020-01-03") == pytest.approx(30.0)
    assert _yhat_at(out, "total", "2020-01-03") == pytest.approx(50.0)

    consistency = check_hierarchical_consistency(out, hierarchy=hierarchy)
    assert consistency["is_consistent"] is True
    assert consistency["n_inconsistencies"] == 0


def test_top_down_reconciliation_splits_parent_forecasts_using_historical_proportions():
    raw = _make_history_raw()
    history_df = to_long(raw, time_col="ds", y_col="sales", id_cols=("region", "store"))
    hierarchy = build_hierarchy_spec(raw, id_cols=("region", "store"), root="total")

    forecast_df = pd.DataFrame(
        {
            "unique_id": ["total"],
            "ds": [pd.Timestamp("2020-01-03")],
            "yhat": [50.0],
        }
    )

    out = reconcile_hierarchical_forecasts(
        forecast_df=forecast_df,
        hierarchy=hierarchy,
        method="top_down",
        history_df=history_df,
    )

    assert _yhat_at(out, "total", "2020-01-03") == pytest.approx(50.0)
    assert _yhat_at(out, "region=north", "2020-01-03") == pytest.approx(20.0)
    assert _yhat_at(out, "region=south", "2020-01-03") == pytest.approx(30.0)
    assert _yhat_at(out, "region=north|store=a", "2020-01-03") == pytest.approx(15.0)
    assert _yhat_at(out, "region=north|store=b", "2020-01-03") == pytest.approx(5.0)
    assert _yhat_at(out, "region=south|store=c", "2020-01-03") == pytest.approx(30.0)


def test_eval_hierarchical_forecast_df_reports_consistency_summary():
    raw = _make_history_raw()
    history_df = to_long(raw, time_col="ds", y_col="sales", id_cols=("region", "store"))
    hierarchy = build_hierarchy_spec(raw, id_cols=("region", "store"), root="total")

    forecast_df = pd.DataFrame(
        {
            "unique_id": ["total"],
            "ds": [pd.Timestamp("2020-01-03")],
            "yhat": [50.0],
        }
    )

    payload = eval_hierarchical_forecast_df(
        forecast_df=forecast_df,
        hierarchy=hierarchy,
        method="top_down",
        history_df=history_df,
    )

    assert payload["method"] == "top_down"
    assert payload["is_consistent"] is True
    assert payload["n_inconsistencies"] == 0
    assert payload["n_series"] == 6
