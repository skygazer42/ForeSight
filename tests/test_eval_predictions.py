from foresight.cv import cross_validation_predictions
from foresight.eval_predictions import evaluate_predictions


def test_evaluate_predictions_includes_by_step_metrics():
    df = cross_validation_predictions(
        model="naive-last",
        dataset="catfish",
        y_col="Total",
        horizon=3,
        step_size=3,
        min_train_size=12,
        n_windows=10,
    )
    out = evaluate_predictions(df)
    assert out["n_points"] == len(df)
    assert len(out["mae_by_step"]) == 3
    assert len(out["rmse_by_step"]) == 3
    assert len(out["mape_by_step"]) == 3
    assert len(out["smape_by_step"]) == 3
