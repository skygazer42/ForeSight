from foresight.eval import eval_naive_last


def test_eval_includes_metrics_by_step():
    out = eval_naive_last(
        dataset="catfish",
        y_col="Total",
        horizon=4,
        step=4,
        min_train_size=12,
    )
    assert len(out["mae_by_step"]) == 4
    assert len(out["rmse_by_step"]) == 4
