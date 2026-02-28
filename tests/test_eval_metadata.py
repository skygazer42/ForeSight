from foresight.eval import eval_naive_last


def test_eval_includes_basic_metadata():
    out = eval_naive_last(
        dataset="catfish",
        y_col="Total",
        horizon=3,
        step=3,
        min_train_size=12,
    )
    assert "n_obs" in out
    assert "n_points" in out
