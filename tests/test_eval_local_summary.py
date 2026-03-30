import numpy as np
import pytest

from foresight.services.evaluation import (
    _new_eval_metric_state,
    _summarize_eval_model_long_df_results,
    _update_eval_metric_state,
)


def test_local_eval_summary_supports_metric_state_without_raw_arrays() -> None:
    metric_state = _new_eval_metric_state(horizon=2)
    _update_eval_metric_state(
        metric_state,
        y_true=np.array([[1.0, 2.0], [3.0, 4.0]]),
        y_pred=np.array([[1.0, 1.0], [2.0, 5.0]]),
    )

    payload = _summarize_eval_model_long_df_results(
        model="naive-last",
        horizon=2,
        step=1,
        min_train_size=4,
        max_windows=2,
        max_train_size=None,
        conformal_levels=None,
        conformal_per_step=True,
        results={
            "metric_state": metric_state,
            "collect_raw_arrays": False,
            "y_true_by_step": [[], []],
            "y_pred_by_step": [[], []],
            "n_series": 1,
            "n_series_skipped": 0,
            "n_windows": 2,
        },
    )

    assert payload["n_points"] == 4
    assert payload["mae"] == pytest.approx(0.75)
    assert payload["rmse"] == pytest.approx(np.sqrt(0.75))
    assert payload["mae_by_step"] == pytest.approx([0.5, 1.0])
