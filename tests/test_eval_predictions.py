import foresight.eval_predictions as eval_predictions_mod
import numpy as np
import pytest
from foresight.cv import cross_validation_predictions
from foresight.eval_predictions import (
    _mean_by_step_from_inverse,
    _validated_interval_arrays,
    _interval_score_vector,
    _step_group_inverse_counts,
    _vectorized_interval_metrics,
    _vectorized_pinball_summary,
    _weighted_interval_score_by_step,
    evaluate_predictions,
    evaluate_quantile_predictions,
)


def test_step_group_inverse_counts_returns_sorted_steps_inverse_and_counts() -> None:
    steps, inverse, counts = _step_group_inverse_counts([2, 1, 2, 3, 1])

    assert steps == [1, 2, 3]
    assert inverse.tolist() == [1, 0, 1, 2, 0]
    assert counts.tolist() == [2.0, 2.0, 1.0]


def test_interval_score_vector_returns_width_penalty_sum() -> None:
    score = _interval_score_vector(
        y=[0.0, 1.0, 2.0, 3.0],
        lo=[-1.0, 0.0, 1.0, 4.0],
        hi=[1.0, 2.0, 3.0, 5.0],
        alpha=0.2,
    )

    assert score.tolist() == pytest.approx([2.0, 2.0, 2.0, 11.0])


def test_validated_interval_arrays_returns_float_arrays() -> None:
    y, lo, hi = _validated_interval_arrays(
        y=[0, 1],
        lo=[-1, 0],
        hi=[1, 2],
    )

    assert y.dtype == float
    assert lo.dtype == float
    assert hi.dtype == float
    assert y.tolist() == [0.0, 1.0]


def test_mean_by_step_from_inverse_aggregates_weighted_values() -> None:
    values = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)
    inverse = np.array([0, 1, 0, 1], dtype=int)
    counts = np.array([2.0, 2.0], dtype=float)

    out = _mean_by_step_from_inverse(values, inverse=inverse, counts=counts, n_steps=2)

    assert out.tolist() == pytest.approx([4.0, 6.0])


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


def test_evaluate_predictions_uses_vectorized_metric_aggregation(monkeypatch):
    df = cross_validation_predictions(
        model="naive-last",
        dataset="catfish",
        y_col="Total",
        horizon=3,
        step_size=3,
        min_train_size=12,
        n_windows=3,
    )

    original_vectorized = eval_predictions_mod._vectorized_point_metrics
    calls = {"count": 0}

    def _counting_vectorized(*args, **kwargs):
        calls["count"] += 1
        return original_vectorized(*args, **kwargs)

    monkeypatch.setattr(eval_predictions_mod, "_vectorized_point_metrics", _counting_vectorized)

    out = evaluate_predictions(df)

    assert out["n_points"] == len(df)
    assert len(out["mae_by_step"]) == 3
    assert calls["count"] == 1


def test_vectorized_pinball_summary_reports_per_quantile_and_crps() -> None:
    y = [0.0, 2.0]
    qhat = [[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]]

    out = _vectorized_pinball_summary(y, qhat, [10, 50, 90])

    assert out["quantiles"] == [10, 50, 90]
    assert out["pinball_p10"] == pytest.approx(0.1)
    assert out["pinball_p50"] == pytest.approx(0.0)
    assert out["pinball_p90"] == pytest.approx(0.1)
    assert out["pinball_mean"] == pytest.approx((0.1 + 0.0 + 0.1) / 3.0)
    assert out["crps"] >= 0.0


def test_vectorized_interval_metrics_reports_overall_and_by_step() -> None:
    out = _vectorized_interval_metrics(
        y=[0.0, 1.0, 2.0, 3.0],
        lo=[-1.0, 0.0, 1.0, 4.0],
        hi=[1.0, 2.0, 3.0, 5.0],
        alpha=0.2,
        step_values=[1, 2, 1, 2],
    )

    assert out["coverage"] == pytest.approx(0.75)
    assert out["mean_width"] == pytest.approx(1.75)
    assert out["interval_score"] >= out["mean_width"]
    assert out["winkler_score"] == pytest.approx(out["interval_score"])
    assert out["coverage_by_step"] == pytest.approx([1.0, 0.5])
    assert out["mean_width_by_step"] == pytest.approx([2.0, 1.5])


def test_evaluate_quantile_predictions_uses_vectorized_pinball_summary(monkeypatch):
    df = cross_validation_predictions(
        model="naive-last",
        dataset="catfish",
        y_col="Total",
        horizon=3,
        step_size=3,
        min_train_size=12,
        n_windows=2,
    ).assign(
        yhat_p10=lambda frame: frame["yhat"] - 1.0,
        yhat_p50=lambda frame: frame["yhat"],
        yhat_p90=lambda frame: frame["yhat"] + 1.0,
    )

    original_pinball_summary = eval_predictions_mod._vectorized_pinball_summary
    calls = {"count": 0}

    def _counting_pinball_summary(*args, **kwargs):
        calls["count"] += 1
        return original_pinball_summary(*args, **kwargs)

    monkeypatch.setattr(
        eval_predictions_mod,
        "_vectorized_pinball_summary",
        _counting_pinball_summary,
    )

    out = evaluate_quantile_predictions(df)

    assert out["quantiles"] == [10, 50, 90]
    assert calls["count"] == 1


def test_weighted_interval_score_by_step_uses_vectorized_aggregation(monkeypatch):
    y = [0.0, 1.0, 2.0, 3.0]
    median = [0.0, 1.0, 2.0, 3.0]
    intervals = [
        ([-1.0, 0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0], 0.2),
        ([-2.0, -1.0, 0.0, 1.0], [2.0, 3.0, 4.0, 5.0], 0.4),
    ]

    def _forbid_weighted_interval_score(*args, **kwargs):
        raise AssertionError("weighted_interval_score should not be called per step")

    monkeypatch.setattr(
        eval_predictions_mod,
        "weighted_interval_score",
        _forbid_weighted_interval_score,
    )

    out = _weighted_interval_score_by_step(
        y,
        median,
        wis_intervals=[
            (eval_predictions_mod.np.asarray(lo, dtype=float), eval_predictions_mod.np.asarray(hi, dtype=float), alpha)
            for lo, hi, alpha in intervals
        ],
        step_values=[1, 2, 1, 2],
    )

    assert len(out) == 2
    assert all(value >= 0.0 for value in out)
