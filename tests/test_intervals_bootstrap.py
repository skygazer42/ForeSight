import numpy as np

from foresight.intervals import bootstrap_intervals
from foresight.models.naive import naive_last


def test_bootstrap_intervals_degenerate_when_residuals_zero():
    y = np.ones(12, dtype=float)
    out = bootstrap_intervals(
        y,
        horizon=3,
        forecaster=naive_last,
        min_train_size=5,
        n_samples=200,
        quantiles=(0.1, 0.9),
        seed=0,
    )
    assert out["yhat"].tolist() == [1.0, 1.0, 1.0]
    assert out["lower"].tolist() == [1.0, 1.0, 1.0]
    assert out["upper"].tolist() == [1.0, 1.0, 1.0]
