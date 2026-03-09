import numpy as np

from foresight.models.registry import make_forecaster


def test_pipeline_diff1_naive_last_behaves_like_drift():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    f = make_forecaster("pipeline", base="naive-last", transforms=("diff1",))
    yhat = f(y, 3)
    assert yhat.shape == (3,)
    assert np.allclose(yhat, np.array([6.0, 7.0, 8.0]))


def test_pipeline_standardize_naive_last_preserves_last_value_after_inverse():
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    f = make_forecaster("pipeline", base="naive-last", transforms=("standardize",))
    yhat = f(y, 2)
    assert yhat.shape == (2,)
    assert np.allclose(yhat, np.array([10.0, 10.0]))
