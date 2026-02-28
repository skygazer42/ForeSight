import numpy as np

from foresight.models.ar import ar_ols_forecast


def _gen_ar2(n: int) -> np.ndarray:
    y = np.zeros(n, dtype=float)
    y[0] = 1.0
    y[1] = 2.0
    for t in range(2, n):
        y[t] = 0.3 * y[t - 1] + 0.2 * y[t - 2]
    return y


def test_ar_ols_forecast_matches_noiseless_ar2():
    y = _gen_ar2(60)
    train = y[:50]
    true_future = y[50:53]
    pred = ar_ols_forecast(train, horizon=3, p=2)
    assert np.allclose(pred, true_future, atol=1e-9)
