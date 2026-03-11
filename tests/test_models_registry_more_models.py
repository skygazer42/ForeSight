import numpy as np

from foresight.models.registry import make_forecaster


def test_new_core_models_smoke():
    y_dense = np.sin(np.arange(80, dtype=float) / 4.0) + 0.1 * np.arange(80, dtype=float)
    y_sparse = np.array([0, 0, 5, 0, 0, 0, 3, 0, 0, 2, 0, 0], dtype=float)

    cases = [
        ("theta-auto", {}, y_dense),
        ("fourier-multi", {"periods": (7, 30), "orders": (3, 2)}, y_dense),
        ("fft", {"top_k": 3}, y_dense),
        ("analog-knn", {"lags": 12, "k": 5}, y_dense),
        ("kalman-level", {}, y_dense),
        ("kalman-trend", {}, y_dense),
        ("croston-sba", {"alpha": 0.2}, y_sparse),
        ("croston-sbj", {"alpha": 0.2}, y_sparse),
        ("croston-opt", {"grid_size": 7}, y_sparse),
        ("les", {"alpha": 0.2, "beta": 0.2}, y_sparse),
    ]

    for key, params, y in cases:
        f = make_forecaster(key, **params)
        yhat = f(y, 3)
        assert yhat.shape == (3,)
        assert np.all(np.isfinite(yhat))


def test_foundation_wrapper_catalog_models_smoke(tmp_path):
    checkpoint = tmp_path / "foundation-fixture.json"
    checkpoint.write_text('{"bias": 1.5, "scale": 1.0, "use_trend": true}', encoding="utf-8")
    y = np.array([1.0, 2.0, 4.0], dtype=float)

    for key in ("lag-llama", "moirai"):
        f = make_forecaster(key, backend="fixture-json", checkpoint_path=str(checkpoint))
        yhat = f(y, 2)
        assert yhat.shape == (2,)
        assert np.all(np.isfinite(yhat))
