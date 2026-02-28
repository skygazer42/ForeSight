from __future__ import annotations

import numpy as np

from foresight.models.registry import make_forecaster


def main() -> None:
    """
    Demo: a few additional algorithms in the model zoo.

    Run:
      pip install -e ".[dev]"
      python examples/more_algorithms.py
    """
    rng = np.random.default_rng(0)

    # Dense series: trend + multiple seasonalities
    t = np.arange(365, dtype=float)
    y_dense = (
        0.01 * t
        + 2.0 * np.sin(2.0 * np.pi * t / 7.0)
        + 1.0 * np.sin(2.0 * np.pi * t / 30.0)
        + 0.2 * rng.standard_normal(t.shape[0])
    )

    horizon = 14
    for key, params in [
        ("fourier-multi", {"periods": (7, 30), "orders": (3, 2), "include_trend": True}),
        ("fft", {"top_k": 5, "include_trend": True}),
        ("kalman-trend", {}),
        ("analog-knn", {"lags": 28, "k": 10, "normalize": True, "weights": "distance"}),
    ]:
        f = make_forecaster(key, **params)
        yhat = f(y_dense, horizon)
        print(f"\n=== {key} (dense) ===")
        print(np.round(yhat[:7], 3))

    # Intermittent / sparse demand series
    y_sparse = np.array([0, 0, 5, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 4], dtype=float)
    horizon = 10
    for key, params in [
        ("croston-sba", {"alpha": 0.2}),
        ("croston-opt", {"grid_size": 11}),
        ("les", {"alpha": 0.2, "beta": 0.2}),
        ("tsb", {"alpha": 0.2, "beta": 0.05}),
    ]:
        f = make_forecaster(key, **params)
        yhat = f(y_sparse, horizon)
        print(f"\n=== {key} (sparse) ===")
        print(np.round(yhat, 3))


if __name__ == "__main__":
    main()
