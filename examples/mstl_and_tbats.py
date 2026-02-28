from __future__ import annotations

import numpy as np

from foresight.models.registry import make_forecaster


def main() -> None:
    """
    Demo: multi-seasonal decomposition + TBATS-style baselines (optional statsmodels).

    Run:
      pip install -e ".[dev,stats]"
      python examples/mstl_and_tbats.py
    """
    try:
        import statsmodels  # noqa: F401
    except Exception:
        print('This example requires statsmodels. Install with: pip install -e ".[dev,stats]"')
        return

    rng = np.random.default_rng(0)

    # Dense series with two seasonalities: weekly + ~monthly
    t = np.arange(400, dtype=float)
    y = (
        0.01 * t
        + 1.5 * np.sin(2.0 * np.pi * t / 7.0)
        + 0.8 * np.sin(2.0 * np.pi * t / 30.0)
        + 0.2 * rng.standard_normal(t.shape[0])
    )

    horizon = 21

    models = [
        ("mstl-arima", {"periods": (7, 30), "order": (1, 0, 0)}),
        ("mstl-auto-arima", {"periods": (7, 30), "max_p": 1, "max_d": 1, "max_q": 1}),
        ("tbats-lite", {"periods": (7, 30), "orders": (3, 2), "arima_order": (1, 0, 0)}),
    ]

    for key, params in models:
        f = make_forecaster(key, **params)
        yhat = f(y, horizon)
        print(f"\n=== {key} ===")
        print(np.round(yhat[:10], 3))


if __name__ == "__main__":
    main()
