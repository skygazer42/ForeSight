import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import make_multivariate_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("key", "overrides"),
    (
        ("torch-stid-ema-multivariate", {}),
        ("torch-stgcn-swa-multivariate", {"swa_start_epoch": 0}),
        ("torch-graphwavenet-sam-multivariate", {}),
        ("torch-astgcn-regularized-multivariate", {"warmup_epochs": 1}),
        ("torch-agcrn-longhorizon-multivariate", {}),
        ("torch-stemgnn-lookahead-multivariate", {}),
    ),
)
def test_multivariate_strategy_presets_smoke(
    key: str, overrides: dict[str, int]
) -> None:
    t = np.arange(96.0)
    train = pd.DataFrame(
        {
            "n0": np.sin(t / 6.0) + 0.01 * t,
            "n1": np.cos(t / 7.0) + 0.02 * t,
            "n2": np.sin(t / 9.0 + 0.5) + 0.015 * t,
            "n3": np.cos(t / 8.0 + 0.2) + 0.005 * t,
        }
    )

    forecaster = make_multivariate_forecaster(
        key,
        epochs=2,
        batch_size=16,
        device="cpu",
        seed=0,
        patience=2,
        **overrides,
    )
    fc = forecaster(train, horizon=3)

    assert isinstance(fc, np.ndarray)
    assert fc.shape == (3, 4)
    assert np.all(np.isfinite(fc))
