import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("key", "overrides"),
    (
        ("torch-nhits-ema-direct", {"warmup_epochs": 1}),
        ("torch-nbeats-sam-direct", {"warmup_epochs": 1}),
        ("torch-tide-regularized-direct", {"warmup_epochs": 1}),
        ("torch-dlinear-lookahead-direct", {"warmup_epochs": 1}),
        ("torch-nlinear-longhorizon-direct", {"warmup_epochs": 1}),
        ("torch-timemixer-ema-direct", {"warmup_epochs": 1}),
    ),
)
def test_wave103_baseline_local_strategy_presets_smoke(key: str, overrides: dict[str, int]) -> None:
    y = np.sin(np.arange(128, dtype=float) / 6.0) + 0.02 * np.arange(128, dtype=float)
    forecaster = make_forecaster(
        key,
        lags=48,
        epochs=1,
        batch_size=16,
        patience=1,
        val_split=0.0,
        device="cpu",
        seed=0,
        **overrides,
    )
    yhat = forecaster(y, 5)

    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))
