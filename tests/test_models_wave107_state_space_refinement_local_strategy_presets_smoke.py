import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("key", "overrides"),
    (
        ("torch-s4-ema-direct", {"warmup_epochs": 1}),
        ("torch-s4d-swa-direct", {"swa_start_epoch": 0}),
        ("torch-s5-sam-direct", {"warmup_epochs": 1}),
        ("torch-mamba2-regularized-direct", {"warmup_epochs": 1}),
        ("torch-timesmamba-lookahead-direct", {"warmup_epochs": 1}),
        ("torch-pathformer-swa-direct", {"swa_start_epoch": 0}),
    ),
)
def test_wave107_state_space_refinement_local_strategy_presets_smoke(
    key: str, overrides: dict[str, int]
) -> None:
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
