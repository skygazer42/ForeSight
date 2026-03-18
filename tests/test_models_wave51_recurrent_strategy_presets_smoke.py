import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("key", "overrides"),
    (
        ("torch-rnnpaper-lstm-ema-direct", {}),
        ("torch-rnnpaper-gru-swa-direct", {"swa_start_epoch": 0}),
        ("torch-rnnpaper-qrnn-lookahead-direct", {}),
        ("torch-rnnzoo-lstm-sam-direct", {}),
        ("torch-rnnzoo-gru-regularized-direct", {"warmup_epochs": 1}),
        ("torch-rnnzoo-qrnn-longhorizon-direct", {}),
    ),
)
def test_wave51_recurrent_strategy_presets_smoke(
    key: str, overrides: dict[str, int]
) -> None:
    y = np.sin(np.arange(80, dtype=float) / 4.0) + 0.03 * np.arange(80, dtype=float)
    forecaster = make_forecaster(
        key,
        hidden_size=8,
        lags=16,
        epochs=2,
        batch_size=16,
        patience=1,
        seed=0,
        device="cpu",
        **overrides,
    )
    yhat = forecaster(y, 5)

    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))
