import importlib.util

import numpy as np
import pytest
from foresight.models.registry import make_forecaster


def test_torch_reservoir_scaffold_module_imports() -> None:
    assert importlib.util.find_spec("foresight.models.torch_reservoir") is not None


@pytest.mark.parametrize(
    "key",
    (
        "torch-esn-direct",
        "torch-deep-esn-direct",
        "torch-liquid-state-direct",
    ),
)
def test_torch_reservoir_models_smoke_when_torch_installed(key: str) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    y = np.sin(np.arange(96, dtype=float) / 5.0) + 0.01 * np.arange(96, dtype=float)
    forecaster = make_forecaster(
        key,
        lags=24,
        hidden_size=16,
        spectral_radius=0.8,
        leak=0.9,
        epochs=2,
        batch_size=16,
        seed=0,
        device="cpu",
    )
    yhat = forecaster(y, 3)

    assert yhat.shape == (3,)
    assert np.all(np.isfinite(yhat))
