import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


def test_torch_structured_rnn_scaffold_module_imports() -> None:
    assert importlib.util.find_spec("foresight.models.torch_structured_rnn") is not None


@pytest.mark.parametrize(
    "key",
    (
        "torch-multidim-rnn-direct",
        "torch-grid-lstm-direct",
        "torch-structural-rnn-direct",
    ),
)
def test_torch_structured_rnn_models_smoke_when_torch_installed(key: str) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    y = np.sin(np.arange(96, dtype=float) / 6.0) + 0.02 * np.arange(96, dtype=float)
    forecaster = make_forecaster(
        key,
        lags=24,
        hidden_size=16,
        epochs=2,
        batch_size=16,
        seed=0,
        device="cpu",
    )
    yhat = forecaster(y, 3)

    assert yhat.shape == (3,)
    assert np.all(np.isfinite(yhat))
