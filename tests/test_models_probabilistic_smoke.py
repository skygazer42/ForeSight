import importlib.util
import re

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


def test_probabilistic_scaffold_module_imports() -> None:
    assert importlib.util.find_spec("foresight.models.torch_probabilistic") is not None


@pytest.mark.parametrize(
    "key",
    (
        "torch-timegrad-direct",
        "torch-tactis-direct",
    ),
)
def test_probabilistic_models_smoke_when_torch_installed(key: str) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    y = np.sin(np.arange(96, dtype=float) / 5.0) + 0.03 * np.arange(96, dtype=float)
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


@pytest.mark.parametrize(
    "key",
    (
        "torch-timegrad-direct",
        "torch-tactis-direct",
    ),
)
@pytest.mark.parametrize(
    ("param_name", "param_value", "message"),
    (
        ("horizon_loss_decay", 0.0, "horizon_loss_decay must be > 0"),
        ("sam_rho", -0.1, "sam_rho must be >= 0"),
    ),
)
def test_probabilistic_models_forward_shared_training_validation(
    key: str,
    param_name: str,
    param_value: float,
    message: str,
) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    y = np.sin(np.arange(96, dtype=float) / 5.0) + 0.03 * np.arange(96, dtype=float)
    forecaster = make_forecaster(
        key,
        lags=24,
        hidden_size=16,
        epochs=2,
        batch_size=16,
        seed=0,
        device="cpu",
        **{param_name: param_value},
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        forecaster(y, 3)
