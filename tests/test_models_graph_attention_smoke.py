import importlib.util
import re

import numpy as np
import pytest

from foresight.models.registry import make_multivariate_forecaster


def test_graph_attention_scaffold_module_imports() -> None:
    assert importlib.util.find_spec("foresight.models.torch_graph_attention") is not None


@pytest.mark.parametrize(
    "key",
    (
        "torch-astgcn-multivariate",
        "torch-gman-multivariate",
    ),
)
def test_graph_attention_models_smoke_when_torch_installed(key: str) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    t = np.arange(96.0)
    train = np.stack(
        [
            np.sin(t / 6.0) + 0.01 * t,
            np.cos(t / 7.0) + 0.02 * t,
            np.sin(t / 8.0 + 0.4) + 0.015 * t,
            np.cos(t / 9.0 + 0.1) + 0.012 * t,
        ],
        axis=1,
    )

    forecaster = make_multivariate_forecaster(
        key,
        lags=24,
        d_model=16,
        num_heads=4,
        epochs=2,
        batch_size=16,
        seed=0,
        device="cpu",
    )
    yhat = forecaster(train, 3)

    assert yhat.shape == (3, 4)
    assert np.all(np.isfinite(yhat))


@pytest.mark.parametrize(
    "key",
    (
        "torch-astgcn-multivariate",
        "torch-gman-multivariate",
    ),
)
@pytest.mark.parametrize(
    ("param_name", "param_value", "message"),
    (
        ("horizon_loss_decay", 0.0, "horizon_loss_decay must be > 0"),
        ("sam_rho", -0.1, "sam_rho must be >= 0"),
    ),
)
def test_graph_attention_models_forward_shared_training_validation(
    key: str,
    param_name: str,
    param_value: float,
    message: str,
) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    t = np.arange(96.0)
    train = np.stack(
        [
            np.sin(t / 6.0) + 0.01 * t,
            np.cos(t / 7.0) + 0.02 * t,
            np.sin(t / 8.0 + 0.4) + 0.015 * t,
            np.cos(t / 9.0 + 0.1) + 0.012 * t,
        ],
        axis=1,
    )

    forecaster = make_multivariate_forecaster(
        key,
        lags=24,
        d_model=16,
        num_heads=4,
        epochs=2,
        batch_size=16,
        seed=0,
        device="cpu",
        **{param_name: param_value},
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        forecaster(train, 3)
