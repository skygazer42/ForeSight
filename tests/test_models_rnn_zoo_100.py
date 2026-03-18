import importlib.util
import re

import numpy as np
import pytest

from foresight.models.registry import get_model_spec, list_models, make_forecaster


def _rnnzoo_bases() -> list[str]:
    return [
        "elman",
        "lstm",
        "gru",
        "peephole-lstm",
        "cifg-lstm",
        "janet",
        "indrnn",
        "minimalrnn",
        "mgu",
        "fastrnn",
        "fastgrnn",
        "mut1",
        "mut2",
        "mut3",
        "ran",
        "scrn",
        "rhn",
        "clockwork",
        "qrnn",
        "phased-lstm",
    ]


def _rnnzoo_expected_keys() -> list[str]:
    keys: list[str] = []
    for base in _rnnzoo_bases():
        keys.append(f"torch-rnnzoo-{base}-direct")
        keys.append(f"torch-rnnzoo-{base}-bidir-direct")
        keys.append(f"torch-rnnzoo-{base}-ln-direct")
        keys.append(f"torch-rnnzoo-{base}-attn-direct")
        keys.append(f"torch-rnnzoo-{base}-proj-direct")
    return keys


def test_rnnzoo_100_models_are_registered():
    keys = set(list_models())
    expected = _rnnzoo_expected_keys()
    missing = [k for k in expected if k not in keys]
    assert not missing, f"Missing {len(missing)} rnnzoo models, e.g. {missing[:5]}"


def test_rnnzoo_100_models_are_marked_optional_torch():
    for key in _rnnzoo_expected_keys():
        spec = get_model_spec(key)
        assert "torch" in spec.requires


@pytest.mark.parametrize(
    "key",
    [
        "torch-rnnzoo-elman-direct",
        "torch-rnnzoo-peephole-lstm-ln-direct",
        "torch-rnnzoo-indrnn-attn-direct",
        "torch-rnnzoo-qrnn-proj-direct",
        "torch-rnnzoo-phased-lstm-bidir-direct",
    ],
)
def test_rnnzoo_models_smoke_when_torch_installed(key: str):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    y = np.sin(np.arange(80, dtype=float) / 4.0) + 0.03 * np.arange(80, dtype=float)
    f = make_forecaster(
        key,
        lags=16,
        hidden_size=8,
        epochs=2,
        batch_size=16,
        patience=1,
        seed=0,
        device="cpu",
    )
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.parametrize(
    "key",
    [
        "torch-rnnzoo-elman-direct",
        "torch-rnnzoo-peephole-lstm-ln-direct",
        "torch-rnnzoo-indrnn-attn-direct",
        "torch-rnnzoo-qrnn-proj-direct",
        "torch-rnnzoo-phased-lstm-bidir-direct",
    ],
)
@pytest.mark.parametrize(
    ("param_name", "param_value", "message"),
    (
        ("horizon_loss_decay", 0.0, "horizon_loss_decay must be > 0"),
        ("sam_rho", -0.1, "sam_rho must be >= 0"),
    ),
)
def test_rnnzoo_models_forward_shared_training_validation(
    key: str,
    param_name: str,
    param_value: float,
    message: str,
):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    y = np.sin(np.arange(80, dtype=float) / 4.0) + 0.03 * np.arange(80, dtype=float)
    f = make_forecaster(
        key,
        lags=16,
        hidden_size=8,
        epochs=2,
        batch_size=16,
        patience=1,
        seed=0,
        device="cpu",
        **{param_name: param_value},
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        f(y, 5)
