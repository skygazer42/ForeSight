import importlib.util

import numpy as np
import pytest

from foresight.models.registry import get_model_spec, list_models, make_forecaster


def _rnnpaper_ids() -> list[str]:
    # Keep this list stable: it defines the public model keys.
    return [
        "elman-srn",
        "jordan-srn",
        "bidirectional-rnn",
        "multi-dimensional-rnn",
        "gated-feedback-rnn",
        "hierarchical-multiscale-rnn",
        "clockwork-rnn",
        "dilated-rnn",
        "skip-rnn",
        "sliced-rnn",
        "lstm",
        "forget-gate-lstm",
        "peephole-lstm",
        "lstm-projection",
        "cifg-lstm",
        "chrono-lstm",
        "phased-lstm",
        "grid-lstm",
        "tree-lstm",
        "nested-lstm",
        "on-lstm",
        "lattice-rnn",
        "lattice-lstm",
        "wmc-lstm",
        "gru",
        "gru-variant-1",
        "gru-variant-2",
        "gru-variant-3",
        "mgu",
        "mgu1",
        "mgu2",
        "mgu3",
        "ligru",
        "sru",
        "qrnn",
        "indrnn",
        "minimalrnn",
        "cfn",
        "ran",
        "atr",
        "mut1",
        "mut2",
        "mut3",
        "fast-rnn",
        "fast-grnn",
        "fru",
        "rwa",
        "rhn",
        "scrn",
        "antisymmetric-rnn",
        "cornn",
        "unicornn",
        "lem",
        "tau-gru",
        "dg-rnn",
        "star",
        "strongly-typed-rnn",
        "multiplicative-lstm",
        "brc",
        "nbrc",
        "residual-rnn",
        "unitary-rnn",
        "orthogonal-rnn",
        "eunn",
        "goru",
        "ode-rnn",
        "neural-cde",
        "echo-state-network",
        "deep-esn",
        "liquid-state-machine",
        "conceptor-rnn",
        "deep-ar",
        "mqrnn",
        "deepstate",
        "lstnet",
        "esrnn",
        "neural-turing-machine",
        "differentiable-neural-computer",
        "memory-networks",
        "end-to-end-memory-networks",
        "dynamic-memory-networks",
        "pointer-network",
        "pointer-sentinel-mixture",
        "copynet",
        "rnn-transducer",
        "seq2seq",
        "rnn-encoder-decoder",
        "bahdanau-attention",
        "luong-attention",
        "neural-stack",
        "neural-queue",
        "neural-ram",
        "recurrent-attention-model",
        "convlstm",
        "convgru",
        "trajgru",
        "predrnn",
        "predrnn-plus-plus",
        "dcrnn",
        "structural-rnn",
    ]


def _rnnpaper_expected_keys() -> list[str]:
    return [f"torch-rnnpaper-{paper_id}-direct" for paper_id in _rnnpaper_ids()]


def test_rnnpaper_100_models_are_registered():
    keys = set(list_models())
    expected = _rnnpaper_expected_keys()
    missing = [k for k in expected if k not in keys]
    assert not missing, f"Missing {len(missing)} rnnpaper models, e.g. {missing[:5]}"


def test_rnnpaper_100_models_are_marked_optional_torch():
    for key in _rnnpaper_expected_keys():
        spec = get_model_spec(key)
        assert "torch" in spec.requires


@pytest.mark.parametrize(
    "key",
    [
        "torch-rnnpaper-elman-srn-direct",
        "torch-rnnpaper-jordan-srn-direct",
        "torch-rnnpaper-peephole-lstm-direct",
        "torch-rnnpaper-sru-direct",
        "torch-rnnpaper-echo-state-network-direct",
    ],
)
def test_rnnpaper_models_smoke_when_torch_installed(key: str):
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
