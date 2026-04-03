import importlib.util
import re

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import make_forecaster, make_global_forecaster

WAVE1A_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-informer-direct",
        {
            "lags": 48,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-autoformer-direct",
        {
            "lags": 48,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "ma_window": 7,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-nonstationary-transformer-direct",
        {
            "lags": 48,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-fedformer-direct",
        {
            "lags": 48,
            "d_model": 32,
            "num_layers": 1,
            "ffn_dim": 64,
            "modes": 8,
            "ma_window": 7,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-itransformer-direct",
        {
            "lags": 48,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-timesnet-direct",
        {
            "lags": 48,
            "d_model": 32,
            "num_layers": 1,
            "top_k": 3,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-tft-direct",
        {
            "lags": 48,
            "d_model": 32,
            "nhead": 4,
            "lstm_layers": 1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-timemixer-direct",
        {
            "lags": 48,
            "d_model": 16,
            "num_blocks": 2,
            "token_mixing_hidden": 32,
            "channel_mixing_hidden": 32,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-sparsetsf-direct",
        {
            "lags": 48,
            "period_len": 12,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

LIGHTWEIGHT_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-lightts-direct",
        {
            "lags": 48,
            "chunk_len": 12,
            "d_model": 32,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-frets-direct",
        {
            "lags": 48,
            "d_model": 32,
            "num_layers": 2,
            "top_k_freqs": 8,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

DECOMP_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-film-direct",
        {
            "lags": 48,
            "d_model": 32,
            "num_layers": 2,
            "ma_window": 7,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-micn-direct",
        {
            "lags": 48,
            "d_model": 32,
            "num_layers": 2,
            "kernel_sizes": (3, 5),
            "ma_window": 7,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

WAVE1B_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-koopa-direct",
        {
            "lags": 48,
            "d_model": 32,
            "latent_dim": 16,
            "num_blocks": 2,
            "ma_window": 7,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-samformer-direct",
        {
            "lags": 48,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

RETENTION_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-retnet-direct",
        {
            "lags": 32,
            "d_model": 24,
            "nhead": 4,
            "num_layers": 1,
            "ffn_dim": 48,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-retnet-recursive",
        {
            "lags": 32,
            "d_model": 24,
            "nhead": 4,
            "num_layers": 1,
            "ffn_dim": 48,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

TIMEXER_LOCAL_SMOKE_CASE = (
    "torch-timexer-direct",
    {
        "lags": 32,
        "x_cols": ("promo",),
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dropout": 0.1,
        "epochs": 2,
        "batch_size": 16,
    },
)

STATE_SPACE_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-lmu-direct",
        {
            "lags": 48,
            "d_model": 32,
            "memory_dim": 16,
            "num_layers": 1,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-s4d-direct",
        {
            "lags": 48,
            "d_model": 32,
            "num_layers": 2,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

CT_RNN_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-ltc-direct",
        {
            "lags": 48,
            "hidden_size": 24,
            "num_layers": 1,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-cfc-direct",
        {
            "lags": 48,
            "hidden_size": 24,
            "num_layers": 1,
            "backbone_hidden": 32,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

REVIVAL_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-xlstm-direct",
        {
            "lags": 48,
            "hidden_size": 24,
            "num_layers": 1,
            "proj_factor": 2,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-mamba2-direct",
        {
            "lags": 48,
            "d_model": 32,
            "num_layers": 2,
            "conv_kernel": 3,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

SSM_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-s4-direct",
        {
            "lags": 48,
            "d_model": 32,
            "num_layers": 2,
            "state_dim": 16,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-s5-direct",
        {
            "lags": 48,
            "d_model": 32,
            "num_layers": 2,
            "state_dim": 16,
            "heads": 2,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

RECURRENT_REVIVAL_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-griffin-direct",
        {
            "lags": 48,
            "hidden_size": 24,
            "num_layers": 1,
            "conv_kernel": 3,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-hawk-direct",
        {
            "lags": 48,
            "hidden_size": 24,
            "num_layers": 1,
            "expansion_factor": 2,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

SHARED_TRAINING_FORWARDING_TORCH_LOCAL_SMOKE_CASES = (
    STATE_SPACE_TORCH_LOCAL_SMOKE_CASES
    + CT_RNN_TORCH_LOCAL_SMOKE_CASES
    + REVIVAL_TORCH_LOCAL_SMOKE_CASES
    + SSM_TORCH_LOCAL_SMOKE_CASES
    + RECURRENT_REVIVAL_TORCH_LOCAL_SMOKE_CASES
)

LATENT_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-perceiver-direct",
        {
            "lags": 64,
            "d_model": 32,
            "latent_len": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-perceiver-deep-direct",
        {
            "lags": 64,
            "d_model": 32,
            "latent_len": 16,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

SEGMENTED_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-segrnn-direct",
        {
            "lags": 72,
            "segment_len": 12,
            "d_model": 32,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-segrnn-deep-direct",
        {
            "lags": 72,
            "segment_len": 12,
            "d_model": 32,
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

MODERN_CONV_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-moderntcn-direct",
        {
            "lags": 96,
            "patch_len": 8,
            "d_model": 32,
            "num_blocks": 2,
            "expansion_factor": 2.0,
            "kernel_size": 9,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-moderntcn-deep-direct",
        {
            "lags": 96,
            "patch_len": 8,
            "d_model": 32,
            "num_blocks": 4,
            "expansion_factor": 2.0,
            "kernel_size": 9,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

BASIS_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-basisformer-direct",
        {
            "lags": 96,
            "patch_len": 8,
            "d_model": 32,
            "num_bases": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-basisformer-deep-direct",
        {
            "lags": 96,
            "patch_len": 8,
            "d_model": 32,
            "num_bases": 16,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

GRID_RECURRENT_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-witran-direct",
        {
            "lags": 96,
            "grid_cols": 12,
            "d_model": 32,
            "hidden_size": 32,
            "nhead": 4,
            "num_layers": 1,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-witran-deep-direct",
        {
            "lags": 96,
            "grid_cols": 12,
            "d_model": 32,
            "hidden_size": 32,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

LAG_GRAPH_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-crossgnn-direct",
        {
            "lags": 96,
            "d_model": 32,
            "num_blocks": 2,
            "top_k": 8,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-crossgnn-deep-direct",
        {
            "lags": 96,
            "d_model": 32,
            "num_blocks": 4,
            "top_k": 8,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

MULTISCALE_ROUTING_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-pathformer-direct",
        {
            "lags": 96,
            "d_model": 32,
            "expert_patch_lens": (4, 8, 16),
            "num_blocks": 2,
            "top_k": 2,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-pathformer-deep-direct",
        {
            "lags": 96,
            "d_model": 32,
            "expert_patch_lens": (4, 8, 16),
            "num_blocks": 4,
            "top_k": 2,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]

PATCH_SSM_TORCH_LOCAL_SMOKE_CASES = [
    (
        "torch-timesmamba-direct",
        {
            "lags": 96,
            "patch_len": 8,
            "d_model": 32,
            "state_size": 32,
            "num_blocks": 2,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
    (
        "torch-timesmamba-deep-direct",
        {
            "lags": 96,
            "patch_len": 8,
            "d_model": 32,
            "state_size": 32,
            "num_blocks": 4,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 16,
        },
    ),
]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_xformer_local_smoke():
    y = np.sin(np.arange(140, dtype=float) / 4.0) + 0.05 * np.arange(140, dtype=float)
    f = make_forecaster(
        "torch-xformer-performer-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))

    f2 = make_forecaster(
        "torch-xformer-probsparse-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        probsparse_top_u=16,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat2 = f2(y, 5)
    assert yhat2.shape == (5,)
    assert np.all(np.isfinite(yhat2))

    f3 = make_forecaster(
        "torch-xformer-autocorr-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        autocorr_top_k=4,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat3 = f3(y, 5)
    assert yhat3.shape == (5,)
    assert np.all(np.isfinite(yhat3))

    f4 = make_forecaster(
        "torch-xformer-reformer-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        reformer_bucket_size=8,
        reformer_n_hashes=1,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat4 = f4(y, 5)
    assert yhat4.shape == (5,)
    assert np.all(np.isfinite(yhat4))

    f5 = make_forecaster(
        "torch-xformer-logsparse-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        local_window=8,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat5 = f5(y, 5)
    assert yhat5.shape == (5,)
    assert np.all(np.isfinite(yhat5))

    f6 = make_forecaster(
        "torch-xformer-longformer-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        local_window=8,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat6 = f6(y, 5)
    assert yhat6.shape == (5,)
    assert np.all(np.isfinite(yhat6))

    f7 = make_forecaster(
        "torch-xformer-bigbird-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        local_window=8,
        bigbird_random_k=8,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat7 = f7(y, 5)
    assert yhat7.shape == (5,)
    assert np.all(np.isfinite(yhat7))


@pytest.mark.parametrize(
    ("param_name", "param_value", "message"),
    (
        ("horizon_loss_decay", 0.0, "horizon_loss_decay must be > 0"),
        ("sam_rho", -0.1, "sam_rho must be >= 0"),
    ),
)
def test_torch_xformer_local_forwards_shared_training_validation(
    param_name: str, param_value: float, message: str
):
    y = np.sin(np.arange(140, dtype=float) / 4.0) + 0.05 * np.arange(140, dtype=float)
    f = make_forecaster(
        "torch-xformer-performer-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
        **{param_name: param_value},
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        f(y, 5)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), WAVE1A_TORCH_LOCAL_SMOKE_CASES)
def test_wave1a_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(140, dtype=float) / 5.0) + 0.05 * np.arange(140, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), LIGHTWEIGHT_TORCH_LOCAL_SMOKE_CASES)
def test_lightweight_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(144, dtype=float) / 6.0) + 0.03 * np.arange(144, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), DECOMP_TORCH_LOCAL_SMOKE_CASES)
def test_decomposition_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(144, dtype=float) / 7.0) + 0.04 * np.arange(144, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), WAVE1B_TORCH_LOCAL_SMOKE_CASES)
def test_wave1b_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(150, dtype=float) / 8.0) + 0.02 * np.arange(150, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), RETENTION_TORCH_LOCAL_SMOKE_CASES)
def test_retnet_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(150, dtype=float) / 8.5) + 0.02 * np.arange(150, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_timexer_local_smoke():
    y = np.sin(np.arange(96, dtype=float) / 6.0) + 0.04 * np.arange(96, dtype=float)
    train_exog = (np.arange(96, dtype=float) % 5 == 0).astype(float).reshape(-1, 1)
    future_exog = np.array([[1.0], [0.0], [1.0], [0.0], [1.0]], dtype=float)
    key, params = TIMEXER_LOCAL_SMOKE_CASE
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5, train_exog=train_exog, future_exog=future_exog)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), STATE_SPACE_TORCH_LOCAL_SMOKE_CASES)
def test_state_space_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(128, dtype=float) / 5.0) + 0.03 * np.arange(128, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), CT_RNN_TORCH_LOCAL_SMOKE_CASES)
def test_continuous_time_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(128, dtype=float) / 6.0) + 0.02 * np.arange(128, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), REVIVAL_TORCH_LOCAL_SMOKE_CASES)
def test_revival_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(128, dtype=float) / 7.0) + 0.025 * np.arange(128, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), SSM_TORCH_LOCAL_SMOKE_CASES)
def test_ssm_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(128, dtype=float) / 8.0) + 0.02 * np.arange(128, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), RECURRENT_REVIVAL_TORCH_LOCAL_SMOKE_CASES)
def test_recurrent_revival_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(128, dtype=float) / 9.0) + 0.02 * np.arange(128, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), SHARED_TRAINING_FORWARDING_TORCH_LOCAL_SMOKE_CASES)
@pytest.mark.parametrize(
    ("param_name", "param_value", "message"),
    (
        ("horizon_loss_decay", 0.0, "horizon_loss_decay must be > 0"),
        ("sam_rho", -0.1, "sam_rho must be >= 0"),
    ),
)
def test_extended_torch_local_models_forward_shared_training_validation(
    key: str,
    params: dict[str, object],
    param_name: str,
    param_value: float,
    message: str,
):
    y = np.sin(np.arange(128, dtype=float) / 10.0) + 0.02 * np.arange(128, dtype=float)
    f = make_forecaster(
        key, **params, patience=2, device="cpu", seed=0, **{param_name: param_value}
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        f(y, 5)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), LATENT_TORCH_LOCAL_SMOKE_CASES)
def test_latent_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(160, dtype=float) / 9.5) + 0.02 * np.arange(160, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), SEGMENTED_TORCH_LOCAL_SMOKE_CASES)
def test_segmented_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(168, dtype=float) / 10.0) + 0.015 * np.arange(168, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), MODERN_CONV_TORCH_LOCAL_SMOKE_CASES)
def test_modern_conv_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(192, dtype=float) / 10.5) + 0.015 * np.arange(192, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), BASIS_TORCH_LOCAL_SMOKE_CASES)
def test_basis_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(192, dtype=float) / 11.0) + 0.02 * np.arange(192, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), GRID_RECURRENT_TORCH_LOCAL_SMOKE_CASES)
def test_grid_recurrent_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(192, dtype=float) / 11.5) + 0.018 * np.arange(192, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), LAG_GRAPH_TORCH_LOCAL_SMOKE_CASES)
def test_lag_graph_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(192, dtype=float) / 12.0) + 0.016 * np.arange(192, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), MULTISCALE_ROUTING_TORCH_LOCAL_SMOKE_CASES)
def test_multiscale_routing_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(192, dtype=float) / 12.5) + 0.015 * np.arange(192, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
@pytest.mark.parametrize(("key", "params"), PATCH_SSM_TORCH_LOCAL_SMOKE_CASES)
def test_patch_ssm_torch_local_smoke(key: str, params: dict[str, object]):
    y = np.sin(np.arange(192, dtype=float) / 13.0) + 0.014 * np.arange(192, dtype=float)
    f = make_forecaster(key, **params, patience=2, device="cpu", seed=0)
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_seq2seq_local_smoke():
    y = np.sin(np.arange(120, dtype=float) / 5.0) + 0.03 * np.arange(120, dtype=float)
    f = make_forecaster(
        "torch-seq2seq-attn-lstm-direct",
        lags=24,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        teacher_forcing=0.6,
        teacher_forcing_final=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_mamba_rwkv_local_smoke():
    y = np.sin(np.arange(140, dtype=float) / 6.0) + 0.04 * np.arange(140, dtype=float)

    f1 = make_forecaster(
        "torch-mamba-direct",
        lags=48,
        d_model=32,
        num_layers=1,
        conv_kernel=3,
        dropout=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat1 = f1(y, 5)
    assert yhat1.shape == (5,)
    assert np.all(np.isfinite(yhat1))

    f2 = make_forecaster(
        "torch-rwkv-direct",
        lags=48,
        d_model=32,
        num_layers=1,
        ffn_dim=64,
        dropout=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat2 = f2(y, 5)
    assert yhat2.shape == (5,)
    assert np.all(np.isfinite(yhat2))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_hyena_local_smoke():
    y = np.sin(np.arange(140, dtype=float) / 7.0) + 0.04 * np.arange(140, dtype=float)
    f = make_forecaster(
        "torch-hyena-direct",
        lags=48,
        d_model=32,
        num_layers=1,
        ffn_dim=64,
        kernel_size=32,
        dropout=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_dilated_rnn_and_kan_local_smoke():
    y = np.sin(np.arange(160, dtype=float) / 9.0) + 0.03 * np.arange(160, dtype=float)

    f1 = make_forecaster(
        "torch-dilated-rnn-direct",
        lags=48,
        cell="gru",
        hidden_size=32,
        num_layers=2,
        dilation_base=2,
        dropout=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat1 = f1(y, 5)
    assert yhat1.shape == (5,)
    assert np.all(np.isfinite(yhat1))

    f2 = make_forecaster(
        "torch-kan-direct",
        lags=48,
        d_model=32,
        num_layers=1,
        grid_size=8,
        grid_range=2.0,
        dropout=0.0,
        linear_skip=True,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat2 = f2(y, 5)
    assert yhat2.shape == (5,)
    assert np.all(np.isfinite(yhat2))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_scinet_and_etsformer_local_smoke():
    y = np.sin(np.arange(160, dtype=float) / 8.0) + 0.03 * np.arange(160, dtype=float)

    f1 = make_forecaster(
        "torch-scinet-direct",
        lags=48,
        d_model=32,
        num_stages=2,
        conv_kernel=5,
        ffn_dim=64,
        dropout=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat1 = f1(y, 5)
    assert yhat1.shape == (5,)
    assert np.all(np.isfinite(yhat1))

    f2 = make_forecaster(
        "torch-etsformer-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        alpha_init=0.3,
        beta_init=0.1,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat2 = f2(y, 5)
    assert yhat2.shape == (5,)
    assert np.all(np.isfinite(yhat2))

    f3 = make_forecaster(
        "torch-esrnn-direct",
        lags=48,
        cell="gru",
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        alpha_init=0.3,
        beta_init=0.1,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat3 = f3(y, 5)
    assert yhat3.shape == (5,)
    assert np.all(np.isfinite(yhat3))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_xformer_and_rnn_global_smoke():
    import pandas as pd

    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=80, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.2), ("s2", 0.8)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.01 * np.arange(ds.size)
        y = base + 0.05 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.1).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    cutoff = ds[-6]
    horizon = 5

    g1 = make_global_forecaster(
        "torch-xformer-performer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1 = g1(long_df, cutoff, horizon)
    assert set(pred1.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1["yhat"].to_numpy(dtype=float)))

    g1b = make_global_forecaster(
        "torch-xformer-probsparse-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        probsparse_top_u=16,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1b = g1b(long_df, cutoff, horizon)
    assert set(pred1b.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1b["yhat"].to_numpy(dtype=float)))

    g1c = make_global_forecaster(
        "torch-xformer-autocorr-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        autocorr_top_k=4,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1c = g1c(long_df, cutoff, horizon)
    assert set(pred1c.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1c["yhat"].to_numpy(dtype=float)))

    g1d = make_global_forecaster(
        "torch-xformer-reformer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        reformer_bucket_size=8,
        reformer_n_hashes=1,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1d = g1d(long_df, cutoff, horizon)
    assert set(pred1d.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1d["yhat"].to_numpy(dtype=float)))

    g1e = make_global_forecaster(
        "torch-xformer-logsparse-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        local_window=8,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1e = g1e(long_df, cutoff, horizon)
    assert set(pred1e.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1e["yhat"].to_numpy(dtype=float)))

    g1f = make_global_forecaster(
        "torch-xformer-longformer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        local_window=8,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1f = g1f(long_df, cutoff, horizon)
    assert set(pred1f.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1f["yhat"].to_numpy(dtype=float)))

    g1g = make_global_forecaster(
        "torch-xformer-bigbird-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        local_window=8,
        bigbird_random_k=8,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1g = g1g(long_df, cutoff, horizon)
    assert set(pred1g.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1g["yhat"].to_numpy(dtype=float)))

    g2 = make_global_forecaster(
        "torch-rnn-gru-global",
        context_length=32,
        hidden_size=32,
        num_layers=1,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred2 = g2(long_df, cutoff, horizon)
    assert set(pred2.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred2["yhat"].to_numpy(dtype=float)))

    g3 = make_global_forecaster(
        "torch-patchtst-global",
        context_length=32,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred3 = g3(long_df, cutoff, horizon)
    assert set(pred3.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred3["yhat"].to_numpy(dtype=float)))

    g4 = make_global_forecaster(
        "torch-tsmixer-global",
        context_length=32,
        d_model=32,
        num_blocks=2,
        token_mixing_hidden=64,
        channel_mixing_hidden=64,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred4 = g4(long_df, cutoff, horizon)
    assert set(pred4.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred4["yhat"].to_numpy(dtype=float)))

    g5 = make_global_forecaster(
        "torch-itransformer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred5 = g5(long_df, cutoff, horizon)
    assert set(pred5.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred5["yhat"].to_numpy(dtype=float)))

    g6 = make_global_forecaster(
        "torch-timesnet-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        top_k=2,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred6 = g6(long_df, cutoff, horizon)
    assert set(pred6.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred6["yhat"].to_numpy(dtype=float)))

    g7 = make_global_forecaster(
        "torch-seq2seq-attn-lstm-global",
        context_length=32,
        hidden_size=32,
        num_layers=1,
        attention="bahdanau",
        teacher_forcing=0.6,
        teacher_forcing_final=0.0,
        quantiles="0.1,0.5,0.9",
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred7 = g7(long_df, cutoff, horizon)
    assert set(pred7.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred7["yhat"].to_numpy(dtype=float)))

    g8 = make_global_forecaster(
        "torch-tcn-global",
        context_length=32,
        channels=(32, 32),
        kernel_size=3,
        dilation_base=2,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred8 = g8(long_df, cutoff, horizon)
    assert set(pred8.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred8["yhat"].to_numpy(dtype=float)))

    g9 = make_global_forecaster(
        "torch-nbeats-global",
        context_length=32,
        num_blocks=2,
        num_layers=2,
        layer_width=64,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred9 = g9(long_df, cutoff, horizon)
    assert set(pred9.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred9["yhat"].to_numpy(dtype=float)))

    g10 = make_global_forecaster(
        "torch-nhits-global",
        context_length=32,
        pool_sizes=(1, 2),
        num_blocks=2,
        num_layers=2,
        layer_width=64,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred10 = g10(long_df, cutoff, horizon)
    assert set(pred10.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred10["yhat"].to_numpy(dtype=float)))

    g11 = make_global_forecaster(
        "torch-tide-global",
        context_length=32,
        d_model=32,
        hidden_size=64,
        dropout=0.0,
        quantiles="0.1,0.5,0.9",
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred11 = g11(long_df, cutoff, horizon)
    assert set(pred11.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred11["yhat"].to_numpy(dtype=float)))

    g12 = make_global_forecaster(
        "torch-wavenet-global",
        context_length=32,
        channels=16,
        num_layers=4,
        kernel_size=2,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred12 = g12(long_df, cutoff, horizon)
    assert set(pred12.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred12["yhat"].to_numpy(dtype=float)))

    g13 = make_global_forecaster(
        "torch-resnet1d-global",
        context_length=32,
        channels=16,
        num_blocks=2,
        kernel_size=3,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred13 = g13(long_df, cutoff, horizon)
    assert set(pred13.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred13["yhat"].to_numpy(dtype=float)))

    g14 = make_global_forecaster(
        "torch-inception-global",
        context_length=32,
        channels=16,
        num_blocks=2,
        kernel_sizes=(3, 5),
        bottleneck_channels=8,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred14 = g14(long_df, cutoff, horizon)
    assert set(pred14.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred14["yhat"].to_numpy(dtype=float)))

    g15 = make_global_forecaster(
        "torch-lstnet-global",
        context_length=32,
        cnn_channels=8,
        kernel_size=3,
        rnn_hidden=16,
        skip=8,
        highway_window=16,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred15 = g15(long_df, cutoff, horizon)
    assert set(pred15.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred15["yhat"].to_numpy(dtype=float)))

    g16 = make_global_forecaster(
        "torch-fnet-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred16 = g16(long_df, cutoff, horizon)
    assert set(pred16.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred16["yhat"].to_numpy(dtype=float)))

    g17 = make_global_forecaster(
        "torch-gmlp-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        ffn_dim=32,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred17 = g17(long_df, cutoff, horizon)
    assert set(pred17.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred17["yhat"].to_numpy(dtype=float)))

    g18 = make_global_forecaster(
        "torch-ssm-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred18 = g18(long_df, cutoff, horizon)
    assert set(pred18.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred18["yhat"].to_numpy(dtype=float)))

    g19 = make_global_forecaster(
        "torch-transformer-encdec-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        quantiles="0.1,0.5,0.9",
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred19 = g19(long_df, cutoff, horizon)
    assert set(pred19.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred19["yhat"].to_numpy(dtype=float)))

    g20 = make_global_forecaster(
        "torch-nlinear-global",
        context_length=32,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred20 = g20(long_df, cutoff, horizon)
    assert set(pred20.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred20["yhat"].to_numpy(dtype=float)))

    g21 = make_global_forecaster(
        "torch-dlinear-global",
        context_length=32,
        ma_window=7,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred21 = g21(long_df, cutoff, horizon)
    assert set(pred21.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred21["yhat"].to_numpy(dtype=float)))

    g22 = make_global_forecaster(
        "torch-deepar-global",
        context_length=32,
        hidden_size=32,
        num_layers=1,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred22 = g22(long_df, cutoff, horizon)
    assert set(pred22.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred22["yhat"].to_numpy(dtype=float)))

    g23 = make_global_forecaster(
        "torch-fedformer-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        ffn_dim=64,
        modes=8,
        ma_window=7,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred23 = g23(long_df, cutoff, horizon)
    assert set(pred23.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred23["yhat"].to_numpy(dtype=float)))

    g24 = make_global_forecaster(
        "torch-nonstationary-transformer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred24 = g24(long_df, cutoff, horizon)
    assert set(pred24.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred24["yhat"].to_numpy(dtype=float)))

    g25 = make_global_forecaster(
        "torch-mamba-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        conv_kernel=3,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred25 = g25(long_df, cutoff, horizon)
    assert set(pred25.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred25["yhat"].to_numpy(dtype=float)))

    g26 = make_global_forecaster(
        "torch-rwkv-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        ffn_dim=64,
        dropout=0.0,
        quantiles="0.1,0.5,0.9",
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred26 = g26(long_df, cutoff, horizon)
    assert set(pred26.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred26["yhat"].to_numpy(dtype=float)))

    g27 = make_global_forecaster(
        "torch-hyena-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        ffn_dim=64,
        kernel_size=32,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred27 = g27(long_df, cutoff, horizon)
    assert set(pred27.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred27["yhat"].to_numpy(dtype=float)))

    g28 = make_global_forecaster(
        "torch-dilated-rnn-global",
        context_length=32,
        cell="gru",
        d_model=32,
        num_layers=2,
        dilation_base=2,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred28 = g28(long_df, cutoff, horizon)
    assert set(pred28.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred28["yhat"].to_numpy(dtype=float)))

    g29 = make_global_forecaster(
        "torch-kan-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        grid_size=8,
        grid_range=2.0,
        dropout=0.0,
        linear_skip=True,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred29 = g29(long_df, cutoff, horizon)
    assert set(pred29.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred29["yhat"].to_numpy(dtype=float)))

    g30 = make_global_forecaster(
        "torch-scinet-global",
        context_length=32,
        d_model=32,
        num_stages=2,
        conv_kernel=5,
        ffn_dim=64,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred30 = g30(long_df, cutoff, horizon)
    assert set(pred30.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred30["yhat"].to_numpy(dtype=float)))

    g31 = make_global_forecaster(
        "torch-etsformer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        alpha_init=0.3,
        beta_init=0.1,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred31 = g31(long_df, cutoff, horizon)
    assert set(pred31.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred31["yhat"].to_numpy(dtype=float)))

    g32 = make_global_forecaster(
        "torch-esrnn-global",
        context_length=32,
        cell="gru",
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        alpha_init=0.3,
        beta_init=0.1,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred32 = g32(long_df, cutoff, horizon)
    assert set(pred32.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred32["yhat"].to_numpy(dtype=float)))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_timexer_global_smoke():
    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=72, freq="D")
    rows = []
    for uid, bias in [("s0", 0.0), ("s1", 0.5)]:
        promo = (rng.random(ds.size) < 0.2).astype(float)
        y = bias + np.sin(np.arange(ds.size, dtype=float) / 7.0) + 2.0 * promo
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    cutoff = ds[-6]
    g = make_global_forecaster(
        "torch-timexer-global",
        context_length=32,
        x_cols=("promo",),
        d_model=32,
        nhead=4,
        num_layers=1,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        device="cpu",
        seed=0,
    )
    pred = g(long_df, cutoff, 5)
    assert set(pred.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred["yhat"].to_numpy(dtype=float)))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_retnet_global_smoke():
    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=72, freq="D")
    rows = []
    for uid, bias in [("s0", 0.0), ("s1", 0.5)]:
        promo = (rng.random(ds.size) < 0.2).astype(float)
        y = bias + np.sin(np.arange(ds.size, dtype=float) / 7.0) + 1.5 * promo
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    cutoff = ds[-6]
    g = make_global_forecaster(
        "torch-retnet-global",
        context_length=32,
        d_model=24,
        nhead=4,
        num_layers=1,
        ffn_dim=48,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred = g(long_df, cutoff, 5)
    assert set(pred.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred["yhat"].to_numpy(dtype=float)))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_global_cv_preserves_quantile_columns():
    import pandas as pd

    from foresight.cv import cross_validation_predictions_long_df

    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=80, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.2), ("s2", 0.8)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.01 * np.arange(ds.size)
        y = base + 0.05 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.1).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    pred_df = cross_validation_predictions_long_df(
        model="torch-itransformer-global",
        long_df=long_df,
        horizon=3,
        step_size=3,
        min_train_size=40,
        model_params={
            "context_length": 32,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "sample_step": 4,
            "epochs": 1,
            "batch_size": 32,
            "patience": 2,
            "val_split": 0.0,
            "x_cols": ("promo",),
            "quantiles": "0.1,0.5,0.9",
            "device": "cpu",
            "seed": 0,
        },
        n_windows=1,
    )

    assert set(pred_df.columns) >= {"unique_id", "ds", "cutoff", "step", "y", "yhat", "yhat_p10"}
    assert np.all(np.isfinite(pred_df["yhat"].to_numpy(dtype=float)))
