from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

from foresight.models.torch_ct_rnn import (
    torch_cfc_direct_forecast,
    torch_griffin_direct_forecast,
    torch_hawk_direct_forecast,
    torch_lmu_direct_forecast,
    torch_ltc_direct_forecast,
    torch_xlstm_direct_forecast,
)
from foresight.models.torch_probabilistic import torch_probabilistic_direct_forecast
from foresight.models.torch_rnn_paper_zoo import torch_rnnpaper_direct_forecast
from foresight.models.torch_seq2seq import (
    torch_lstnet_direct_forecast,
    torch_seq2seq_direct_forecast,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)


def _series(length: int = 48) -> np.ndarray:
    idx = np.arange(length, dtype=float)
    return np.sin(idx / 4.0) + 0.02 * idx


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (
            torch_lmu_direct_forecast,
            {"d_model": 8, "memory_dim": 4},
        ),
        (
            torch_ltc_direct_forecast,
            {"hidden_size": 6},
        ),
        (
            torch_cfc_direct_forecast,
            {"hidden_size": 6, "backbone_hidden": 8},
        ),
        (
            torch_xlstm_direct_forecast,
            {"hidden_size": 6, "proj_factor": 2},
        ),
        (
            torch_griffin_direct_forecast,
            {"hidden_size": 6, "conv_kernel": 3},
        ),
        (
            torch_hawk_direct_forecast,
            {"hidden_size": 6, "expansion_factor": 2},
        ),
    ],
)
def test_ct_rnn_variants_cover_s117_renamed_training_paths(
    fn: Any, kwargs: dict[str, Any]
) -> None:
    out = fn(
        _series(),
        2,
        lags=8,
        num_layers=1,
        dropout=0.0,
        epochs=1,
        batch_size=8,
        patience=1,
        seed=0,
        device="cpu",
        **kwargs,
    )

    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_probabilistic_direct_forecast_covers_renamed_sequence_tensor_path() -> None:
    out = torch_probabilistic_direct_forecast(
        _series(),
        2,
        variant="timegrad",
        lags=8,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        epochs=1,
        batch_size=8,
        patience=1,
        seed=0,
        device="cpu",
    )

    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_torch_seq2seq_direct_covers_validation_split_training_path() -> None:
    out = torch_seq2seq_direct_forecast(
        _series(56),
        2,
        lags=8,
        cell="lstm",
        attention="bahdanau",
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        teacher_forcing=0.5,
        teacher_forcing_final=0.0,
        epochs=1,
        batch_size=8,
        patience=1,
        val_split=0.2,
        seed=0,
        device="cpu",
    )

    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_torch_lstnet_direct_covers_renamed_sequence_tensor_path() -> None:
    out = torch_lstnet_direct_forecast(
        _series(60),
        2,
        lags=12,
        cnn_channels=4,
        kernel_size=3,
        rnn_hidden=6,
        skip=2,
        highway_window=6,
        dropout=0.0,
        epochs=1,
        batch_size=8,
        patience=1,
        seed=0,
        device="cpu",
    )

    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize(
    ("paper", "kwargs"),
    [
        ("deep-ar", {}),
        ("mqrnn", {}),
        ("lstnet", {"kernel_size": 3}),
        ("multi-dimensional-rnn", {}),
        ("seq2seq", {"val_split": 0.2}),
    ],
)
def test_rnnpaper_variants_cover_renamed_sequence_and_validation_paths(
    paper: str, kwargs: dict[str, Any]
) -> None:
    out = torch_rnnpaper_direct_forecast(
        _series(56),
        2,
        paper=paper,
        lags=8,
        hidden_size=6,
        attn_hidden=4,
        epochs=1,
        batch_size=8,
        patience=1,
        seed=0,
        device="cpu",
        **kwargs,
    )

    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
