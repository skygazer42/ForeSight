import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("key", "overrides"),
    (
        ("torch-seq2seq-lstm-ema-direct", {}),
        ("torch-seq2seq-gru-swa-direct", {"swa_start_epoch": 0}),
        ("torch-seq2seq-attn-lstm-sam-direct", {}),
        ("torch-seq2seq-attn-gru-regularized-direct", {"warmup_epochs": 1}),
        ("torch-seq2seq-lstm-longhorizon-direct", {}),
        ("torch-seq2seq-attn-gru-lookahead-direct", {}),
    ),
)
def test_wave53_seq2seq_strategy_presets_smoke(
    key: str, overrides: dict[str, int]
) -> None:
    y = np.sin(np.arange(96, dtype=float) / 5.0) + 0.03 * np.arange(96, dtype=float)
    forecaster = make_forecaster(
        key,
        lags=24,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        teacher_forcing=0.6,
        teacher_forcing_final=0.0,
        epochs=2,
        batch_size=16,
        patience=1,
        device="cpu",
        seed=0,
        **overrides,
    )
    yhat = forecaster(y, 5)

    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))
