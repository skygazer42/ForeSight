import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

SEQ2SEQ_STRATEGY_PRESET_KEYS = (
    "torch-seq2seq-lstm-ema-direct",
    "torch-seq2seq-gru-swa-direct",
    "torch-seq2seq-attn-lstm-sam-direct",
    "torch-seq2seq-attn-gru-regularized-direct",
    "torch-seq2seq-lstm-longhorizon-direct",
    "torch-seq2seq-attn-gru-lookahead-direct",
)


def test_wave53_seq2seq_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in SEQ2SEQ_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
