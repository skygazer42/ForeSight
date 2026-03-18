import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

MAINSTREAM_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-patchtst-swa-direct",
    "torch-retnet-sam-direct",
    "torch-timesnet-regularized-direct",
    "torch-seq2seq-attn-gru-ema-direct",
    "torch-seq2seq-attn-lstm-lookahead-direct",
    "torch-tft-regularized-direct",
)


def test_wave102_mainstream_local_strategy_presets_raise_importerror_when_torch_missing() -> (
    None
):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in MAINSTREAM_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
