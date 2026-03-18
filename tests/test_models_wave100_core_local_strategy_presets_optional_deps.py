import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

CORE_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-lstm-ema-direct",
    "torch-gru-swa-direct",
    "torch-attn-gru-sam-direct",
    "torch-tcn-regularized-direct",
    "torch-cnn-lookahead-direct",
    "torch-mlp-longhorizon-direct",
)


def test_wave100_core_local_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in CORE_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
