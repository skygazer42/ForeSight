import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

RECURRENT_STRATEGY_PRESET_KEYS = (
    "torch-rnnpaper-lstm-ema-direct",
    "torch-rnnpaper-gru-swa-direct",
    "torch-rnnpaper-qrnn-lookahead-direct",
    "torch-rnnzoo-lstm-sam-direct",
    "torch-rnnzoo-gru-regularized-direct",
    "torch-rnnzoo-qrnn-longhorizon-direct",
)


def test_wave51_recurrent_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in RECURRENT_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
