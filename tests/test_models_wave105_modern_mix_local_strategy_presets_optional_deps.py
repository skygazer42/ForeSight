import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

MODERN_MIX_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-tft-ema-direct",
    "torch-tsmixer-swa-direct",
    "torch-timesnet-sam-direct",
    "torch-patchtst-regularized-direct",
    "torch-retnet-lookahead-direct",
    "torch-timexer-longhorizon-direct",
)


def test_wave105_modern_mix_local_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in MODERN_MIX_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
