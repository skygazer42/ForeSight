import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

STATE_SPACE_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-lmu-ema-direct",
    "torch-ltc-swa-direct",
    "torch-s4-sam-direct",
    "torch-s4d-regularized-direct",
    "torch-s5-lookahead-direct",
    "torch-mamba2-longhorizon-direct",
)


def test_wave98_state_space_local_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in STATE_SPACE_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
