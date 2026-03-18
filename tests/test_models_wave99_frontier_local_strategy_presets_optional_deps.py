import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

FRONTIER_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-cfc-ema-direct",
    "torch-xlstm-swa-direct",
    "torch-griffin-sam-direct",
    "torch-hawk-regularized-direct",
    "torch-perceiver-lookahead-direct",
    "torch-moderntcn-longhorizon-direct",
)


def test_wave99_frontier_local_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in FRONTIER_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
