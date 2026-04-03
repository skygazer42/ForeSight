import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

BASELINE_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-nhits-ema-direct",
    "torch-nbeats-sam-direct",
    "torch-tide-regularized-direct",
    "torch-dlinear-lookahead-direct",
    "torch-nlinear-longhorizon-direct",
    "torch-timemixer-ema-direct",
)


def test_wave103_baseline_local_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in BASELINE_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
