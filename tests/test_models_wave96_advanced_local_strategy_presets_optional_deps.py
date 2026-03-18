import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

ADVANCED_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-retnet-ema-direct",
    "torch-crossformer-swa-direct",
    "torch-pyraformer-sam-direct",
    "torch-lightts-regularized-direct",
    "torch-samformer-lookahead-direct",
    "torch-timesmamba-longhorizon-direct",
)


def test_wave96_advanced_local_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in ADVANCED_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
