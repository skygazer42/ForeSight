import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

EMERGING_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-pathformer-ema-direct",
    "torch-timemixer-swa-direct",
    "torch-tinytimemixer-sam-direct",
    "torch-basisformer-regularized-direct",
    "torch-witran-lookahead-direct",
    "torch-crossgnn-longhorizon-direct",
)


def test_wave97_emerging_local_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in EMERGING_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
