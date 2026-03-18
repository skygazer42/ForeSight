import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

TRANSFORMER_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-informer-ema-direct",
    "torch-autoformer-swa-direct",
    "torch-fedformer-sam-direct",
    "torch-crossformer-regularized-direct",
    "torch-timexer-lookahead-direct",
    "torch-itransformer-longhorizon-direct",
)


def test_wave104_transformer_local_strategy_presets_raise_importerror_when_torch_missing() -> (
    None
):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in TRANSFORMER_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
