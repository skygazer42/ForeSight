import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

TRANSFORMER_COMPLETION_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-informer-longhorizon-direct",
    "torch-autoformer-ema-direct",
    "torch-fedformer-swa-direct",
    "torch-crossformer-lookahead-direct",
    "torch-itransformer-regularized-direct",
    "torch-timexer-swa-direct",
)


def test_wave108_transformer_completion_local_strategy_presets_raise_importerror_when_torch_missing() -> (
    None
):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in TRANSFORMER_COMPLETION_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
