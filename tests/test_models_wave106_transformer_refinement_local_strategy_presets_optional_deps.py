import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

TRANSFORMER_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-informer-sam-direct",
    "torch-autoformer-regularized-direct",
    "torch-fedformer-lookahead-direct",
    "torch-crossformer-ema-direct",
    "torch-itransformer-swa-direct",
    "torch-timexer-ema-direct",
)


def test_wave106_transformer_refinement_local_strategy_presets_raise_importerror_when_torch_missing() -> (
    None
):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in TRANSFORMER_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
