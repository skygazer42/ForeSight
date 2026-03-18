import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

EXTENDED_CORE_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-bigru-ema-direct",
    "torch-bilstm-swa-direct",
    "torch-linear-attn-sam-direct",
    "torch-koopa-regularized-direct",
    "torch-fits-lookahead-direct",
    "torch-film-longhorizon-direct",
)


def test_wave101_extended_core_local_strategy_presets_raise_importerror_when_torch_missing() -> (
    None
):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in EXTENDED_CORE_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
