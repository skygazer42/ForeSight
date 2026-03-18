import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

STATE_SPACE_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-s4-ema-direct",
    "torch-s4d-swa-direct",
    "torch-s5-sam-direct",
    "torch-mamba2-regularized-direct",
    "torch-timesmamba-lookahead-direct",
    "torch-pathformer-swa-direct",
)


def test_wave107_state_space_refinement_local_strategy_presets_raise_importerror_when_torch_missing() -> (
    None
):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in STATE_SPACE_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
