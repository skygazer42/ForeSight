import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_multivariate_forecaster

MULTIVARIATE_STRATEGY_PRESET_KEYS = (
    "torch-stid-ema-multivariate",
    "torch-stgcn-swa-multivariate",
    "torch-graphwavenet-sam-multivariate",
    "torch-astgcn-regularized-multivariate",
    "torch-agcrn-longhorizon-multivariate",
    "torch-stemgnn-lookahead-multivariate",
)


def test_wave50_multivariate_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    train = np.ones((32, 4), dtype=float)
    for key in MULTIVARIATE_STRATEGY_PRESET_KEYS:
        forecaster = make_multivariate_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(train, 3)
