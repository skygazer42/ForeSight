import importlib.util

import numpy as np
import pytest
from foresight.models.registry import make_multivariate_forecaster

GRAPH_SPECTRAL_KEYS = (
    "torch-stemgnn-multivariate",
    "torch-fouriergnn-multivariate",
)


def test_wave1_graph_spectral_models_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    train = np.ones((32, 4), dtype=float)
    for key in GRAPH_SPECTRAL_KEYS:
        forecaster = make_multivariate_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(train, 3)
