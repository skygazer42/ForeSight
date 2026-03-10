import importlib.util

import pytest
from foresight.models.registry import make_forecaster

PROBABILISTIC_KEYS = (
    "torch-timegrad-direct",
    "torch-tactis-direct",
)


def test_wave1_probabilistic_models_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    for key in PROBABILISTIC_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster([1.0, 2.0, 3.0, 4.0], 2)
