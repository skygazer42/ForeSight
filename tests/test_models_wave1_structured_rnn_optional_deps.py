import importlib.util

import pytest
from foresight.models.registry import make_forecaster

STRUCTURED_RNN_KEYS = (
    "torch-multidim-rnn-direct",
    "torch-grid-lstm-direct",
    "torch-structural-rnn-direct",
)


def test_wave1_structured_rnn_models_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    for key in STRUCTURED_RNN_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster([1.0, 2.0, 3.0, 4.0], 2)
