import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster

SEQUENCE_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-patchtst-lookahead-direct",
    "torch-retnet-swa-direct",
    "torch-timesnet-lookahead-direct",
    "torch-seq2seq-gru-regularized-direct",
    "torch-seq2seq-gru-sam-direct",
    "torch-seq2seq-attn-lstm-ema-direct",
)


def test_wave110_sequence_refinement_local_strategy_presets_raise_importerror_when_torch_missing() -> (
    None
):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in SEQUENCE_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)
