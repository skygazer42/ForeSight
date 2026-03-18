import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import make_global_forecaster

GLOBAL_TRANSFORMER_ENCDEC_STRATEGY_PRESET_KEYS = (
    "torch-transformer-encdec-ema-global",
    "torch-transformer-encdec-swa-global",
    "torch-transformer-encdec-sam-global",
    "torch-transformer-encdec-regularized-global",
    "torch-transformer-encdec-longhorizon-global",
    "torch-transformer-encdec-lookahead-global",
)


def test_wave56_global_transformer_encdec_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    ds = pd.date_range("2020-01-01", periods=16, freq="D")
    long_df = pd.DataFrame(
        {
            "unique_id": ["s0"] * 16,
            "ds": ds,
            "y": np.ones(16, dtype=float),
            "promo": np.zeros(16, dtype=float),
        }
    )

    for key in GLOBAL_TRANSFORMER_ENCDEC_STRATEGY_PRESET_KEYS:
        forecaster = make_global_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(long_df, ds[-4], 3)
