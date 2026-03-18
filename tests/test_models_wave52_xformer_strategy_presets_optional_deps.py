import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import make_forecaster, make_global_forecaster

LOCAL_XFORMER_STRATEGY_PRESET_KEYS = (
    "torch-xformer-full-ema-direct",
    "torch-xformer-performer-swa-direct",
    "torch-xformer-linformer-sam-direct",
    "torch-xformer-nystrom-regularized-direct",
    "torch-xformer-bigbird-longhorizon-direct",
    "torch-xformer-longformer-lookahead-direct",
)

GLOBAL_XFORMER_STRATEGY_PRESET_KEYS = (
    "torch-xformer-full-ema-global",
    "torch-xformer-performer-swa-global",
    "torch-xformer-linformer-sam-global",
    "torch-xformer-nystrom-regularized-global",
    "torch-xformer-bigbird-longhorizon-global",
    "torch-xformer-longformer-lookahead-global",
)


def test_wave52_xformer_local_strategy_presets_raise_importerror_when_torch_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    y = np.ones(32, dtype=float)
    for key in LOCAL_XFORMER_STRATEGY_PRESET_KEYS:
        forecaster = make_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(y, 3)


def test_wave52_xformer_global_strategy_presets_raise_importerror_when_torch_missing() -> None:
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

    for key in GLOBAL_XFORMER_STRATEGY_PRESET_KEYS:
        forecaster = make_global_forecaster(key)
        with pytest.raises(ImportError):
            forecaster(long_df, ds[-4], 3)
