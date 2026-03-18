import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import make_global_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("key", "overrides"),
    (
        ("torch-tft-ema-global", {"warmup_epochs": 1}),
        ("torch-tft-swa-global", {"swa_start_epoch": 0}),
        ("torch-tft-sam-global", {"warmup_epochs": 1}),
        ("torch-tft-regularized-global", {"warmup_epochs": 1}),
        ("torch-tft-longhorizon-global", {"warmup_epochs": 1}),
        ("torch-tft-lookahead-global", {"warmup_epochs": 1}),
    ),
)
def test_wave91_global_tft_strategy_presets_smoke(
    key: str, overrides: dict[str, int]
) -> None:
    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=72, freq="D")
    rows = []
    for uid, bias in [("s0", 0.0), ("s1", 0.5)]:
        promo = (rng.random(ds.size) < 0.2).astype(float)
        trend = 0.03 * np.arange(ds.size, dtype=float)
        season = np.sin(np.arange(ds.size, dtype=float) / 7.0)
        y = bias + trend + season + 1.5 * promo
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    forecaster = make_global_forecaster(
        key,
        context_length=24,
        x_cols=("promo",),
        d_model=32,
        nhead=4,
        lstm_layers=1,
        sample_step=3,
        epochs=1,
        val_split=0.0,
        batch_size=16,
        patience=1,
        device="cpu",
        seed=0,
        **overrides,
    )
    pred = forecaster(long_df, ds[-5], 4)

    assert set(pred.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred["yhat"].to_numpy(dtype=float)))
