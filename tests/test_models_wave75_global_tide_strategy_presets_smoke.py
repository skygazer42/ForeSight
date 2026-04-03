import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import make_global_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("key", "overrides"),
    (
        ("torch-tide-ema-global", {"warmup_epochs": 1}),
        ("torch-tide-swa-global", {"swa_start_epoch": 0}),
        ("torch-tide-sam-global", {"warmup_epochs": 1}),
        ("torch-tide-regularized-global", {"warmup_epochs": 1}),
        ("torch-tide-longhorizon-global", {"warmup_epochs": 1}),
        ("torch-tide-lookahead-global", {"warmup_epochs": 1}),
    ),
)
def test_wave75_global_tide_strategy_presets_smoke(key: str, overrides: dict[str, int]) -> None:
    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=72, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.1)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.02 * np.arange(ds.size)
        y = base + 0.03 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.2).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    forecaster = make_global_forecaster(
        key,
        context_length=24,
        d_model=32,
        hidden_size=64,
        dropout=0.0,
        quantiles="0.1,0.5,0.9",
        sample_step=3,
        epochs=1,
        val_split=0.0,
        batch_size=16,
        patience=1,
        x_cols=("promo",),
        device="cpu",
        seed=0,
        **overrides,
    )
    pred = forecaster(long_df, ds[-5], 4)

    assert set(pred.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred["yhat"].to_numpy(dtype=float)))
