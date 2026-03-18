import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import make_global_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("key", "overrides"),
    (
        ("torch-lstnet-ema-global", {"warmup_epochs": 1}),
        ("torch-lstnet-swa-global", {"swa_start_epoch": 0}),
        ("torch-lstnet-sam-global", {"warmup_epochs": 1}),
        ("torch-lstnet-regularized-global", {"warmup_epochs": 1}),
        ("torch-lstnet-longhorizon-global", {"warmup_epochs": 1}),
        ("torch-lstnet-lookahead-global", {"warmup_epochs": 1}),
    ),
)
def test_wave57_global_lstnet_strategy_presets_smoke(
    key: str, overrides: dict[str, int]
) -> None:
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
        context_length=32,
        cnn_channels=8,
        kernel_size=3,
        rnn_hidden=16,
        skip=8,
        highway_window=16,
        dropout=0.0,
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

    assert set(pred.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred["yhat"].to_numpy(dtype=float)))
