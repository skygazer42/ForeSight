import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster, make_global_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_crossformer_local_smoke():
    y = np.sin(np.arange(160, dtype=float) / 6.0) + 0.03 * np.arange(160, dtype=float)
    f = make_forecaster(
        "torch-crossformer-direct",
        lags=64,
        segment_len=8,
        stride=8,
        num_scales=2,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_pyraformer_local_smoke():
    y = np.sin(np.arange(160, dtype=float) / 7.0) + 0.02 * np.arange(160, dtype=float)
    f = make_forecaster(
        "torch-pyraformer-direct",
        lags=64,
        segment_len=8,
        stride=8,
        num_levels=3,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_crossformer_and_pyraformer_global_smoke():
    import pandas as pd

    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=90, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.3), ("s2", 0.7)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.01 * np.arange(ds.size)
        y = base + 0.05 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.1).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    cutoff = ds[-6]
    horizon = 5

    g1 = make_global_forecaster(
        "torch-crossformer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        segment_len=8,
        stride=8,
        num_scales=2,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1 = g1(long_df, cutoff, horizon)
    assert set(pred1.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1["yhat"].to_numpy(dtype=float)))

    g2 = make_global_forecaster(
        "torch-pyraformer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        segment_len=8,
        stride=8,
        num_levels=3,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred2 = g2(long_df, cutoff, horizon)
    assert set(pred2.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred2["yhat"].to_numpy(dtype=float)))
