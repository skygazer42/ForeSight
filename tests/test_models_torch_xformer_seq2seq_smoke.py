import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster, make_global_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_xformer_local_smoke():
    y = np.sin(np.arange(140, dtype=float) / 4.0) + 0.05 * np.arange(140, dtype=float)
    f = make_forecaster(
        "torch-xformer-performer-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
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
def test_torch_seq2seq_local_smoke():
    y = np.sin(np.arange(120, dtype=float) / 5.0) + 0.03 * np.arange(120, dtype=float)
    f = make_forecaster(
        "torch-seq2seq-attn-lstm-direct",
        lags=24,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        teacher_forcing=0.6,
        teacher_forcing_final=0.0,
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
def test_torch_xformer_and_rnn_global_smoke():
    import pandas as pd

    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=80, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.2), ("s2", 0.8)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.01 * np.arange(ds.size)
        y = base + 0.05 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.1).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    cutoff = ds[-6]
    horizon = 5

    g1 = make_global_forecaster(
        "torch-xformer-performer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        epochs=1,
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
        "torch-rnn-gru-global",
        context_length=32,
        hidden_size=32,
        num_layers=1,
        epochs=1,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred2 = g2(long_df, cutoff, horizon)
    assert set(pred2.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred2["yhat"].to_numpy(dtype=float)))
