import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import make_forecaster, make_global_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("key", "overrides"),
    (
        ("torch-xformer-full-ema-direct", {}),
        ("torch-xformer-performer-swa-direct", {"swa_start_epoch": 0}),
        ("torch-xformer-linformer-sam-direct", {}),
        ("torch-xformer-nystrom-regularized-direct", {"warmup_epochs": 1}),
        ("torch-xformer-bigbird-longhorizon-direct", {}),
        ("torch-xformer-longformer-lookahead-direct", {}),
    ),
)
def test_wave52_xformer_local_strategy_presets_smoke(
    key: str, overrides: dict[str, int]
) -> None:
    y = np.sin(np.arange(96, dtype=float) / 4.0) + 0.03 * np.arange(96, dtype=float)
    forecaster = make_forecaster(
        key,
        lags=24,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        local_window=8,
        performer_features=32,
        linformer_k=16,
        nystrom_landmarks=8,
        bigbird_random_k=4,
        epochs=2,
        batch_size=16,
        patience=1,
        seed=0,
        device="cpu",
        **overrides,
    )
    yhat = forecaster(y, 5)

    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("key", "overrides"),
    (
        ("torch-xformer-full-ema-global", {"warmup_epochs": 1}),
        ("torch-xformer-performer-swa-global", {"swa_start_epoch": 0}),
        ("torch-xformer-linformer-sam-global", {"warmup_epochs": 1}),
        ("torch-xformer-nystrom-regularized-global", {"warmup_epochs": 1}),
        ("torch-xformer-bigbird-longhorizon-global", {"warmup_epochs": 1}),
        ("torch-xformer-longformer-lookahead-global", {"warmup_epochs": 1}),
    ),
)
def test_wave52_xformer_global_strategy_presets_smoke(
    key: str, overrides: dict[str, int]
) -> None:
    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=72, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.15)]:
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
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        local_window=8,
        performer_features=32,
        linformer_k=16,
        nystrom_landmarks=8,
        bigbird_random_k=4,
        sample_step=3,
        epochs=1,
        val_split=0.0,
        batch_size=16,
        patience=1,
        x_cols=("promo",),
        seed=0,
        device="cpu",
        **overrides,
    )
    pred = forecaster(long_df, ds[-5], 4)

    assert set(pred.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred["yhat"].to_numpy(dtype=float)))
