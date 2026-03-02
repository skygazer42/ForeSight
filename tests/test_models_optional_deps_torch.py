import importlib.util

import numpy as np
import pytest

from foresight.models.registry import (
    get_model_spec,
    list_models,
    make_forecaster,
    make_global_forecaster,
)


def _torch_model_keys() -> list[str]:
    return [k for k in list_models() if "torch" in get_model_spec(k).requires]


def _torch_local_model_keys() -> list[str]:
    return [k for k in _torch_model_keys() if get_model_spec(k).interface == "local"]


def _torch_global_model_keys() -> list[str]:
    return [k for k in _torch_model_keys() if get_model_spec(k).interface == "global"]


def test_torch_models_are_registered_as_optional():
    for key in _torch_model_keys():
        spec = get_model_spec(key)
        assert "torch" in spec.requires


def test_torch_models_raise_importerror_when_torch_missing():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    for key in _torch_local_model_keys():
        f = make_forecaster(key)
        with pytest.raises(ImportError):
            f([1.0, 2.0, 3.0], 2)

    for key in _torch_global_model_keys():
        import pandas as pd

        g = make_global_forecaster(key)
        with pytest.raises(ImportError):
            g(
                pd.DataFrame(
                    {
                        "unique_id": ["s0", "s0", "s0"],
                        "ds": pd.date_range("2020-01-01", periods=3, freq="D"),
                        "y": [1.0, 2.0, 3.0],
                    }
                ),
                pd.Timestamp("2020-01-02"),
                2,
            )


def test_torch_models_smoke_when_installed():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    y = np.sin(np.arange(120, dtype=float) / 4.0) + 0.05 * np.arange(120, dtype=float)

    cases = [
        ("torch-mlp-direct", {"lags": 24, "hidden_sizes": (16,), "epochs": 8, "batch_size": 16}),
        ("torch-lstm-direct", {"lags": 24, "hidden_size": 16, "epochs": 8, "batch_size": 16}),
        ("torch-gru-direct", {"lags": 24, "hidden_size": 16, "epochs": 8, "batch_size": 16}),
        (
            "torch-tcn-direct",
            {"lags": 24, "channels": (8, 8), "kernel_size": 3, "epochs": 8, "batch_size": 16},
        ),
        (
            "torch-nbeats-direct",
            {
                "lags": 24,
                "num_blocks": 2,
                "num_layers": 2,
                "layer_width": 32,
                "epochs": 8,
                "batch_size": 16,
            },
        ),
        ("torch-nlinear-direct", {"lags": 24, "epochs": 8, "batch_size": 16}),
        ("torch-dlinear-direct", {"lags": 32, "ma_window": 7, "epochs": 8, "batch_size": 16}),
        (
            "torch-transformer-direct",
            {
                "lags": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "epochs": 6,
                "batch_size": 16,
            },
        ),
        (
            "torch-patchtst-direct",
            {
                "lags": 64,
                "patch_len": 8,
                "stride": 4,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "epochs": 6,
                "batch_size": 16,
            },
        ),
        (
            "torch-tsmixer-direct",
            {
                "lags": 32,
                "d_model": 16,
                "num_blocks": 2,
                "token_mixing_hidden": 32,
                "channel_mixing_hidden": 32,
                "epochs": 6,
                "batch_size": 16,
            },
        ),
        (
            "torch-cnn-direct",
            {
                "lags": 32,
                "channels": (8, 8),
                "kernel_size": 3,
                "epochs": 6,
                "batch_size": 16,
            },
        ),
        (
            "torch-bilstm-direct",
            {"lags": 24, "hidden_size": 8, "num_layers": 1, "epochs": 6, "batch_size": 16},
        ),
        (
            "torch-fnet-direct",
            {
                "lags": 32,
                "d_model": 16,
                "num_layers": 1,
                "dim_feedforward": 32,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
        (
            "torch-linear-attn-direct",
            {
                "lags": 32,
                "d_model": 16,
                "num_layers": 1,
                "dim_feedforward": 32,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
    ]

    for key, params in cases:
        f = make_forecaster(key, **params, seed=0, patience=2, device="cpu")
        yhat = f(y, 5)
        assert yhat.shape == (5,)
        assert np.all(np.isfinite(yhat))


def test_torch_global_models_smoke_when_installed():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    import pandas as pd

    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=80, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.5), ("s2", 0.8)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.01 * np.arange(ds.size)
        y = base + 0.05 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.1).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    cutoff = ds[-6]  # leave 5 days for horizon
    horizon = 5

    cases = [
        (
            "torch-tft-global",
            {"context_length": 32, "epochs": 2, "batch_size": 32, "x_cols": ("promo",)},
        ),
        (
            "torch-informer-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-autoformer-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
                "ma_window": 7,
            },
        ),
    ]

    for key, params in cases:
        g = make_global_forecaster(key, **params, seed=0, patience=2, device="cpu")
        pred = g(long_df, cutoff, horizon)
        assert set(pred.columns) >= {"unique_id", "ds", "yhat"}
        assert len(pred) > 0
        assert np.all(np.isfinite(pred["yhat"].to_numpy(dtype=float)))
