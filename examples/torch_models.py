from __future__ import annotations

import numpy as np

from foresight.models.registry import make_forecaster


def main() -> None:
    """
    Demo: modern torch baselines (per-series per-window training).

    Run:
      pip install -e ".[dev,torch]"
      python examples/torch_models.py
    """
    try:
        import torch  # noqa: F401
    except Exception:
        print('This example requires torch. Install with: pip install -e ".[dev,torch]"')
        return

    rng = np.random.default_rng(0)
    t = np.arange(250, dtype=float)
    y = 0.02 * t + np.sin(2.0 * np.pi * t / 30.0) + 0.2 * rng.standard_normal(t.shape[0])

    horizon = 14

    models = [
        ("torch-mlp-direct", {"lags": 48, "hidden_sizes": (32, 32), "epochs": 50}),
        ("torch-lstm-direct", {"lags": 48, "hidden_size": 32, "epochs": 50}),
        ("torch-gru-direct", {"lags": 48, "hidden_size": 32, "epochs": 50}),
        (
            "torch-tcn-direct",
            {"lags": 48, "channels": (16, 16, 16), "kernel_size": 3, "epochs": 50},
        ),
        (
            "torch-nbeats-direct",
            {"lags": 48, "num_blocks": 3, "num_layers": 2, "layer_width": 64, "epochs": 50},
        ),
        ("torch-nlinear-direct", {"lags": 48, "epochs": 50}),
        ("torch-dlinear-direct", {"lags": 48, "ma_window": 9, "epochs": 50}),
        (
            "torch-transformer-direct",
            {
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                "epochs": 50,
            },
        ),
        (
            "torch-patchtst-direct",
            {
                "lags": 192,
                "patch_len": 16,
                "stride": 8,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                "epochs": 50,
            },
        ),
        (
            "torch-tsmixer-direct",
            {
                "lags": 96,
                "d_model": 64,
                "num_blocks": 4,
                "token_mixing_hidden": 128,
                "channel_mixing_hidden": 128,
                "dropout": 0.1,
                "epochs": 50,
            },
        ),
        (
            "torch-cnn-direct",
            {"lags": 96, "channels": (32, 32, 32), "kernel_size": 3, "dropout": 0.1, "epochs": 50},
        ),
        (
            "torch-wavenet-direct",
            {"lags": 96, "channels": 32, "num_layers": 6, "kernel_size": 2, "epochs": 50},
        ),
        (
            "torch-attn-gru-direct",
            {"lags": 48, "hidden_size": 32, "epochs": 50},
        ),
        (
            "torch-fnet-direct",
            {"lags": 96, "d_model": 64, "num_layers": 4, "epochs": 50},
        ),
        (
            "torch-inception-direct",
            {"lags": 96, "channels": 32, "num_blocks": 3, "epochs": 50},
        ),
    ]

    for key, params in models:
        f = make_forecaster(
            key, **params, batch_size=32, lr=1e-3, seed=0, patience=10, device="cpu"
        )
        yhat = f(y, horizon)
        print(f"\n=== {key} ===")
        print(np.round(yhat[:10], 3))


if __name__ == "__main__":
    main()
