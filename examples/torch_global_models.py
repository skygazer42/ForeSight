from __future__ import annotations

import numpy as np
import pandas as pd

from foresight.models.registry import make_global_forecaster


def main() -> None:
    """
    Demo: global/panel Torch models with covariates + time features (and optional quantiles).

    Run:
      pip install -e ".[dev,torch]"
      python examples/torch_global_models.py
    """
    try:
        import torch  # noqa: F401
    except Exception:
        print('This example requires torch. Install with: pip install -e ".[dev,torch]"')
        return

    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=220, freq="D")

    rows = []
    for uid, amp, bias in [("store=0", 1.0, 0.0), ("store=1", 1.4, 0.3), ("store=2", 0.8, -0.2)]:
        t = np.arange(ds.size, dtype=float)
        promo = (rng.random(ds.size) < 0.08).astype(float)
        price = 1.0 + 0.05 * rng.standard_normal(ds.size)
        seasonal = amp * np.sin(2.0 * np.pi * t / 30.0)
        trend = 0.01 * t
        y = (
            bias
            + seasonal
            + trend
            + 0.4 * promo
            - 0.2 * (price - 1.0)
            + 0.1 * rng.standard_normal(ds.size)
        )

        for d, yv, pv, pr in zip(ds, y, promo, price, strict=True):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": float(yv),
                    "promo": float(pv),
                    "price": float(pr),
                }
            )

    long_df = pd.DataFrame(rows).sort_values(["unique_id", "ds"], kind="mergesort")

    horizon = 14
    cutoff = ds[-(horizon + 1)]

    models = [
        ("torch-tft-global", {"context_length": 64, "epochs": 10, "batch_size": 64}),
        (
            "torch-informer-global",
            {"context_length": 96, "d_model": 64, "num_layers": 2, "epochs": 10},
        ),
        (
            "torch-autoformer-global",
            {"context_length": 96, "d_model": 64, "num_layers": 2, "epochs": 10},
        ),
        (
            "torch-patchtst-global",
            {
                "context_length": 96,
                "d_model": 64,
                "num_layers": 2,
                "patch_len": 16,
                "stride": 8,
                "epochs": 10,
            },
        ),
        (
            "torch-tsmixer-global",
            {"context_length": 96, "d_model": 64, "num_blocks": 4, "epochs": 10},
        ),
        (
            "torch-itransformer-global",
            {
                "context_length": 96,
                "d_model": 64,
                "num_layers": 2,
                "epochs": 10,
                "quantiles": "0.1,0.5,0.9",
            },
        ),
        (
            "torch-timesnet-global",
            {"context_length": 96, "d_model": 64, "num_layers": 2, "top_k": 3, "epochs": 10},
        ),
        (
            "torch-tcn-global",
            {"context_length": 96, "channels": (64, 64, 64), "kernel_size": 3, "epochs": 10},
        ),
        (
            "torch-nbeats-global",
            {"context_length": 96, "num_blocks": 3, "layer_width": 256, "epochs": 10},
        ),
        (
            "torch-nhits-global",
            {
                "context_length": 128,
                "pool_sizes": (1, 2, 4),
                "num_blocks": 6,
                "layer_width": 256,
                "epochs": 10,
            },
        ),
        (
            "torch-tide-global",
            {"context_length": 96, "d_model": 64, "hidden_size": 128, "epochs": 10},
        ),
        (
            "torch-seq2seq-attn-lstm-global",
            {
                "context_length": 64,
                "hidden_size": 64,
                "attention": "bahdanau",
                "teacher_forcing": 0.6,
                "teacher_forcing_final": 0.0,
                "epochs": 10,
            },
        ),
    ]

    for key, params in models:
        g = make_global_forecaster(
            key,
            **params,
            x_cols=("promo", "price"),
            add_time_features=True,
            seed=0,
            patience=5,
            device="cpu",
        )
        pred = g(long_df, cutoff, horizon)
        print(f"\n=== {key} ===")
        print(pred.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
