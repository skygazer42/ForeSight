from __future__ import annotations

from typing import Any

import numpy as np

from .torch_rnn_paper_zoo import torch_rnnpaper_direct_forecast

_VARIANT_TO_PAPER = {
    "esn": "echo-state-network",
    "deep-esn": "deep-esn",
    "liquid-state": "liquid-state-machine",
}


def torch_reservoir_direct_forecast(
    train: Any,
    horizon: int,
    *,
    variant: str,
    lags: int = 24,
    hidden_size: int = 32,
    spectral_radius: float = 0.9,
    leak: float = 1.0,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    restore_best: bool = True,
    **params: Any,
) -> np.ndarray:
    variant_s = str(variant).strip().lower()
    try:
        paper_id = _VARIANT_TO_PAPER[variant_s]
    except KeyError as e:
        raise ValueError(
            f"Unknown reservoir variant {variant!r}. "
            f"Expected one of: {sorted(_VARIANT_TO_PAPER)}"
        ) from e

    return torch_rnnpaper_direct_forecast(
        train,
        int(horizon),
        paper=paper_id,
        lags=int(lags),
        hidden_size=int(hidden_size),
        spectral_radius=float(spectral_radius),
        leak=float(leak),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        seed=int(seed),
        normalize=bool(normalize),
        device=str(device),
        patience=int(patience),
        loss=str(loss),
        val_split=float(val_split),
        grad_clip_norm=float(grad_clip_norm),
        optimizer=str(optimizer),
        momentum=float(momentum),
        scheduler=str(scheduler),
        scheduler_step_size=int(scheduler_step_size),
        scheduler_gamma=float(scheduler_gamma),
        restore_best=bool(restore_best),
        **params,
    )
