from __future__ import annotations

# Lane 04 ownership: AGCRN / MTGNN style graph-structure lite families.
from typing import Any

import numpy as np

from .multivariate import torch_graphwavenet_forecast


def torch_graph_structure_forecast(
    train: Any,
    horizon: int,
    *,
    variant: str,
    lags: int = 24,
    d_model: int = 64,
    num_blocks: int = 4,
    kernel_size: int = 2,
    dilation_base: int = 2,
    dropout: float = 0.1,
    adj: Any = "corr",
    adj_path: str = "",
    adj_top_k: int = 8,
    adaptive_adj: bool = True,
    adj_emb_dim: int = 8,
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
    **_params: Any,
) -> np.ndarray:
    """
    AGCRN/MTGNN-style graph-structure lite forecasters.

    Both variants reuse the existing adaptive Graph WaveNet-style multivariate
    backbone with variant-specific defaults. This is an honest graph-structure
    proxy, not a paper-faithful reimplementation of AGCRN or MTGNN.
    """
    variant_s = str(variant).strip().lower()
    if variant_s not in {"agcrn", "mtgnn"}:
        raise ValueError("variant must be one of: agcrn, mtgnn")

    return torch_graphwavenet_forecast(
        train,
        int(horizon),
        lags=int(lags),
        d_model=int(d_model),
        num_blocks=int(num_blocks),
        kernel_size=int(kernel_size),
        dilation_base=int(dilation_base),
        dropout=float(dropout),
        adj=adj,
        adj_path=str(adj_path),
        adj_top_k=int(adj_top_k),
        adaptive_adj=bool(adaptive_adj),
        adj_emb_dim=int(adj_emb_dim),
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
    )
