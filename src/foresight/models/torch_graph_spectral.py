from __future__ import annotations

# Lane 05 ownership: StemGNN / FourierGNN style graph-spectral lite families.
from typing import Any

import numpy as np

from .multivariate import (
    TorchTrainConfig,
    _as_2d_float_array,
    _make_lagged_xy_multivariate,
    _normalize_multivariate_matrix,
    _require_torch,
    _train_loop,
)


def torch_graph_spectral_forecast(
    train: Any,
    horizon: int,
    *,
    variant: str,
    lags: int = 24,
    d_model: int = 64,
    num_blocks: int = 2,
    top_k_freq: int = 8,
    dropout: float = 0.1,
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
) -> np.ndarray:
    """
    StemGNN/FourierGNN-style graph-spectral lite forecasters.

    These models use compact FFT-derived node features and small learned node
    mixers. They are spectral proxies, not full graph spectral reproductions.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_2d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    d = int(d_model)
    blocks = int(num_blocks)
    k_freq = int(top_k_freq)
    drop = float(dropout)
    variant_s = str(variant).strip().lower()
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if d <= 0:
        raise ValueError("d_model must be >= 1")
    if blocks <= 0:
        raise ValueError("num_blocks must be >= 1")
    if k_freq <= 0:
        raise ValueError("top_k_freq must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0,1)")
    if variant_s not in {"stemgnn", "fouriergnn"}:
        raise ValueError("variant must be one of: stemgnn, fouriergnn")

    x_work = x.astype(float, copy=False)
    mean = np.zeros((int(x.shape[1]),), dtype=float)
    std = np.ones((int(x.shape[1]),), dtype=float)
    if bool(normalize):
        x_work, mean, std = _normalize_multivariate_matrix(x_work)

    X, Y = _make_lagged_xy_multivariate(x_work, lags=lag_count, horizon=h)
    n_nodes = int(x.shape[1])
    freq_bins = min(k_freq, lag_count // 2 + 1)

    class _SpectralBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_mix = nn.Linear(n_nodes, n_nodes)
            self.norm = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(2 * d, d),
            )

        def forward(self, xb: Any) -> Any:
            z = self.node_mix(xb.transpose(1, 2)).transpose(1, 2)
            xb = xb + z
            return xb + self.ffn(self.norm(xb))

    class _StemGNNLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.history_proj = nn.Linear(lag_count, d)
            self.spectral_proj = nn.Linear(2 * freq_bins, d)
            self.blocks = nn.ModuleList([_SpectralBlock() for _ in range(blocks)])
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:
            x_nodes = xb.transpose(1, 2)
            spec = torch.fft.rfft(x_nodes, dim=-1)[..., :freq_bins]
            spec_feat = torch.cat([spec.real, spec.imag], dim=-1)
            z = self.history_proj(x_nodes) + self.spectral_proj(spec_feat)
            for blk in self.blocks:
                z = blk(z)
            return self.head(z).transpose(1, 2)

    class _FourierGNNLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spectral_proj = nn.Linear(2 * freq_bins, d)
            self.node_emb = nn.Parameter(torch.zeros((1, n_nodes, d), dtype=torch.float32))
            self.blocks = nn.ModuleList([_SpectralBlock() for _ in range(blocks)])
            self.horizon_emb = nn.Embedding(h, d)
            self.head = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Dropout(p=drop) if drop > 0.0 else nn.Identity(),
                nn.Linear(d, 1),
            )

        def forward(self, xb: Any) -> Any:
            x_nodes = xb.transpose(1, 2)
            spec = torch.fft.rfft(x_nodes, dim=-1)[..., :freq_bins]
            spec_feat = torch.cat([spec.real, spec.imag], dim=-1)
            z = self.spectral_proj(spec_feat) + self.node_emb
            for blk in self.blocks:
                z = blk(z)
            steps = self.horizon_emb(torch.arange(h, device=xb.device, dtype=torch.long))
            z_out = z.unsqueeze(1) + steps.reshape(1, h, 1, d)
            return self.head(z_out).squeeze(-1)

    model: Any
    if variant_s == "stemgnn":
        model = _StemGNNLite()
    else:
        model = _FourierGNNLite()

    cfg = TorchTrainConfig(
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        seed=int(seed),
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
    model = _train_loop(model, X, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:, :].astype(float, copy=False).reshape(1, lag_count, n_nodes)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(h, n_nodes)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std.reshape(1, n_nodes) + mean.reshape(1, n_nodes)
    return np.asarray(yhat, dtype=float)
