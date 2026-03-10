from __future__ import annotations

# Lane 03 ownership: ASTGCN / GMAN style graph-attention lite families.
from typing import Any

import numpy as np

from .multivariate import (
    TorchTrainConfig,
    _as_2d_float_array,
    _make_lagged_xy_multivariate,
    _normalize_multivariate_matrix,
    _require_torch,
    _resolve_adj_matrix,
    _train_loop,
)


def torch_graph_attention_forecast(
    train: Any,
    horizon: int,
    *,
    variant: str,
    lags: int = 24,
    d_model: int = 64,
    num_blocks: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    adj: Any = "corr",
    adj_path: str = "",
    adj_top_k: int = 8,
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
    ASTGCN/GMAN-style graph-attention lite forecasters on wide `(T, N)` targets.

    These are compact attention-based multivariate proxies, not paper-complete
    reproductions of the original graph-attention architectures.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_2d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    d = int(d_model)
    blocks = int(num_blocks)
    heads = int(num_heads)
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
    if heads <= 0:
        raise ValueError("num_heads must be >= 1")
    if d % heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0,1)")
    if variant_s not in {"astgcn", "gman"}:
        raise ValueError("variant must be one of: astgcn, gman")

    x_work = x.astype(float, copy=False)
    mean = np.zeros((int(x.shape[1]),), dtype=float)
    std = np.ones((int(x.shape[1]),), dtype=float)
    if bool(normalize):
        x_work, mean, std = _normalize_multivariate_matrix(x_work)

    X, Y = _make_lagged_xy_multivariate(x_work, lags=lag_count, horizon=h)
    n_nodes = int(x.shape[1])
    adj_mat = _resolve_adj_matrix(
        adj=adj,
        adj_path=str(adj_path),
        n_nodes=n_nodes,
        x_work=x_work,
        top_k=int(adj_top_k),
    )

    class _ASTGCNBlock(nn.Module):
        def __init__(self, *, a: Any) -> None:
            super().__init__()
            self.register_buffer("adj", a)
            self.temporal_attn = nn.MultiheadAttention(
                d, heads, dropout=drop, batch_first=True
            )
            self.spatial_attn = nn.MultiheadAttention(
                d, heads, dropout=drop, batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(d)
            self.spatial_norm = nn.LayerNorm(d)

        def forward(self, xb: Any) -> Any:
            bsz, steps, nodes, width = xb.shape

            z = xb + torch.einsum("ij,btjd->btid", self.adj, xb)

            z_t = z.permute(0, 2, 1, 3).reshape(bsz * nodes, steps, width)
            attn_t, _ = self.temporal_attn(z_t, z_t, z_t, need_weights=False)
            z_t = self.temporal_norm(z_t + attn_t)
            z = z_t.reshape(bsz, nodes, steps, width).permute(0, 2, 1, 3)

            z_s = z[:, -1, :, :]
            attn_s, _ = self.spatial_attn(z_s, z_s, z_s, need_weights=False)
            prior = torch.einsum("ij,bjd->bid", self.adj, z_s)
            z_s = self.spatial_norm(z_s + attn_s + prior)
            z = z.clone()
            z[:, -1, :, :] = z_s
            return z

    class _GMANBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spatial_attn = nn.MultiheadAttention(
                d, heads, dropout=drop, batch_first=True
            )
            self.temporal_attn = nn.MultiheadAttention(
                d, heads, dropout=drop, batch_first=True
            )
            self.spatial_norm = nn.LayerNorm(d)
            self.temporal_norm = nn.LayerNorm(d)

        def forward(self, xb: Any) -> Any:
            bsz, steps, nodes, width = xb.shape

            z_s = xb.reshape(bsz * steps, nodes, width)
            attn_s, _ = self.spatial_attn(z_s, z_s, z_s, need_weights=False)
            z_s = self.spatial_norm(z_s + attn_s)
            z = z_s.reshape(bsz, steps, nodes, width)

            z_t = z.permute(0, 2, 1, 3).reshape(bsz * nodes, steps, width)
            attn_t, _ = self.temporal_attn(z_t, z_t, z_t, need_weights=False)
            z_t = self.temporal_norm(z_t + attn_t)
            return z_t.reshape(bsz, nodes, steps, width).permute(0, 2, 1, 3)

    class _ASTGCNLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(1, d)
            self.time_emb = nn.Parameter(torch.zeros((1, lag_count, 1, d), dtype=torch.float32))
            a = torch.tensor(adj_mat, dtype=torch.float32)
            self.blocks = nn.ModuleList([_ASTGCNBlock(a=a) for _ in range(blocks)])
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:
            z = self.in_proj(xb.unsqueeze(-1)) + self.time_emb
            for blk in self.blocks:
                z = blk(z)
            return self.head(z[:, -1, :, :]).transpose(1, 2)

    class _GMANLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(1, d)
            self.time_emb = nn.Parameter(torch.zeros((1, lag_count, 1, d), dtype=torch.float32))
            self.node_emb = nn.Parameter(torch.zeros((1, 1, n_nodes, d), dtype=torch.float32))
            self.blocks = nn.ModuleList([_GMANBlock() for _ in range(blocks)])
            self.horizon_emb = nn.Embedding(h, d)
            self.head = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Dropout(p=drop) if drop > 0.0 else nn.Identity(),
                nn.Linear(d, 1),
            )

        def forward(self, xb: Any) -> Any:
            z = self.in_proj(xb.unsqueeze(-1)) + self.time_emb + self.node_emb
            for blk in self.blocks:
                z = blk(z)
            ctx = z[:, -1, :, :]
            steps = self.horizon_emb(torch.arange(h, device=xb.device, dtype=torch.long))
            z_out = ctx.unsqueeze(1) + steps.reshape(1, h, 1, d)
            return self.head(z_out).squeeze(-1)

    model: Any
    if variant_s == "astgcn":
        model = _ASTGCNLite()
    else:
        model = _GMANLite()

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
