from __future__ import annotations

# Lane 06 ownership: TimeGrad / TACTiS style probabilistic lite families.
import math
from typing import Any

import numpy as np

from .torch_nn import (
    TorchTrainConfig,
    _as_1d_float_array,
    _make_lagged_xy_multi,
    _make_manual_gru,
    _normalize_series,
    _require_torch,
    _train_loop,
)


def torch_probabilistic_direct_forecast(
    train: Any,
    horizon: int,
    *,
    variant: str,
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    num_heads: int = 4,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "gaussian",
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
    TimeGrad/TACTiS-style lite probabilistic forecasters.

    These implementations keep only a small Gaussian head and return the
    predictive mean as a point forecast. They do not expose full diffusion,
    copula, or trajectory-sampling semantics.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    hidden = int(hidden_size)
    layers = int(num_layers)
    heads = int(num_heads)
    drop = float(dropout)
    variant_s = str(variant).strip().lower()
    loss_s = str(loss).strip().lower()

    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if hidden <= 0:
        raise ValueError("hidden_size must be >= 1")
    if layers <= 0:
        raise ValueError("num_layers must be >= 1")
    if heads <= 0:
        raise ValueError("num_heads must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")
    if variant_s not in {"timegrad", "tactis"}:
        raise ValueError("variant must be one of: timegrad, tactis")
    if variant_s == "tactis" and hidden % heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads for tactis")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _GaussianMeanHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(p=drop) if drop > 0.0 else nn.Identity(),
                nn.Linear(hidden, 2),
            )

        def forward(self, z: Any) -> Any:
            return self.proj(z)

    class _TimeGradLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rnn_drop = drop if layers > 1 else 0.0
            self.rnn = _make_manual_gru(
                input_size=1,
                hidden_size=hidden,
                num_layers=layers,
                dropout=rnn_drop,
                bidirectional=False,
            )
            self.step_emb = nn.Embedding(h, hidden)
            self.head = _GaussianMeanHead()

        def forward(self, xb: Any) -> Any:
            out, _ = self.rnn(xb)
            ctx = out[:, -1, :]
            steps = self.step_emb(torch.arange(h, device=xb.device, dtype=torch.long))
            z = ctx.unsqueeze(1) + steps.unsqueeze(0)
            return self.head(z)

    class _TACTiSLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value_proj = nn.Linear(1, hidden)
            self.pos = nn.Parameter(torch.zeros((1, lag_count, hidden), dtype=torch.float32))
            self.query = nn.Embedding(h, hidden)
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden,
                num_heads=heads,
                dropout=drop,
                batch_first=True,
            )
            self.head = _GaussianMeanHead()

        def forward(self, xb: Any) -> Any:
            ctx = self.value_proj(xb) + self.pos
            q = self.query(torch.arange(h, device=xb.device, dtype=torch.long))
            q = q.unsqueeze(0).expand(int(xb.shape[0]), -1, -1)
            attn_out, _ = self.attn(q, ctx, ctx, need_weights=False)
            return self.head(attn_out)

    def _make_loss() -> Any:
        def _loss(pred: Any, yb: Any) -> Any:
            mu = pred[..., 0]
            sigma = F.softplus(pred[..., 1]) + 1e-3

            if loss_s in {"gaussian", "nll", "probabilistic"}:
                z = (yb - mu) / sigma
                return (
                    0.5 * math.log(2.0 * math.pi) + torch.log(sigma) + 0.5 * (z**2)
                ).mean()
            if loss_s in {"mse", ""}:
                return ((mu - yb) ** 2).mean()
            if loss_s in {"mae", "l1"}:
                return (mu - yb).abs().mean()
            if loss_s in {"huber", "smoothl1"}:
                return F.smooth_l1_loss(mu, yb)
            raise ValueError("loss must be one of: gaussian, mse, mae, huber")

        return _loss

    if variant_s == "timegrad":
        model = _TimeGradLite()
    else:
        model = _TACTiSLite()

    cfg = TorchTrainConfig(
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        seed=int(seed),
        patience=int(patience),
        loss="mse",
        val_split=float(val_split),
        grad_clip_norm=float(grad_clip_norm),
        optimizer=str(optimizer),
        momentum=float(momentum),
        scheduler=str(scheduler),
        scheduler_step_size=int(scheduler_step_size),
        scheduler_gamma=float(scheduler_gamma),
        restore_best=bool(restore_best),
    )
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device), loss_fn_override=_make_loss())

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        pred = model(feat_t)
        yhat_t = pred[0, :, 0].detach().cpu().numpy()

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)
