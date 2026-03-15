from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .torch_nn import (
    TorchTrainConfig,
    _as_1d_float_array,
    _make_lagged_xy_multi,
    _normalize_series,
    _require_torch,
    _train_loop,
)

RNNZooVariant = Literal["direct", "bidir", "ln", "attn", "proj"]


@dataclass(frozen=True)
class RNNZooModelSpec:
    key: str
    base: str
    variant: RNNZooVariant
    description: str


_BASE_DESCRIPTIONS: dict[str, str] = {
    "elman": "Elman RNN (Elman, 1990)",
    "lstm": "LSTM (Hochreiter & Schmidhuber, 1997)",
    "gru": "GRU (Cho et al., 2014)",
    "peephole-lstm": "Peephole LSTM (Gers et al., 2002)",
    "cifg-lstm": "CIFG / Coupled LSTM (Greff et al., 2015)",
    "janet": "JANET / Forget-gate LSTM (van der Westhuizen & Lasenby, 2018)",
    "indrnn": "IndRNN (Li et al., 2018)",
    "minimalrnn": "MinimalRNN (Chen, 2017)",
    "mgu": "MGU / Minimal Gated Unit (Zhou et al., 2016)",
    "fastrnn": "FastRNN (Kusupati et al., 2018)",
    "fastgrnn": "FastGRNN (Kusupati et al., 2018)",
    "mut1": "MUT1 (Jozefowicz et al., 2015)",
    "mut2": "MUT2 (Jozefowicz et al., 2015)",
    "mut3": "MUT3 (Jozefowicz et al., 2015)",
    "ran": "Recurrent Additive Network (Lee et al., 2017)",
    "scrn": "SCRN / Structurally Constrained RNN (Mikolov et al., 2014)",
    "rhn": "Recurrent Highway Network (Zilly et al., 2017)",
    "clockwork": "Clockwork RNN (Koutn\u00edk et al., 2014)",
    "qrnn": "QRNN / Quasi-Recurrent Neural Network (Bradbury et al., 2016)",
    "phased-lstm": "Phased LSTM (Neil et al., 2016)",
}

_VARIANT_DESCRIPTIONS: dict[RNNZooVariant, str] = {
    "direct": "direct head (last hidden -> horizon)",
    "bidir": "bidirectional wrapper (Schuster & Paliwal, 1997)",
    "ln": "LayerNorm head (Ba et al., 2016)",
    "attn": "additive attention pooling (Bahdanau et al., 2015)",
    "proj": "projection head (Sak et al., 2014)",
}


def list_rnnzoo_specs() -> list[RNNZooModelSpec]:
    """
    Return the 100 Torch RNN Zoo model specs (20 bases × 5 variants).
    """
    bases = list(_BASE_DESCRIPTIONS.keys())
    variants: list[RNNZooVariant] = ["direct", "bidir", "ln", "attn", "proj"]
    out: list[RNNZooModelSpec] = []
    for base in bases:
        base_desc = _BASE_DESCRIPTIONS[base]
        for v in variants:
            if v == "direct":
                key = f"torch-rnnzoo-{base}-direct"
            else:
                key = f"torch-rnnzoo-{base}-{v}-direct"
            desc = f"{base_desc}; {_VARIANT_DESCRIPTIONS[v]}"
            out.append(RNNZooModelSpec(key=key, base=base, variant=v, description=desc))
    return out


def _validate_rnnzoo_direct_config(
    *,
    horizon: int,
    lags: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    proj_size: int,
    attn_hidden: int,
    base: str,
    variant: RNNZooVariant,
) -> tuple[int, int, int, int, float, int, int, str, RNNZooVariant]:
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")

    lag_count = int(lags)
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")

    hid = int(hidden_size)
    if hid <= 0:
        raise ValueError("hidden_size must be >= 1")

    layers = int(num_layers)
    if layers <= 0:
        raise ValueError("num_layers must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    proj = int(proj_size)
    if proj <= 0:
        raise ValueError("proj_size must be >= 1")

    attn_hid = int(attn_hidden)
    if attn_hid <= 0:
        raise ValueError("attn_hidden must be >= 1")

    base_s = str(base).strip().lower()
    variant_s: RNNZooVariant = str(variant).strip().lower()  # type: ignore[assignment]
    if base_s not in _BASE_DESCRIPTIONS:
        raise ValueError(f"Unknown rnnzoo base: {base_s!r}")
    if variant_s not in {"direct", "bidir", "ln", "attn", "proj"}:
        raise ValueError("variant must be one of: direct, bidir, ln, attn, proj")

    return h, lag_count, hid, layers, drop, proj, attn_hid, base_s, variant_s


def _normalize_clock_periods(clock_periods: Any) -> tuple[int, ...]:
    if isinstance(clock_periods, str):
        parts = [part.strip() for part in clock_periods.split(",") if part.strip()]
        return tuple(int(part) for part in parts)
    if isinstance(clock_periods, list | tuple):
        return tuple(int(period) for period in clock_periods)
    return (int(clock_periods),)


def _build_rnnzoo_train_config(
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    patience: int,
    loss: str,
    val_split: float,
    grad_clip_norm: float,
    optimizer: str,
    momentum: float,
    scheduler: str,
    scheduler_step_size: int,
    scheduler_gamma: float,
    restore_best: bool,
) -> TorchTrainConfig:
    return TorchTrainConfig(
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


def _prepare_rnnzoo_payload(
    train: Any,
    horizon: int,
    *,
    lags: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    x = _as_1d_float_array(train)
    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=int(lags), horizon=int(horizon))
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)
    return x_work, x_seq, Y, mean, std


def _predict_rnnzoo_direct_model(
    torch: Any,
    model: Any,
    history: np.ndarray,
    *,
    lag_count: int,
    device: str,
) -> np.ndarray:
    feat = history[-int(lag_count) :].astype(float, copy=False).reshape(1, int(lag_count), 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        return model(feat_t).detach().cpu().numpy().reshape(-1)


def _maybe_denormalize_rnnzoo_forecast(
    yhat_t: np.ndarray,
    *,
    normalize: bool,
    mean: float,
    std: float,
) -> np.ndarray:
    yhat = np.asarray(yhat_t, dtype=float).astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_rnnzoo_direct_forecast(
    train: Any,
    horizon: int,
    *,
    base: str,
    variant: RNNZooVariant = "direct",
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    proj_size: int = 16,
    attn_hidden: int = 32,
    # clockwork
    clock_periods: Any = (1, 2, 4, 8),
    # qrnn
    qrnn_kernel_size: int = 3,
    # rhn
    rhn_depth: int = 2,
    # phased lstm
    phased_tau: float = 32.0,
    phased_r_on: float = 0.05,
    phased_leak: float = 0.001,
    # training
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
    Torch RNN Zoo: a compact set of paper-named RNN architectures with a unified
    lag-window direct forecasting interface.

    Notes:
    - These are **lite** implementations aimed at fast baselines and walk-forward evaluation.
    - All models follow (train_1d, horizon) -> yhat[horizon] and train per-series.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    h, lag_count, hid, layers, drop, proj, attn_hid, base_s, variant_s = (
        _validate_rnnzoo_direct_config(
            horizon=horizon,
            lags=lags,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            proj_size=proj_size,
            attn_hidden=attn_hidden,
            base=base,
            variant=variant,
        )
    )
    x_work, x_seq, Y, mean, std = _prepare_rnnzoo_payload(
        train,
        h,
        lags=lag_count,
        normalize=bool(normalize),
    )

    def _ensure_device(xb: Any) -> Any:
        dev = torch.device(str(device))
        return xb.to(dev)

    class _SeqEncoder(nn.Module):
        def forward(self, xb: Any) -> Any:
            raise NotImplementedError

    class _ElmanEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.ModuleList(
                [nn.Linear(1, hid, bias=True)]
                + [nn.Linear(hid, hid, bias=True) for _ in range(int(layers) - 1)]
            )
            self.h2h = nn.ModuleList([nn.Linear(hid, hid, bias=False) for _ in range(int(layers))])

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            hs = [
                torch.zeros((B, hid), device=xb.device, dtype=xb.dtype) for _ in range(int(layers))
            ]
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                for i in range(int(layers)):
                    h_prev = hs[i]
                    h_t = torch.tanh(self.x2h[i](x_t) + self.h2h[i](h_prev))
                    hs[i] = h_t
                    x_t = h_t
                    if i < int(layers) - 1 and drop > 0.0 and self.training:
                        x_t = F.dropout(x_t, p=float(drop), training=True)
                outs.append(x_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _GRUEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.ModuleList(
                [nn.Linear(1, 3 * hid, bias=True)]
                + [nn.Linear(hid, 3 * hid, bias=True) for _ in range(int(layers) - 1)]
            )
            self.h2h = nn.ModuleList(
                [nn.Linear(hid, 3 * hid, bias=False) for _ in range(int(layers))]
            )

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            hs = [
                torch.zeros((B, hid), device=xb.device, dtype=xb.dtype) for _ in range(int(layers))
            ]
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                for i in range(int(layers)):
                    h_prev = hs[i]
                    gi = self.x2h[i](x_t)
                    gh = self.h2h[i](h_prev)
                    i_r, i_z, i_n = gi.chunk(3, dim=-1)
                    h_r, h_z, h_n = gh.chunk(3, dim=-1)
                    r = torch.sigmoid(i_r + h_r)
                    z = torch.sigmoid(i_z + h_z)
                    n = torch.tanh(i_n + r * h_n)
                    h_t = (1.0 - z) * n + z * h_prev
                    hs[i] = h_t
                    x_t = h_t
                    if i < int(layers) - 1 and drop > 0.0 and self.training:
                        x_t = F.dropout(x_t, p=float(drop), training=True)
                outs.append(x_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _LSTMEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.ModuleList(
                [nn.Linear(1, 4 * hid, bias=True)]
                + [nn.Linear(hid, 4 * hid, bias=True) for _ in range(int(layers) - 1)]
            )
            self.h2h = nn.ModuleList(
                [nn.Linear(hid, 4 * hid, bias=False) for _ in range(int(layers))]
            )

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            hs = [
                torch.zeros((B, hid), device=xb.device, dtype=xb.dtype) for _ in range(int(layers))
            ]
            cs = [
                torch.zeros((B, hid), device=xb.device, dtype=xb.dtype) for _ in range(int(layers))
            ]
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                for i in range(int(layers)):
                    h_prev = hs[i]
                    c_prev = cs[i]
                    gates = self.x2h[i](x_t) + self.h2h[i](h_prev)
                    i_g, f_g, g_g, o_g = gates.chunk(4, dim=-1)
                    i_g = torch.sigmoid(i_g)
                    f_g = torch.sigmoid(f_g)
                    g_g = torch.tanh(g_g)
                    o_g = torch.sigmoid(o_g)
                    c_t = f_g * c_prev + i_g * g_g
                    h_t = o_g * torch.tanh(c_t)
                    hs[i] = h_t
                    cs[i] = c_t
                    x_t = h_t
                    if i < int(layers) - 1 and drop > 0.0 and self.training:
                        x_t = F.dropout(x_t, p=float(drop), training=True)
                outs.append(x_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _PeepholeLSTMEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, 4 * hid, bias=True)
            self.h2h = nn.Linear(hid, 4 * hid, bias=False)
            self.w_ci = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))
            self.w_cf = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))
            self.w_co = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            c_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                gates = self.x2h(x_t) + self.h2h(h_t)
                i, f, g, o = gates.chunk(4, dim=-1)
                i = torch.sigmoid(i + c_t * self.w_ci)
                f = torch.sigmoid(f + c_t * self.w_cf)
                g = torch.tanh(g)
                c_t = f * c_t + i * g
                o = torch.sigmoid(o + c_t * self.w_co)
                h_t = o * torch.tanh(c_t)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _CIFGLSTMEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, 3 * hid, bias=True)
            self.h2h = nn.Linear(hid, 3 * hid, bias=False)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            c_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                gates = self.x2h(x_t) + self.h2h(h_t)
                f, g, o = gates.chunk(3, dim=-1)
                f = torch.sigmoid(f)
                i = 1.0 - f
                g = torch.tanh(g)
                c_t = f * c_t + i * g
                o = torch.sigmoid(o)
                h_t = o * torch.tanh(c_t)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _JANETEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, 2 * hid, bias=True)
            self.h2h = nn.Linear(hid, 2 * hid, bias=False)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            c_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                gates = self.x2h(x_t) + self.h2h(h_t)
                f, g = gates.chunk(2, dim=-1)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                c_t = f * c_t + (1.0 - f) * g
                h_t = torch.tanh(c_t)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _IndRNNEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, hid, bias=True)
            self.u = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                pre = self.x2h(x_t) + h_t * self.u
                h_t = torch.relu(pre)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _MinimalRNNEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2z = nn.Linear(1, hid, bias=True)
            self.x2u = nn.Linear(1, hid, bias=True)
            self.h2u = nn.Linear(hid, hid, bias=False)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                z = torch.tanh(self.x2z(x_t))
                u = torch.sigmoid(self.x2u(x_t) + self.h2u(h_t))
                h_t = (1.0 - u) * h_t + u * z
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _MGUEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2f = nn.Linear(1, hid, bias=True)
            self.h2f = nn.Linear(hid, hid, bias=False)
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                f = torch.sigmoid(self.x2f(x_t) + self.h2f(h_t))
                cand = torch.tanh(self.x2h(x_t) + self.h2h(f * h_t))
                h_t = (1.0 - f) * h_t + f * cand
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _FastRNNEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)
            self.alpha_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.beta_logit = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            alpha = torch.sigmoid(self.alpha_logit)
            beta = torch.sigmoid(self.beta_logit)
            for t in range(int(T)):
                x_t = xb[:, t, :]
                h_hat = torch.tanh(self.x2h(x_t) + self.h2h(h_t))
                h_t = alpha * h_t + beta * h_hat
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _FastGRNNEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2z = nn.Linear(1, hid, bias=True)
            self.h2z = nn.Linear(hid, hid, bias=False)
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)
            self.alpha_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.beta_logit = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            alpha = torch.sigmoid(self.alpha_logit)
            beta = torch.sigmoid(self.beta_logit)
            for t in range(int(T)):
                x_t = xb[:, t, :]
                z = torch.sigmoid(self.x2z(x_t) + self.h2z(h_t))
                h_hat = torch.tanh(self.x2h(x_t) + self.h2h(z * h_t))
                h_t = (alpha * (1.0 - z) + beta) * h_t + z * h_hat
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _MUT1Encoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.w1x = nn.Linear(1, hid, bias=True)
            self.w2h = nn.Linear(hid, hid, bias=False)
            self.w3x = nn.Linear(1, hid, bias=False)
            self.w4h = nn.Linear(hid, hid, bias=False)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                a = self.w1x(x_t) * self.w2h(h_t)
                b = self.w3x(x_t) + self.w4h(h_t)
                h_t = torch.tanh(a + b)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _MUT2Encoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.w1x = nn.Linear(1, hid, bias=True)
            self.w2h = nn.Linear(hid, hid, bias=False)
            self.w3x = nn.Linear(1, hid, bias=False)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                a = self.w1x(x_t) * self.w2h(h_t)
                b = self.w3x(x_t)
                h_t = torch.tanh(a + b)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _MUT3Encoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.w1x = nn.Linear(1, hid, bias=True)
            self.w2h = nn.Linear(hid, hid, bias=False)
            self.w3h = nn.Linear(hid, hid, bias=True)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                a = self.w1x(x_t) * self.w2h(h_t)
                b = self.w3h(h_t)
                h_t = torch.tanh(a + b)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _RANEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2f = nn.Linear(1, hid, bias=True)
            self.h2f = nn.Linear(hid, hid, bias=False)
            self.x2i = nn.Linear(1, hid, bias=True)
            self.h2i = nn.Linear(hid, hid, bias=False)
            self.x2c = nn.Linear(1, hid, bias=True)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            c_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                f = torch.sigmoid(self.x2f(x_t) + self.h2f(h_t))
                i = torch.sigmoid(self.x2i(x_t) + self.h2i(h_t))
                x_proj = self.x2c(x_t)
                c_t = f * c_t + i * x_proj
                h_t = torch.tanh(c_t)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _SCRNEncoder(_SeqEncoder):
        def __init__(self, *, alpha: float = 0.95) -> None:
            super().__init__()
            self.alpha = float(alpha)
            self.x2s = nn.Linear(1, hid, bias=True)
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)
            self.s2h = nn.Linear(hid, hid, bias=False)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            s_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            a = float(self.alpha)
            for t in range(int(T)):
                x_t = xb[:, t, :]
                s_t = (1.0 - a) * s_t + a * self.x2s(x_t)
                h_t = torch.tanh(self.x2h(x_t) + self.h2h(h_t) + self.s2h(s_t))
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _RHNEncoder(_SeqEncoder):
        def __init__(self, *, depth: int) -> None:
            super().__init__()
            d = int(depth)
            if d <= 0:
                raise ValueError("rhn_depth must be >= 1")
            self.depth = d
            self.x2h = nn.ModuleList(
                [nn.Linear(1, hid, bias=True)] + [nn.Linear(1, hid) for _ in range(d - 1)]
            )
            self.h2h = nn.ModuleList([nn.Linear(hid, hid, bias=False) for _ in range(d)])
            self.x2t = nn.ModuleList(
                [nn.Linear(1, hid, bias=True)] + [nn.Linear(1, hid) for _ in range(d - 1)]
            )
            self.h2t = nn.ModuleList([nn.Linear(hid, hid, bias=False) for _ in range(d)])

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                h_in = h_t
                for i in range(self.depth):
                    if i == 0:
                        xi = x_t
                    else:
                        xi = torch.zeros_like(x_t)
                    trans = torch.tanh(self.x2h[i](xi) + self.h2h[i](h_in))
                    gate = torch.sigmoid(self.x2t[i](xi) + self.h2t[i](h_in))
                    h_in = gate * trans + (1.0 - gate) * h_in
                h_t = h_in
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _ClockworkEncoder(_SeqEncoder):
        def __init__(self, *, periods: tuple[int, ...]) -> None:
            super().__init__()
            if not periods:
                raise ValueError("clock_periods must be non-empty")
            if any(int(p) <= 0 for p in periods):
                raise ValueError("clock_periods must be positive ints")
            self.periods = tuple(int(p) for p in periods)
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)

            # Partition hidden units across modules.
            m = len(self.periods)
            base = hid // m
            sizes = [base] * m
            sizes[-1] = hid - base * (m - 1)
            self.slices: list[slice] = []
            start = 0
            for sz in sizes:
                self.slices.append(slice(start, start + sz))
                start += sz

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                cand = torch.tanh(self.x2h(x_t) + self.h2h(h_t))
                # Only update modules whose period divides (t+1) (1-indexed time).
                step = int(t + 1)
                mask = torch.zeros((hid,), device=xb.device, dtype=xb.dtype)
                for p, sl in zip(self.periods, self.slices, strict=True):
                    if step % int(p) == 0:
                        mask[sl] = 1.0
                mask = mask.view(1, hid)
                h_t = mask * cand + (1.0 - mask) * h_t
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _QRNNEncoder(_SeqEncoder):
        def __init__(self, *, kernel_size: int) -> None:
            super().__init__()
            k = int(kernel_size)
            if k <= 0:
                raise ValueError("qrnn_kernel_size must be >= 1")
            self.kernel_size = k
            self.conv = nn.Conv1d(1, 3 * hid, kernel_size=k, bias=True)

        def forward(self, xb: Any) -> Any:
            # xb: (B, T, 1) -> (B, 1, T)
            xch = xb.transpose(1, 2)
            pad = self.kernel_size - 1
            if pad > 0:
                xch = nn.functional.pad(xch, (pad, 0))
            gates = self.conv(xch).transpose(1, 2)  # (B,T,3H)
            z, f, o = gates.chunk(3, dim=-1)
            z = torch.tanh(z)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)

            B, T, _ = z.shape
            c_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                c_t = f[:, t, :] * c_t + (1.0 - f[:, t, :]) * z[:, t, :]
                h_t = o[:, t, :] * c_t
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _PhasedLSTMEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, 4 * hid, bias=True)
            self.h2h = nn.Linear(hid, 4 * hid, bias=False)

            self.tau = nn.Parameter(torch.tensor(float(phased_tau), dtype=torch.float32))
            self.phase = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        def _time_gate(self, t_idx: Any) -> Any:
            # t_idx: (T,) float tensor
            tau = torch.clamp(self.tau, min=1e-3)
            r_on = float(phased_r_on)
            leak = float(phased_leak)
            # phi in [0,1)
            phi = torch.remainder(t_idx - self.phase, tau) / tau
            # piecewise triangular gate with leak when closed
            r_on_t = torch.tensor(r_on, device=t_idx.device, dtype=t_idx.dtype)
            half = 0.5 * r_on_t
            k = torch.where(
                phi < half,
                2.0 * phi / r_on_t,
                torch.where(phi < r_on_t, 2.0 - 2.0 * phi / r_on_t, leak * phi),
            )
            return torch.clamp(k, 0.0, 1.0)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            c_t = torch.zeros((B, hid), device=xb.device, dtype=xb.dtype)
            outs = []

            t_idx = torch.arange(int(T), device=xb.device, dtype=xb.dtype)
            k_t = self._time_gate(t_idx)  # (T,)
            for t in range(int(T)):
                x_t = xb[:, t, :]
                gates = self.x2h(x_t) + self.h2h(h_t)
                i, f, g, o = gates.chunk(4, dim=-1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)

                c_cand = f * c_t + i * g
                h_cand = o * torch.tanh(c_cand)

                kt = k_t[t].view(1, 1)
                c_t = kt * c_cand + (1.0 - kt) * c_t
                h_t = kt * h_cand + (1.0 - kt) * h_t

                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    def _make_base_encoder() -> _SeqEncoder:
        simple_builders: dict[str, Any] = {
            "elman": _ElmanEncoder,
            "lstm": _LSTMEncoder,
            "gru": _GRUEncoder,
            "peephole-lstm": _PeepholeLSTMEncoder,
            "cifg-lstm": _CIFGLSTMEncoder,
            "janet": _JANETEncoder,
            "indrnn": _IndRNNEncoder,
            "minimalrnn": _MinimalRNNEncoder,
            "mgu": _MGUEncoder,
            "fastrnn": _FastRNNEncoder,
            "fastgrnn": _FastGRNNEncoder,
            "mut1": _MUT1Encoder,
            "mut2": _MUT2Encoder,
            "mut3": _MUT3Encoder,
            "ran": _RANEncoder,
            "scrn": _SCRNEncoder,
            "phased-lstm": _PhasedLSTMEncoder,
        }
        if base_s == "rhn":
            return _RHNEncoder(depth=int(rhn_depth))
        if base_s == "clockwork":
            return _ClockworkEncoder(periods=_normalize_clock_periods(clock_periods))
        if base_s == "qrnn":
            return _QRNNEncoder(kernel_size=int(qrnn_kernel_size))
        builder = simple_builders.get(base_s)
        if builder is not None:
            return builder()
        raise ValueError(f"Unknown rnnzoo base: {base_s!r}")

    class _BidirectionalWrapper(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.fwd = _make_base_encoder()
            self.bwd = _make_base_encoder()

        def forward(self, xb: Any) -> Any:
            out_f = self.fwd(xb)
            out_b = self.bwd(torch.flip(xb, dims=[1]))
            out_b = torch.flip(out_b, dims=[1])
            return torch.cat([out_f, out_b], dim=-1)

    class _RNNZooNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.variant = variant_s
            self.encoder = (
                _BidirectionalWrapper() if self.variant == "bidir" else _make_base_encoder()
            )
            self.ln = None
            if self.variant == "ln":
                self.ln = nn.LayerNorm(
                    (2 * hid if self.variant == "bidir" else hid),
                )
            self.attn_w = None
            self.attn_v = None
            if self.variant == "attn":
                enc_dim = 2 * hid if self.variant == "bidir" else hid
                self.attn_w = nn.Linear(enc_dim, int(attn_hid), bias=True)
                self.attn_v = nn.Linear(int(attn_hid), 1, bias=False)

            self.proj = None
            head_in = 2 * hid if self.variant == "bidir" else hid
            if self.variant == "proj":
                self.proj = nn.Linear(head_in, proj, bias=True)
                head_in = proj
            self.head = nn.Linear(int(head_in), h, bias=True)

        def _attention_pool(self, out_seq: Any) -> Any:
            assert self.attn_w is not None
            assert self.attn_v is not None
            # out_seq: (B,T,D)
            scores = self.attn_v(torch.tanh(self.attn_w(out_seq))).squeeze(-1)  # (B,T)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,T,1)
            return torch.sum(w * out_seq, dim=1)

        def forward(self, xb: Any) -> Any:
            xb = _ensure_device(xb)
            out_seq = self.encoder(xb)
            if self.variant == "ln":
                assert self.ln is not None
                out_seq = self.ln(out_seq)

            if self.variant == "attn":
                ctx = self._attention_pool(out_seq)
            else:
                ctx = out_seq[:, -1, :]

            if self.proj is not None:
                ctx = self.proj(ctx)
            return self.head(ctx)

    model = _RNNZooNet()

    cfg = _build_rnnzoo_train_config(
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        patience=patience,
        loss=loss,
        val_split=val_split,
        grad_clip_norm=grad_clip_norm,
        optimizer=optimizer,
        momentum=momentum,
        scheduler=scheduler,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        restore_best=restore_best,
    )
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

    yhat_t = _predict_rnnzoo_direct_model(
        torch,
        model,
        x_work,
        lag_count=lag_count,
        device=str(device),
    )
    return _maybe_denormalize_rnnzoo_forecast(
        yhat_t,
        normalize=bool(normalize),
        mean=mean,
        std=std,
    )
