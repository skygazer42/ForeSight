from __future__ import annotations

from typing import Any

import numpy as np

from .torch_nn import (
    TorchTrainConfig,
    _as_1d_float_array,
    _make_lagged_xy_multi,
    _normalize_series,
    _require_torch,
    _train_loop,
)

HORIZON_MIN_ERROR = "horizon must be >= 1"
LAGS_MIN_ERROR = "lags must be >= 1"
HIDDEN_SIZE_MIN_ERROR = "hidden_size must be >= 1"
NUM_LAYERS_MIN_ERROR = "num_layers must be >= 1"
DROPOUT_RANGE_ERROR = "dropout must be in [0,1)"


def torch_lmu_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    memory_dim: int = 32,
    num_layers: int = 1,
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
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    min_epochs: int = 1,
    amp: bool = False,
    amp_dtype: str = "auto",
    warmup_epochs: int = 0,
    min_lr: float = 0.0,
    grad_accum_steps: int = 1,
    monitor: str = "auto",
    monitor_mode: str = "min",
    min_delta: float = 0.0,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    scheduler_patience: int = 5,
    grad_clip_mode: str = "norm",
    grad_clip_value: float = 0.0,
    scheduler_plateau_factor: float = 0.1,
    scheduler_plateau_threshold: float = 1e-4,
    ema_decay: float = 0.0,
    ema_warmup_epochs: int = 0,
    swa_start_epoch: int = -1,
    lookahead_steps: int = 0,
    lookahead_alpha: float = 0.5,
    sam_rho: float = 0.0,
    sam_adaptive: bool = False,
    horizon_loss_decay: float = 1.0,
    input_dropout: float = 0.0,
    temporal_dropout: float = 0.0,
    grad_noise_std: float = 0.0,
    gc_mode: str = "off",
    agc_clip_factor: float = 0.0,
    agc_eps: float = 1e-3,
    checkpoint_dir: str = "",
    save_best_checkpoint: bool = False,
    save_last_checkpoint: bool = False,
    resume_checkpoint_path: str = "",
    resume_checkpoint_strict: bool = True,
) -> np.ndarray:
    """
    Torch LMU-style recurrent memory model (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    d = int(d_model)
    mem_dim = int(memory_dim)
    layers = int(num_layers)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lag_count <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if d <= 0:
        raise ValueError("d_model must be >= 1")
    if mem_dim <= 0:
        raise ValueError("memory_dim must be >= 1")
    if layers <= 0:
        raise ValueError(NUM_LAYERS_MIN_ERROR)
    if not (0.0 <= drop < 1.0):
        raise ValueError(DROPOUT_RANGE_ERROR)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _LMUBlock(nn.Module):
        def __init__(self, *, in_dim: int) -> None:
            super().__init__()
            self.in_proj = nn.Linear(int(in_dim), d)
            self.mem_in = nn.Linear(d, mem_dim)
            self.mix = nn.Linear(d + mem_dim, d)
            self.log_decay = nn.Parameter(torch.zeros((mem_dim,), dtype=torch.float32))
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any) -> Any:
            bsz = int(xb.shape[0])
            mem = xb.new_zeros((bsz, mem_dim))
            a = torch.exp(-F.softplus(self.log_decay)).reshape(1, mem_dim)
            outs: list[Any] = []
            for t in range(int(xb.shape[1])):
                u = torch.tanh(self.mem_in(self.in_proj(xb[:, t, :])))
                mem = a * mem + (1.0 - a) * u
                h_t = torch.tanh(self.mix(torch.cat([self.in_proj(xb[:, t, :]), mem], dim=-1)))
                outs.append(self.drop(h_t))
            return torch.stack(outs, dim=1)

    class _LMUDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [_LMUBlock(in_dim=1 if i == 0 else d) for i in range(layers)]
            )
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:
            z = xb
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            return self.head(z[:, -1, :])

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
        scheduler_restart_period=int(scheduler_restart_period),
        scheduler_restart_mult=int(scheduler_restart_mult),
        scheduler_pct_start=float(scheduler_pct_start),
        restore_best=bool(restore_best),
        min_epochs=int(min_epochs),
        amp=bool(amp),
        amp_dtype=str(amp_dtype),
        warmup_epochs=int(warmup_epochs),
        min_lr=float(min_lr),
        grad_accum_steps=int(grad_accum_steps),
        monitor=str(monitor),
        monitor_mode=str(monitor_mode),
        min_delta=float(min_delta),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        scheduler_patience=int(scheduler_patience),
        grad_clip_mode=str(grad_clip_mode),
        grad_clip_value=float(grad_clip_value),
        scheduler_plateau_factor=float(scheduler_plateau_factor),
        scheduler_plateau_threshold=float(scheduler_plateau_threshold),
        ema_decay=float(ema_decay),
        ema_warmup_epochs=int(ema_warmup_epochs),
        swa_start_epoch=int(swa_start_epoch),
        lookahead_steps=int(lookahead_steps),
        lookahead_alpha=float(lookahead_alpha),
        sam_rho=float(sam_rho),
        sam_adaptive=bool(sam_adaptive),
        horizon_loss_decay=float(horizon_loss_decay),
        input_dropout=float(input_dropout),
        temporal_dropout=float(temporal_dropout),
        grad_noise_std=float(grad_noise_std),
        gc_mode=str(gc_mode),
        agc_clip_factor=float(agc_clip_factor),
        agc_eps=float(agc_eps),
        checkpoint_dir=str(checkpoint_dir),
        save_best_checkpoint=bool(save_best_checkpoint),
        save_last_checkpoint=bool(save_last_checkpoint),
        resume_checkpoint_path=str(resume_checkpoint_path),
        resume_checkpoint_strict=bool(resume_checkpoint_strict),
    )
    model = _train_loop(_LMUDirect(), x_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_ltc_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    hidden_size: int = 64,
    num_layers: int = 1,
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
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    min_epochs: int = 1,
    amp: bool = False,
    amp_dtype: str = "auto",
    warmup_epochs: int = 0,
    min_lr: float = 0.0,
    grad_accum_steps: int = 1,
    monitor: str = "auto",
    monitor_mode: str = "min",
    min_delta: float = 0.0,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    scheduler_patience: int = 5,
    grad_clip_mode: str = "norm",
    grad_clip_value: float = 0.0,
    scheduler_plateau_factor: float = 0.1,
    scheduler_plateau_threshold: float = 1e-4,
    ema_decay: float = 0.0,
    ema_warmup_epochs: int = 0,
    swa_start_epoch: int = -1,
    lookahead_steps: int = 0,
    lookahead_alpha: float = 0.5,
    sam_rho: float = 0.0,
    sam_adaptive: bool = False,
    horizon_loss_decay: float = 1.0,
    input_dropout: float = 0.0,
    temporal_dropout: float = 0.0,
    grad_noise_std: float = 0.0,
    gc_mode: str = "off",
    agc_clip_factor: float = 0.0,
    agc_eps: float = 1e-3,
    checkpoint_dir: str = "",
    save_best_checkpoint: bool = False,
    save_last_checkpoint: bool = False,
    resume_checkpoint_path: str = "",
    resume_checkpoint_strict: bool = True,
) -> np.ndarray:
    """
    Torch LTC-style liquid time-constant recurrent model (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    hid = int(hidden_size)
    layers = int(num_layers)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lag_count <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if hid <= 0:
        raise ValueError(HIDDEN_SIZE_MIN_ERROR)
    if layers <= 0:
        raise ValueError(NUM_LAYERS_MIN_ERROR)
    if not (0.0 <= drop < 1.0):
        raise ValueError(DROPOUT_RANGE_ERROR)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _LTCBlock(nn.Module):
        def __init__(self, *, in_dim: int) -> None:
            super().__init__()
            self.in_proj = nn.Linear(int(in_dim), hid)
            self.rec_proj = nn.Linear(hid, hid, bias=False)
            self.tau_net = nn.Linear(int(in_dim) + hid, hid)
            self.drive_net = nn.Linear(int(in_dim) + hid, hid)
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any) -> Any:
            state = xb.new_zeros((int(xb.shape[0]), hid))
            outs: list[Any] = []
            for t in range(int(xb.shape[1])):
                inp = xb[:, t, :]
                fused = torch.cat([inp, state], dim=-1)
                tau = torch.sigmoid(self.tau_net(fused))
                drive = torch.tanh(self.in_proj(inp) + self.rec_proj(state) + self.drive_net(fused))
                state = (1.0 - tau) * state + tau * drive
                outs.append(self.drop(state))
            return torch.stack(outs, dim=1)

    class _LTCDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [_LTCBlock(in_dim=1 if i == 0 else hid) for i in range(layers)]
            )
            self.norm = nn.LayerNorm(hid)
            self.head = nn.Linear(hid, h)

        def forward(self, xb: Any) -> Any:
            z = xb
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            return self.head(z[:, -1, :])

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
        scheduler_restart_period=int(scheduler_restart_period),
        scheduler_restart_mult=int(scheduler_restart_mult),
        scheduler_pct_start=float(scheduler_pct_start),
        restore_best=bool(restore_best),
        min_epochs=int(min_epochs),
        amp=bool(amp),
        amp_dtype=str(amp_dtype),
        warmup_epochs=int(warmup_epochs),
        min_lr=float(min_lr),
        grad_accum_steps=int(grad_accum_steps),
        monitor=str(monitor),
        monitor_mode=str(monitor_mode),
        min_delta=float(min_delta),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        scheduler_patience=int(scheduler_patience),
        grad_clip_mode=str(grad_clip_mode),
        grad_clip_value=float(grad_clip_value),
        scheduler_plateau_factor=float(scheduler_plateau_factor),
        scheduler_plateau_threshold=float(scheduler_plateau_threshold),
        ema_decay=float(ema_decay),
        ema_warmup_epochs=int(ema_warmup_epochs),
        swa_start_epoch=int(swa_start_epoch),
        lookahead_steps=int(lookahead_steps),
        lookahead_alpha=float(lookahead_alpha),
        sam_rho=float(sam_rho),
        sam_adaptive=bool(sam_adaptive),
        horizon_loss_decay=float(horizon_loss_decay),
        input_dropout=float(input_dropout),
        temporal_dropout=float(temporal_dropout),
        grad_noise_std=float(grad_noise_std),
        gc_mode=str(gc_mode),
        agc_clip_factor=float(agc_clip_factor),
        agc_eps=float(agc_eps),
        checkpoint_dir=str(checkpoint_dir),
        save_best_checkpoint=bool(save_best_checkpoint),
        save_last_checkpoint=bool(save_last_checkpoint),
        resume_checkpoint_path=str(resume_checkpoint_path),
        resume_checkpoint_strict=bool(resume_checkpoint_strict),
    )
    model = _train_loop(_LTCDirect(), x_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_cfc_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    hidden_size: int = 64,
    num_layers: int = 1,
    backbone_hidden: int = 128,
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
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    min_epochs: int = 1,
    amp: bool = False,
    amp_dtype: str = "auto",
    warmup_epochs: int = 0,
    min_lr: float = 0.0,
    grad_accum_steps: int = 1,
    monitor: str = "auto",
    monitor_mode: str = "min",
    min_delta: float = 0.0,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    scheduler_patience: int = 5,
    grad_clip_mode: str = "norm",
    grad_clip_value: float = 0.0,
    scheduler_plateau_factor: float = 0.1,
    scheduler_plateau_threshold: float = 1e-4,
    ema_decay: float = 0.0,
    ema_warmup_epochs: int = 0,
    swa_start_epoch: int = -1,
    lookahead_steps: int = 0,
    lookahead_alpha: float = 0.5,
    sam_rho: float = 0.0,
    sam_adaptive: bool = False,
    horizon_loss_decay: float = 1.0,
    input_dropout: float = 0.0,
    temporal_dropout: float = 0.0,
    grad_noise_std: float = 0.0,
    gc_mode: str = "off",
    agc_clip_factor: float = 0.0,
    agc_eps: float = 1e-3,
    checkpoint_dir: str = "",
    save_best_checkpoint: bool = False,
    save_last_checkpoint: bool = False,
    resume_checkpoint_path: str = "",
    resume_checkpoint_strict: bool = True,
) -> np.ndarray:
    """
    Torch CfC-style closed-form continuous-time recurrent model (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    hid = int(hidden_size)
    layers = int(num_layers)
    bb = int(backbone_hidden)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lag_count <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if hid <= 0:
        raise ValueError(HIDDEN_SIZE_MIN_ERROR)
    if layers <= 0:
        raise ValueError(NUM_LAYERS_MIN_ERROR)
    if bb <= 0:
        raise ValueError("backbone_hidden must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError(DROPOUT_RANGE_ERROR)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _CfCBlock(nn.Module):
        def __init__(self, *, in_dim: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(int(in_dim) + hid, bb),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(bb, 2 * hid),
            )
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any) -> Any:
            state = xb.new_zeros((int(xb.shape[0]), hid))
            outs: list[Any] = []
            for t in range(int(xb.shape[1])):
                fused = torch.cat([xb[:, t, :], state], dim=-1)
                ff = self.backbone(fused)
                cand, gate = ff.chunk(2, dim=-1)
                cand = torch.tanh(cand)
                gate = torch.sigmoid(gate)
                state = gate * state + (1.0 - gate) * cand
                outs.append(self.drop(state))
            return torch.stack(outs, dim=1)

    class _CfCDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [_CfCBlock(in_dim=1 if i == 0 else hid) for i in range(layers)]
            )
            self.norm = nn.LayerNorm(hid)
            self.head = nn.Linear(hid, h)

        def forward(self, xb: Any) -> Any:
            z = xb
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            return self.head(z[:, -1, :])

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
        scheduler_restart_period=int(scheduler_restart_period),
        scheduler_restart_mult=int(scheduler_restart_mult),
        scheduler_pct_start=float(scheduler_pct_start),
        restore_best=bool(restore_best),
        min_epochs=int(min_epochs),
        amp=bool(amp),
        amp_dtype=str(amp_dtype),
        warmup_epochs=int(warmup_epochs),
        min_lr=float(min_lr),
        grad_accum_steps=int(grad_accum_steps),
        monitor=str(monitor),
        monitor_mode=str(monitor_mode),
        min_delta=float(min_delta),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        scheduler_patience=int(scheduler_patience),
        grad_clip_mode=str(grad_clip_mode),
        grad_clip_value=float(grad_clip_value),
        scheduler_plateau_factor=float(scheduler_plateau_factor),
        scheduler_plateau_threshold=float(scheduler_plateau_threshold),
        ema_decay=float(ema_decay),
        ema_warmup_epochs=int(ema_warmup_epochs),
        swa_start_epoch=int(swa_start_epoch),
        lookahead_steps=int(lookahead_steps),
        lookahead_alpha=float(lookahead_alpha),
        sam_rho=float(sam_rho),
        sam_adaptive=bool(sam_adaptive),
        horizon_loss_decay=float(horizon_loss_decay),
        input_dropout=float(input_dropout),
        temporal_dropout=float(temporal_dropout),
        grad_noise_std=float(grad_noise_std),
        gc_mode=str(gc_mode),
        agc_clip_factor=float(agc_clip_factor),
        agc_eps=float(agc_eps),
        checkpoint_dir=str(checkpoint_dir),
        save_best_checkpoint=bool(save_best_checkpoint),
        save_last_checkpoint=bool(save_last_checkpoint),
        resume_checkpoint_path=str(resume_checkpoint_path),
        resume_checkpoint_strict=bool(resume_checkpoint_strict),
    )
    model = _train_loop(_CfCDirect(), x_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_xlstm_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    hidden_size: int = 64,
    num_layers: int = 1,
    proj_factor: int = 2,
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
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    min_epochs: int = 1,
    amp: bool = False,
    amp_dtype: str = "auto",
    warmup_epochs: int = 0,
    min_lr: float = 0.0,
    grad_accum_steps: int = 1,
    monitor: str = "auto",
    monitor_mode: str = "min",
    min_delta: float = 0.0,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    scheduler_patience: int = 5,
    grad_clip_mode: str = "norm",
    grad_clip_value: float = 0.0,
    scheduler_plateau_factor: float = 0.1,
    scheduler_plateau_threshold: float = 1e-4,
    ema_decay: float = 0.0,
    ema_warmup_epochs: int = 0,
    swa_start_epoch: int = -1,
    lookahead_steps: int = 0,
    lookahead_alpha: float = 0.5,
    sam_rho: float = 0.0,
    sam_adaptive: bool = False,
    horizon_loss_decay: float = 1.0,
    input_dropout: float = 0.0,
    temporal_dropout: float = 0.0,
    grad_noise_std: float = 0.0,
    gc_mode: str = "off",
    agc_clip_factor: float = 0.0,
    agc_eps: float = 1e-3,
    checkpoint_dir: str = "",
    save_best_checkpoint: bool = False,
    save_last_checkpoint: bool = False,
    resume_checkpoint_path: str = "",
    resume_checkpoint_strict: bool = True,
) -> np.ndarray:
    """
    Torch xLSTM-style expanded-gate recurrent model (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    hid = int(hidden_size)
    layers = int(num_layers)
    expand = int(proj_factor)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lag_count <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if hid <= 0:
        raise ValueError(HIDDEN_SIZE_MIN_ERROR)
    if layers <= 0:
        raise ValueError(NUM_LAYERS_MIN_ERROR)
    if expand <= 0:
        raise ValueError("proj_factor must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError(DROPOUT_RANGE_ERROR)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)
    inner = hid * expand

    class _xLSTMBlock(nn.Module):
        def __init__(self, *, in_dim: int) -> None:
            super().__init__()
            self.in_proj = nn.Linear(int(in_dim), inner)
            self.h_proj = nn.Linear(hid, inner, bias=False)
            self.out_proj = nn.Linear(inner, hid)
            self.forget_gate = nn.Linear(int(in_dim) + hid, hid)
            self.input_gate = nn.Linear(int(in_dim) + hid, hid)
            self.output_gate = nn.Linear(int(in_dim) + hid, hid)
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any) -> Any:
            h_state = xb.new_zeros((int(xb.shape[0]), hid))
            c_state = xb.new_zeros((int(xb.shape[0]), hid))
            outs: list[Any] = []
            for t in range(int(xb.shape[1])):
                inp = xb[:, t, :]
                fused = torch.cat([inp, h_state], dim=-1)
                f = torch.sigmoid(self.forget_gate(fused))
                i = torch.sigmoid(self.input_gate(fused))
                o = torch.sigmoid(self.output_gate(fused))
                cand = torch.tanh(self.out_proj(self.in_proj(inp) + self.h_proj(h_state)))
                c_state = f * c_state + i * cand
                h_state = o * torch.tanh(c_state)
                outs.append(self.drop(h_state))
            return torch.stack(outs, dim=1)

    class _xLSTMDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [_xLSTMBlock(in_dim=1 if i == 0 else hid) for i in range(layers)]
            )
            self.norm = nn.LayerNorm(hid)
            self.head = nn.Linear(hid, h)

        def forward(self, xb: Any) -> Any:
            z = xb
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            return self.head(z[:, -1, :])

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
        scheduler_restart_period=int(scheduler_restart_period),
        scheduler_restart_mult=int(scheduler_restart_mult),
        scheduler_pct_start=float(scheduler_pct_start),
        restore_best=bool(restore_best),
        min_epochs=int(min_epochs),
        amp=bool(amp),
        amp_dtype=str(amp_dtype),
        warmup_epochs=int(warmup_epochs),
        min_lr=float(min_lr),
        grad_accum_steps=int(grad_accum_steps),
        monitor=str(monitor),
        monitor_mode=str(monitor_mode),
        min_delta=float(min_delta),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        scheduler_patience=int(scheduler_patience),
        grad_clip_mode=str(grad_clip_mode),
        grad_clip_value=float(grad_clip_value),
        scheduler_plateau_factor=float(scheduler_plateau_factor),
        scheduler_plateau_threshold=float(scheduler_plateau_threshold),
        ema_decay=float(ema_decay),
        ema_warmup_epochs=int(ema_warmup_epochs),
        swa_start_epoch=int(swa_start_epoch),
        lookahead_steps=int(lookahead_steps),
        lookahead_alpha=float(lookahead_alpha),
        sam_rho=float(sam_rho),
        sam_adaptive=bool(sam_adaptive),
        horizon_loss_decay=float(horizon_loss_decay),
        input_dropout=float(input_dropout),
        temporal_dropout=float(temporal_dropout),
        grad_noise_std=float(grad_noise_std),
        gc_mode=str(gc_mode),
        agc_clip_factor=float(agc_clip_factor),
        agc_eps=float(agc_eps),
        checkpoint_dir=str(checkpoint_dir),
        save_best_checkpoint=bool(save_best_checkpoint),
        save_last_checkpoint=bool(save_last_checkpoint),
        resume_checkpoint_path=str(resume_checkpoint_path),
        resume_checkpoint_strict=bool(resume_checkpoint_strict),
    )
    model = _train_loop(_xLSTMDirect(), x_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_griffin_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    hidden_size: int = 64,
    num_layers: int = 1,
    conv_kernel: int = 3,
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
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    min_epochs: int = 1,
    amp: bool = False,
    amp_dtype: str = "auto",
    warmup_epochs: int = 0,
    min_lr: float = 0.0,
    grad_accum_steps: int = 1,
    monitor: str = "auto",
    monitor_mode: str = "min",
    min_delta: float = 0.0,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    scheduler_patience: int = 5,
    grad_clip_mode: str = "norm",
    grad_clip_value: float = 0.0,
    scheduler_plateau_factor: float = 0.1,
    scheduler_plateau_threshold: float = 1e-4,
    ema_decay: float = 0.0,
    ema_warmup_epochs: int = 0,
    swa_start_epoch: int = -1,
    lookahead_steps: int = 0,
    lookahead_alpha: float = 0.5,
    sam_rho: float = 0.0,
    sam_adaptive: bool = False,
    horizon_loss_decay: float = 1.0,
    input_dropout: float = 0.0,
    temporal_dropout: float = 0.0,
    grad_noise_std: float = 0.0,
    gc_mode: str = "off",
    agc_clip_factor: float = 0.0,
    agc_eps: float = 1e-3,
    checkpoint_dir: str = "",
    save_best_checkpoint: bool = False,
    save_last_checkpoint: bool = False,
    resume_checkpoint_path: str = "",
    resume_checkpoint_strict: bool = True,
) -> np.ndarray:
    """
    Torch Griffin-style recurrent hybrid model (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    hid = int(hidden_size)
    layers = int(num_layers)
    k = int(conv_kernel)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lag_count <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if hid <= 0:
        raise ValueError(HIDDEN_SIZE_MIN_ERROR)
    if layers <= 0:
        raise ValueError(NUM_LAYERS_MIN_ERROR)
    if k <= 0:
        raise ValueError("conv_kernel must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError(DROPOUT_RANGE_ERROR)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _GriffinBlock(nn.Module):
        def __init__(self, *, in_dim: int) -> None:
            super().__init__()
            self.in_proj = nn.Linear(int(in_dim), hid)
            self.rec_proj = nn.Linear(hid, hid, bias=False)
            self.dwconv = nn.Conv1d(hid, hid, kernel_size=int(k), groups=hid)
            self.gate = nn.Linear(hid, hid)
            self.out_proj = nn.Linear(hid, hid)
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any) -> Any:
            z = self.in_proj(xb)
            z_ch = z.transpose(1, 2)
            z_pad = F.pad(z_ch, (int(k) - 1, 0))
            z = self.dwconv(z_pad).transpose(1, 2)
            state = xb.new_zeros((int(xb.shape[0]), hid))
            outs: list[Any] = []
            for t in range(int(xb.shape[1])):
                cand = torch.tanh(z[:, t, :] + self.rec_proj(state))
                g = torch.sigmoid(self.gate(cand))
                state = g * state + (1.0 - g) * cand
                outs.append(self.drop(self.out_proj(state)))
            return torch.stack(outs, dim=1)

    class _GriffinDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [_GriffinBlock(in_dim=1 if i == 0 else hid) for i in range(layers)]
            )
            self.norm = nn.LayerNorm(hid)
            self.head = nn.Linear(hid, h)

        def forward(self, xb: Any) -> Any:
            z = xb
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            return self.head(z[:, -1, :])

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
        scheduler_restart_period=int(scheduler_restart_period),
        scheduler_restart_mult=int(scheduler_restart_mult),
        scheduler_pct_start=float(scheduler_pct_start),
        restore_best=bool(restore_best),
        min_epochs=int(min_epochs),
        amp=bool(amp),
        amp_dtype=str(amp_dtype),
        warmup_epochs=int(warmup_epochs),
        min_lr=float(min_lr),
        grad_accum_steps=int(grad_accum_steps),
        monitor=str(monitor),
        monitor_mode=str(monitor_mode),
        min_delta=float(min_delta),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        scheduler_patience=int(scheduler_patience),
        grad_clip_mode=str(grad_clip_mode),
        grad_clip_value=float(grad_clip_value),
        scheduler_plateau_factor=float(scheduler_plateau_factor),
        scheduler_plateau_threshold=float(scheduler_plateau_threshold),
        ema_decay=float(ema_decay),
        ema_warmup_epochs=int(ema_warmup_epochs),
        swa_start_epoch=int(swa_start_epoch),
        lookahead_steps=int(lookahead_steps),
        lookahead_alpha=float(lookahead_alpha),
        sam_rho=float(sam_rho),
        sam_adaptive=bool(sam_adaptive),
        horizon_loss_decay=float(horizon_loss_decay),
        input_dropout=float(input_dropout),
        temporal_dropout=float(temporal_dropout),
        grad_noise_std=float(grad_noise_std),
        gc_mode=str(gc_mode),
        agc_clip_factor=float(agc_clip_factor),
        agc_eps=float(agc_eps),
        checkpoint_dir=str(checkpoint_dir),
        save_best_checkpoint=bool(save_best_checkpoint),
        save_last_checkpoint=bool(save_last_checkpoint),
        resume_checkpoint_path=str(resume_checkpoint_path),
        resume_checkpoint_strict=bool(resume_checkpoint_strict),
    )
    model = _train_loop(_GriffinDirect(), x_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_hawk_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    hidden_size: int = 64,
    num_layers: int = 1,
    expansion_factor: int = 2,
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
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    min_epochs: int = 1,
    amp: bool = False,
    amp_dtype: str = "auto",
    warmup_epochs: int = 0,
    min_lr: float = 0.0,
    grad_accum_steps: int = 1,
    monitor: str = "auto",
    monitor_mode: str = "min",
    min_delta: float = 0.0,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    scheduler_patience: int = 5,
    grad_clip_mode: str = "norm",
    grad_clip_value: float = 0.0,
    scheduler_plateau_factor: float = 0.1,
    scheduler_plateau_threshold: float = 1e-4,
    ema_decay: float = 0.0,
    ema_warmup_epochs: int = 0,
    swa_start_epoch: int = -1,
    lookahead_steps: int = 0,
    lookahead_alpha: float = 0.5,
    sam_rho: float = 0.0,
    sam_adaptive: bool = False,
    horizon_loss_decay: float = 1.0,
    input_dropout: float = 0.0,
    temporal_dropout: float = 0.0,
    grad_noise_std: float = 0.0,
    gc_mode: str = "off",
    agc_clip_factor: float = 0.0,
    agc_eps: float = 1e-3,
    checkpoint_dir: str = "",
    save_best_checkpoint: bool = False,
    save_last_checkpoint: bool = False,
    resume_checkpoint_path: str = "",
    resume_checkpoint_strict: bool = True,
) -> np.ndarray:
    """
    Torch Hawk-style gated recurrent mixer (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    hid = int(hidden_size)
    layers = int(num_layers)
    exp = int(expansion_factor)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lag_count <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if hid <= 0:
        raise ValueError(HIDDEN_SIZE_MIN_ERROR)
    if layers <= 0:
        raise ValueError(NUM_LAYERS_MIN_ERROR)
    if exp <= 0:
        raise ValueError("expansion_factor must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError(DROPOUT_RANGE_ERROR)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)
    inner = hid * exp

    class _HawkBlock(nn.Module):
        def __init__(self, *, in_dim: int) -> None:
            super().__init__()
            self.in_proj = nn.Linear(int(in_dim), inner)
            self.state_proj = nn.Linear(hid, inner, bias=False)
            self.mix = nn.Linear(inner, hid)
            self.gate = nn.Linear(int(in_dim) + hid, hid)
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any) -> Any:
            state = xb.new_zeros((int(xb.shape[0]), hid))
            outs: list[Any] = []
            for t in range(int(xb.shape[1])):
                inp = xb[:, t, :]
                fused = torch.cat([inp, state], dim=-1)
                gate = torch.sigmoid(self.gate(fused))
                cand = torch.tanh(self.mix(self.in_proj(inp) + self.state_proj(state)))
                state = gate * state + (1.0 - gate) * cand
                outs.append(self.drop(state))
            return torch.stack(outs, dim=1)

    class _HawkDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [_HawkBlock(in_dim=1 if i == 0 else hid) for i in range(layers)]
            )
            self.norm = nn.LayerNorm(hid)
            self.head = nn.Linear(hid, h)

        def forward(self, xb: Any) -> Any:
            z = xb
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            return self.head(z[:, -1, :])

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
        scheduler_restart_period=int(scheduler_restart_period),
        scheduler_restart_mult=int(scheduler_restart_mult),
        scheduler_pct_start=float(scheduler_pct_start),
        restore_best=bool(restore_best),
        min_epochs=int(min_epochs),
        amp=bool(amp),
        amp_dtype=str(amp_dtype),
        warmup_epochs=int(warmup_epochs),
        min_lr=float(min_lr),
        grad_accum_steps=int(grad_accum_steps),
        monitor=str(monitor),
        monitor_mode=str(monitor_mode),
        min_delta=float(min_delta),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        scheduler_patience=int(scheduler_patience),
        grad_clip_mode=str(grad_clip_mode),
        grad_clip_value=float(grad_clip_value),
        scheduler_plateau_factor=float(scheduler_plateau_factor),
        scheduler_plateau_threshold=float(scheduler_plateau_threshold),
        ema_decay=float(ema_decay),
        ema_warmup_epochs=int(ema_warmup_epochs),
        swa_start_epoch=int(swa_start_epoch),
        lookahead_steps=int(lookahead_steps),
        lookahead_alpha=float(lookahead_alpha),
        sam_rho=float(sam_rho),
        sam_adaptive=bool(sam_adaptive),
        horizon_loss_decay=float(horizon_loss_decay),
        input_dropout=float(input_dropout),
        temporal_dropout=float(temporal_dropout),
        grad_noise_std=float(grad_noise_std),
        gc_mode=str(gc_mode),
        agc_clip_factor=float(agc_clip_factor),
        agc_eps=float(agc_eps),
        checkpoint_dir=str(checkpoint_dir),
        save_best_checkpoint=bool(save_best_checkpoint),
        save_last_checkpoint=bool(save_last_checkpoint),
        resume_checkpoint_path=str(resume_checkpoint_path),
        resume_checkpoint_strict=bool(resume_checkpoint_strict),
    )
    model = _train_loop(_HawkDirect(), x_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)
