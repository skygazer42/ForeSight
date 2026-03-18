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

DMODEL_MIN_ERROR = "d_model must be >= 1"
NUM_LAYERS_MIN_ERROR = "num_layers must be >= 1"
DROPOUT_RANGE_ERROR = "dropout must be in [0,1)"


def _fit_local_sequence_model(
    train: Any,
    horizon: int,
    *,
    lags: int,
    normalize: bool,
    device: str,
    cfg: TorchTrainConfig,
    build_model: Any,
) -> np.ndarray:
    torch = _require_torch()

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)
    model = _train_loop(build_model(lag_count, h), x_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_s4d_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
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
    Torch S4D-style diagonal state-space model (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    d = int(d_model)
    layers = int(num_layers)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(DMODEL_MIN_ERROR)
    if layers <= 0:
        raise ValueError(NUM_LAYERS_MIN_ERROR)
    if not (0.0 <= drop < 1.0):
        raise ValueError(DROPOUT_RANGE_ERROR)

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

    def _build_model(lag_count: int, h: int) -> Any:
        class _S4DBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(d)
                self.u_proj = nn.Linear(d, d)
                self.gate = nn.Linear(d, d)
                self.log_decay = nn.Parameter(torch.zeros((d,), dtype=torch.float32))
                self.out_proj = nn.Linear(d, d)
                self.drop = nn.Dropout(p=drop)

            def forward(self, xb: Any) -> Any:
                z = self.norm(xb)
                u = self.u_proj(z)
                g = torch.sigmoid(self.gate(z))
                a = torch.exp(-F.softplus(self.log_decay)).reshape(1, 1, d)

                state = torch.zeros((int(u.shape[0]), d), device=u.device, dtype=u.dtype)
                outs: list[Any] = []
                for t in range(int(u.shape[1])):
                    state = a[:, 0, :] * state + (1.0 - a[:, 0, :]) * u[:, t, :]
                    y_t = self.out_proj(state)
                    outs.append(g[:, t, :] * y_t + (1.0 - g[:, t, :]) * u[:, t, :])
                y_seq = torch.stack(outs, dim=1)
                return xb + self.drop(y_seq)

        class _S4DDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
                self.layers = nn.ModuleList([_S4DBlock() for _ in range(layers)])
                self.norm = nn.LayerNorm(d)
                self.head = nn.Linear(d, h)

            def forward(self, xb: Any) -> Any:
                z = self.in_proj(xb) + self.pos
                for layer in self.layers:
                    z = layer(z)
                z = self.norm(z)
                return self.head(z[:, -1, :])

        return _S4DDirect()

    return _fit_local_sequence_model(
        train,
        horizon,
        lags=int(lags),
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
        build_model=_build_model,
    )


def torch_mamba2_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
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
    Torch Mamba-2-style selective state-space refinement (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    d = int(d_model)
    layers = int(num_layers)
    k = int(conv_kernel)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(DMODEL_MIN_ERROR)
    if layers <= 0:
        raise ValueError(NUM_LAYERS_MIN_ERROR)
    if k <= 0:
        raise ValueError("conv_kernel must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError(DROPOUT_RANGE_ERROR)

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

    def _build_model(lag_count: int, h: int) -> Any:
        class _Mamba2Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(d)
                self.in_proj = nn.Linear(d, 2 * d)
                self.delta_proj = nn.Linear(d, d)
                self.log_A = nn.Parameter(torch.zeros((d,), dtype=torch.float32))
                self.dwconv = nn.Conv1d(d, d, kernel_size=int(k), groups=d)
                self.skip_gate = nn.Linear(d, d)
                self.out_proj = nn.Linear(d, d)
                self.drop = nn.Dropout(p=drop)

            def forward(self, xb: Any) -> Any:
                z = self.norm(xb)
                uv = self.in_proj(z)
                u, v = uv.chunk(2, dim=-1)
                u_ch = u.transpose(1, 2)
                u_pad = F.pad(u_ch, (int(k) - 1, 0))
                u = self.dwconv(u_pad).transpose(1, 2)
                delta = F.softplus(self.delta_proj(z))
                a = torch.exp(-delta * F.softplus(self.log_A).reshape(1, 1, -1))
                gate = torch.sigmoid(self.skip_gate(z))

                state = torch.zeros((int(u.shape[0]), d), device=u.device, dtype=u.dtype)
                outs: list[Any] = []
                for t in range(int(u.shape[1])):
                    state = a[:, t, :] * state + (1.0 - a[:, t, :]) * (u[:, t, :] + v[:, t, :])
                    y_t = self.out_proj(state)
                    outs.append(gate[:, t, :] * y_t + (1.0 - gate[:, t, :]) * u[:, t, :])
                return xb + self.drop(torch.stack(outs, dim=1))

        class _Mamba2Direct(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(1, d)
                self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
                self.layers = nn.ModuleList([_Mamba2Block() for _ in range(layers)])
                self.norm = nn.LayerNorm(d)
                self.head = nn.Linear(d, h)

            def forward(self, xb: Any) -> Any:
                z = self.embed(xb) + self.pos
                for layer in self.layers:
                    z = layer(z)
                z = self.norm(z)
                return self.head(z[:, -1, :])

        return _Mamba2Direct()

    return _fit_local_sequence_model(
        train,
        horizon,
        lags=int(lags),
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
        build_model=_build_model,
    )


def torch_s4_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    state_dim: int = 32,
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
    Torch S4-style structured state-space model (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    d = int(d_model)
    layers = int(num_layers)
    sdim = int(state_dim)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(DMODEL_MIN_ERROR)
    if layers <= 0:
        raise ValueError(NUM_LAYERS_MIN_ERROR)
    if sdim <= 0:
        raise ValueError("state_dim must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError(DROPOUT_RANGE_ERROR)

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

    def _build_model(lag_count: int, h: int) -> Any:
        class _S4Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(d)
                self.in_proj = nn.Linear(d, sdim)
                self.out_proj = nn.Linear(sdim, d)
                self.skip = nn.Linear(d, d)
                self.log_decay = nn.Parameter(torch.zeros((sdim,), dtype=torch.float32))
                self.mix_in = nn.Parameter(torch.randn(sdim, d, dtype=torch.float32) * 0.02)
                self.mix_state = nn.Parameter(torch.randn(sdim, sdim, dtype=torch.float32) * 0.02)
                self.drop = nn.Dropout(p=drop)

            def forward(self, xb: Any) -> Any:
                z = self.norm(xb)
                u = self.in_proj(z)
                a = torch.exp(-F.softplus(self.log_decay)).reshape(1, sdim)
                state = torch.zeros((int(u.shape[0]), sdim), device=u.device, dtype=u.dtype)
                outs: list[Any] = []
                for t in range(int(u.shape[1])):
                    u_t = u[:, t, :] + z[:, t, :] @ self.mix_in.T
                    state = a * state + (1.0 - a) * u_t + torch.tanh(state @ self.mix_state.T)
                    y_t = self.out_proj(state)
                    outs.append(y_t)
                y_seq = torch.stack(outs, dim=1)
                return xb + self.drop(y_seq) + 0.1 * self.skip(z)

        class _S4Direct(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(1, d)
                self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
                self.layers = nn.ModuleList([_S4Block() for _ in range(layers)])
                self.norm = nn.LayerNorm(d)
                self.head = nn.Linear(d, h)

            def forward(self, xb: Any) -> Any:
                z = self.embed(xb) + self.pos
                for layer in self.layers:
                    z = layer(z)
                z = self.norm(z)
                return self.head(z[:, -1, :])

        return _S4Direct()

    return _fit_local_sequence_model(
        train,
        horizon,
        lags=int(lags),
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
        build_model=_build_model,
    )


def torch_s5_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    state_dim: int = 32,
    heads: int = 2,
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
    Torch S5-style multi-state-space model (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    d = int(d_model)
    layers = int(num_layers)
    sdim = int(state_dim)
    n_heads = int(heads)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(DMODEL_MIN_ERROR)
    if layers <= 0:
        raise ValueError(NUM_LAYERS_MIN_ERROR)
    if sdim <= 0:
        raise ValueError("state_dim must be >= 1")
    if n_heads <= 0:
        raise ValueError("heads must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError(DROPOUT_RANGE_ERROR)

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

    def _build_model(lag_count: int, h: int) -> Any:
        head_dim = max(1, d // n_heads)

        class _S5Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(d)
                self.in_proj = nn.Linear(d, n_heads * sdim)
                self.out_proj = nn.Linear(n_heads * sdim, d)
                self.log_decay = nn.Parameter(torch.zeros((n_heads, sdim), dtype=torch.float32))
                self.mix = nn.Parameter(
                    torch.randn(n_heads, sdim, head_dim, dtype=torch.float32) * 0.02
                )
                self.gate = nn.Linear(d, d)
                self.drop = nn.Dropout(p=drop)

            def forward(self, xb: Any) -> Any:
                z = self.norm(xb)
                u = self.in_proj(z).reshape(int(z.shape[0]), int(z.shape[1]), n_heads, sdim)
                a = torch.exp(-F.softplus(self.log_decay)).reshape(1, n_heads, sdim)
                state = torch.zeros(
                    (int(z.shape[0]), n_heads, sdim), device=z.device, dtype=z.dtype
                )
                outs: list[Any] = []
                for t in range(int(z.shape[1])):
                    u_t = u[:, t, :, :]
                    state = a * state + (1.0 - a) * u_t
                    y_heads = torch.einsum("bhs,hsk->bhk", state, self.mix)
                    y_flat = y_heads.reshape(int(z.shape[0]), -1)
                    if int(y_flat.shape[1]) < d:
                        pad = y_flat.new_zeros((int(z.shape[0]), d - int(y_flat.shape[1])))
                        y_flat = torch.cat([y_flat, pad], dim=1)
                    elif int(y_flat.shape[1]) > d:
                        y_flat = y_flat[:, :d]
                    y_t = self.out_proj(state.reshape(int(z.shape[0]), -1))
                    outs.append(torch.sigmoid(self.gate(z[:, t, :])) * y_t + 0.1 * y_flat)
                return xb + self.drop(torch.stack(outs, dim=1))

        class _S5Direct(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(1, d)
                self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
                self.layers = nn.ModuleList([_S5Block() for _ in range(layers)])
                self.norm = nn.LayerNorm(d)
                self.head = nn.Linear(d, h)

            def forward(self, xb: Any) -> Any:
                z = self.embed(xb) + self.pos
                for layer in self.layers:
                    z = layer(z)
                z = self.norm(z)
                return self.head(z[:, -1, :])

        return _S5Direct()

    return _fit_local_sequence_model(
        train,
        horizon,
        lags=int(lags),
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
        build_model=_build_model,
    )
