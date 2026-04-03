from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..optional_deps import missing_dependency_message
from .torch_nn import TorchTrainConfig, _normalize_series, _require_torch, _train_loop

HORIZON_MIN_ERROR = "horizon must be >= 1"
LAGS_MIN_ERROR = "lags must be >= 1"
D_MODEL_MIN_ERROR = "d_model must be >= 1"
NUM_BLOCKS_MIN_ERROR = "num_blocks must be >= 1"
DROPOUT_RANGE_ERROR = "dropout must be in [0,1)"


def _as_2d_float_array(train: Any) -> np.ndarray:
    if isinstance(train, pd.DataFrame):
        arr = train.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(train, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D multivariate series, got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 target columns for multivariate forecasting, got {arr.shape[1]}"
        )
    return arr


def _validated_multivariate_horizon_lags(
    *,
    horizon: int,
    lags: int,
) -> tuple[int, int]:
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lag_count <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    return h, lag_count


def _make_lagged_xy_multivariate(
    x: np.ndarray, *, lags: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    c = int(x.shape[1])
    h, lag_count = _validated_multivariate_horizon_lags(horizon=horizon, lags=lags)
    if n < lag_count + h:
        raise ValueError(f"Need >= lags+horizon rows (lags={lag_count}, horizon={h}), got {n}")

    rows = n - lag_count - h + 1
    X = np.empty((rows, lag_count, c), dtype=float)
    Y = np.empty((rows, h, c), dtype=float)
    for i in range(rows):
        t = i + lag_count
        X[i] = x[t - lag_count : t, :]
        Y[i] = x[t : t + h, :]
    return X, Y


def _latest_multivariate_window(x: np.ndarray, *, lag_count: int) -> np.ndarray:
    n_nodes = int(x.shape[1])
    return x[-int(lag_count) :, :].astype(float, copy=False).reshape(1, int(lag_count), n_nodes)


def _validated_torch_multivariate_model_dims(
    *,
    d_model: int,
    num_blocks: int,
    dropout: float,
) -> tuple[int, int, float]:
    d = int(d_model)
    blocks = int(num_blocks)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(D_MODEL_MIN_ERROR)
    if blocks <= 0:
        raise ValueError(NUM_BLOCKS_MIN_ERROR)
    if not (0.0 <= drop < 1.0):
        raise ValueError(DROPOUT_RANGE_ERROR)
    return d, blocks, drop


def _validated_graph_kernel_size(kernel_size: int) -> int:
    k = int(kernel_size)
    if k <= 0:
        raise ValueError("kernel_size must be >= 1")
    return k


def var_forecast(
    train: Any,
    horizon: int,
    *,
    maxlags: int = 1,
    trend: str = "c",
    ic: str | None = None,
) -> np.ndarray:
    """
    Vector autoregression forecast via statsmodels VAR (optional dependency).

    The input contract is a 2D matrix with shape (n_obs, n_targets), or a pandas
    DataFrame whose columns correspond to target series.
    """
    try:
        from statsmodels.tsa.api import VAR  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(missing_dependency_message("stats", subject="var_forecast")) from e

    x = _as_2d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)

    maxlags_int = int(maxlags)
    if maxlags_int <= 0:
        raise ValueError("maxlags must be >= 1")
    if x.shape[0] <= maxlags_int:
        raise ValueError(
            f"var_forecast requires more rows than maxlags; got n_obs={x.shape[0]}, maxlags={maxlags_int}"
        )

    ic_final = None if ic is None or str(ic).strip().lower() in {"", "none", "null"} else str(ic)

    model = VAR(x)
    res = model.fit(maxlags=maxlags_int, ic=ic_final, trend=str(trend))
    if int(res.k_ar) <= 0:
        raise ValueError("VAR selected zero autoregressive lags; increase maxlags or disable ic")

    fc = res.forecast(x[-int(res.k_ar) :], steps=int(horizon))
    out = np.asarray(fc, dtype=float)
    if out.shape != (int(horizon), x.shape[1]):
        raise ValueError(
            f"VAR forecaster must return shape ({int(horizon)}, {x.shape[1]}), got {out.shape}"
        )
    return out


def torch_stid_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 24,
    d_model: int = 64,
    num_blocks: int = 2,
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
    Torch STID-style multivariate baseline (lite) on a wide target matrix.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_2d_float_array(train)
    h, lag_count = _validated_multivariate_horizon_lags(horizon=horizon, lags=lags)
    d, blocks, drop = _validated_torch_multivariate_model_dims(
        d_model=d_model,
        num_blocks=num_blocks,
        dropout=dropout,
    )

    x_work, mean, std, X, Y, c = _prepare_torch_multivariate_training_data(
        x,
        lags=lag_count,
        horizon=h,
        normalize=bool(normalize),
    )

    class _STIDBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(2 * d, d),
            )

        def forward(self, xb: Any) -> Any:
            return xb + self.ffn(self.norm(xb))

    class _STIDDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_emb = nn.Parameter(torch.zeros((c, d), dtype=torch.float32))
            self.history_proj = nn.Linear(lag_count, d)
            self.blocks = nn.ModuleList([_STIDBlock() for _ in range(blocks)])
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # (B, T, C)
            x_nodes = xb.transpose(1, 2)  # (B, C, T)
            z = self.history_proj(x_nodes) + self.node_emb.unsqueeze(0)
            for blk in self.blocks:
                z = blk(z)
            yhat = self.head(z).transpose(1, 2)  # (B, h, C)
            return yhat

    cfg = _build_torch_multivariate_train_config(
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
        scheduler_restart_period=scheduler_restart_period,
        scheduler_restart_mult=scheduler_restart_mult,
        scheduler_pct_start=scheduler_pct_start,
        restore_best=restore_best,
        min_epochs=min_epochs,
        amp=amp,
        amp_dtype=amp_dtype,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        grad_accum_steps=grad_accum_steps,
        monitor=monitor,
        monitor_mode=monitor_mode,
        min_delta=min_delta,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        scheduler_patience=scheduler_patience,
        grad_clip_mode=grad_clip_mode,
        grad_clip_value=grad_clip_value,
        scheduler_plateau_factor=scheduler_plateau_factor,
        scheduler_plateau_threshold=scheduler_plateau_threshold,
        ema_decay=ema_decay,
        ema_warmup_epochs=ema_warmup_epochs,
        swa_start_epoch=swa_start_epoch,
        lookahead_steps=lookahead_steps,
        lookahead_alpha=lookahead_alpha,
        sam_rho=sam_rho,
        sam_adaptive=sam_adaptive,
        horizon_loss_decay=horizon_loss_decay,
        input_dropout=input_dropout,
        temporal_dropout=temporal_dropout,
        grad_noise_std=grad_noise_std,
        gc_mode=gc_mode,
        agc_clip_factor=agc_clip_factor,
        agc_eps=agc_eps,
        checkpoint_dir=checkpoint_dir,
        save_best_checkpoint=save_best_checkpoint,
        save_last_checkpoint=save_last_checkpoint,
        resume_checkpoint_path=resume_checkpoint_path,
        resume_checkpoint_strict=resume_checkpoint_strict,
    )
    model = _train_loop(_STIDDirect(), X, Y, cfg=cfg, device=str(device))

    return _finalize_torch_multivariate_forecast(
        model,
        torch_mod=torch,
        x_work=x_work,
        lag_count=lag_count,
        horizon=h,
        n_nodes=c,
        device=str(device),
        mean=mean,
        std=std,
        normalize=bool(normalize),
    )


def _row_normalize_adj(adj: np.ndarray) -> np.ndarray:
    a = np.asarray(adj, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adj must be a square 2D matrix, got shape {a.shape}")
    denom = a.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return a / denom


def _load_adj_matrix_from_path(adj_path: str) -> np.ndarray | None:
    if not str(adj_path).strip():
        return None

    path = Path(str(adj_path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"adj_path not found: {path}")
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=float)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path).to_numpy(dtype=float, copy=False)
    raise ValueError("adj_path must end with .npy or .csv/.txt")


def _corr_topk_adj_matrix(x_work: np.ndarray, *, n_nodes: int, top_k: int) -> np.ndarray:
    if x_work.ndim != 2 or int(x_work.shape[1]) != int(n_nodes):
        raise ValueError("corr adjacency requires x_work with shape (T, n_nodes)")

    corr = np.corrcoef(x_work.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)
    mat = corr.astype(float, copy=False)

    k = int(top_k)
    if 0 < k < int(n_nodes):
        masked = np.zeros_like(mat)
        for i in range(int(n_nodes)):
            idx = np.argpartition(-mat[i], kth=k - 1)[:k]
            masked[i, idx] = 1.0
        mat = mat * masked
    return mat


def _resolve_builtin_adj_matrix(
    *,
    adj: Any,
    n_nodes: int,
    x_work: np.ndarray,
    top_k: int,
) -> np.ndarray:
    if adj is None:
        adj_s = "identity"
    else:
        adj_s = str(adj).strip().lower() if isinstance(adj, str) else ""

    if adj_s in {"", "none", "null"}:
        adj_s = "identity"

    if adj_s in {"identity", "eye", "i"}:
        return np.eye(int(n_nodes), dtype=float)
    if adj_s in {"ring", "cycle"}:
        mat = np.zeros((int(n_nodes), int(n_nodes)), dtype=float)
        for i in range(int(n_nodes)):
            mat[i, (i - 1) % int(n_nodes)] = 1.0
            mat[i, (i + 1) % int(n_nodes)] = 1.0
        return mat
    if adj_s in {"fully-connected", "fully", "all"}:
        return np.ones((int(n_nodes), int(n_nodes)), dtype=float)
    if adj_s in {"corr", "correlation"}:
        return _corr_topk_adj_matrix(x_work, n_nodes=int(n_nodes), top_k=int(top_k))
    return np.asarray(adj, dtype=float)


def _resolve_adj_matrix(
    *,
    adj: Any,
    adj_path: str,
    n_nodes: int,
    x_work: np.ndarray,
    top_k: int,
) -> np.ndarray:
    if int(n_nodes) <= 0:
        raise ValueError("n_nodes must be >= 1")

    mat = _load_adj_matrix_from_path(adj_path)
    if mat is None:
        mat = _resolve_builtin_adj_matrix(
            adj=adj,
            n_nodes=int(n_nodes),
            x_work=x_work,
            top_k=int(top_k),
        )

    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"adj must be a square 2D matrix, got shape {mat.shape}")
    if mat.shape[0] != int(n_nodes):
        raise ValueError(f"adj must be shape ({int(n_nodes)},{int(n_nodes)}), got {mat.shape}")

    out = mat.astype(float, copy=False)
    out = out + np.eye(int(n_nodes), dtype=float)
    return _row_normalize_adj(out)


def _normalize_multivariate_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_work = x.astype(float, copy=False)
    mean = np.zeros((int(x.shape[1]),), dtype=float)
    std = np.ones((int(x.shape[1]),), dtype=float)
    cols: list[np.ndarray] = []
    for j in range(int(x.shape[1])):
        col_scaled, mean_j, std_j = _normalize_series(x_work[:, j])
        cols.append(col_scaled)
        mean[j] = mean_j
        std[j] = std_j
    return (np.stack(cols, axis=1), mean, std)


def _maybe_normalize_multivariate_matrix(
    x: np.ndarray,
    *,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not bool(normalize):
        return (
            x.astype(float, copy=False),
            np.zeros((int(x.shape[1]),), dtype=float),
            np.ones((int(x.shape[1]),), dtype=float),
        )
    return _normalize_multivariate_matrix(x)


def _maybe_denormalize_multivariate_forecast(
    yhat: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    normalize: bool,
) -> np.ndarray:
    out = np.asarray(yhat, dtype=float)
    if not bool(normalize):
        return out
    n_nodes = int(mean.shape[0])
    return out * std.reshape(1, n_nodes) + mean.reshape(1, n_nodes)


def _prepare_torch_multivariate_training_data(
    x: np.ndarray,
    *,
    lags: int,
    horizon: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    h, lag_count = _validated_multivariate_horizon_lags(horizon=horizon, lags=lags)
    x_work, mean, std = _maybe_normalize_multivariate_matrix(x, normalize=bool(normalize))
    X, Y = _make_lagged_xy_multivariate(x_work, lags=lag_count, horizon=h)
    return x_work, mean, std, X, Y, int(x.shape[1])


def _predict_torch_multivariate_forecast(
    model: Any,
    *,
    torch_mod: Any,
    x_work: np.ndarray,
    lag_count: int,
    horizon: int,
    n_nodes: int,
    device: str,
) -> np.ndarray:
    feat = _latest_multivariate_window(x_work, lag_count=lag_count)
    with torch_mod.no_grad():
        feat_t = torch_mod.tensor(
            feat,
            dtype=torch_mod.float32,
            device=torch_mod.device(str(device)),
        )
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(int(horizon), int(n_nodes))
    return np.asarray(yhat_t, dtype=float)


def _finalize_torch_multivariate_forecast(
    model: Any,
    *,
    torch_mod: Any,
    x_work: np.ndarray,
    lag_count: int,
    horizon: int,
    n_nodes: int,
    device: str,
    mean: np.ndarray,
    std: np.ndarray,
    normalize: bool,
) -> np.ndarray:
    return _maybe_denormalize_multivariate_forecast(
        _predict_torch_multivariate_forecast(
            model,
            torch_mod=torch_mod,
            x_work=x_work,
            lag_count=lag_count,
            horizon=horizon,
            n_nodes=n_nodes,
            device=device,
        ),
        mean=mean,
        std=std,
        normalize=bool(normalize),
    )


def _build_torch_multivariate_train_config(
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
    scheduler_restart_period: int,
    scheduler_restart_mult: int,
    scheduler_pct_start: float,
    restore_best: bool,
    min_epochs: int,
    amp: bool,
    amp_dtype: str,
    warmup_epochs: int,
    min_lr: float,
    grad_accum_steps: int,
    monitor: str,
    monitor_mode: str,
    min_delta: float,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    scheduler_patience: int,
    grad_clip_mode: str,
    grad_clip_value: float,
    scheduler_plateau_factor: float,
    scheduler_plateau_threshold: float,
    ema_decay: float,
    ema_warmup_epochs: int,
    swa_start_epoch: int,
    lookahead_steps: int,
    lookahead_alpha: float,
    sam_rho: float,
    sam_adaptive: bool,
    horizon_loss_decay: float,
    input_dropout: float,
    temporal_dropout: float,
    grad_noise_std: float,
    gc_mode: str,
    agc_clip_factor: float,
    agc_eps: float,
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
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


def torch_stgcn_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 24,
    d_model: int = 64,
    num_blocks: int = 2,
    kernel_size: int = 3,
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
    Torch STGCN-style spatiotemporal baseline (lite) on a wide `(T, N)` target matrix.

    Input contract: `(n_obs, n_nodes)` matrix (or DataFrame columns = nodes).
    Output contract: `(horizon, n_nodes)` matrix.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_2d_float_array(train)
    h, lag_count = _validated_multivariate_horizon_lags(horizon=horizon, lags=lags)
    k = _validated_graph_kernel_size(kernel_size)
    d, blocks, drop = _validated_torch_multivariate_model_dims(
        d_model=d_model,
        num_blocks=num_blocks,
        dropout=dropout,
    )

    x_work, mean, std, X, Y, n_nodes = _prepare_torch_multivariate_training_data(
        x,
        lags=lag_count,
        horizon=h,
        normalize=bool(normalize),
    )
    adj_mat = _resolve_adj_matrix(
        adj=adj,
        adj_path=str(adj_path),
        n_nodes=n_nodes,
        x_work=x_work,
        top_k=int(adj_top_k),
    )

    class _STGCNBlock(nn.Module):
        def __init__(self, *, a: Any) -> None:
            super().__init__()
            self.register_buffer("adj", a)
            self.temporal = nn.Conv2d(d, d, kernel_size=(k, 1), bias=True)
            self.norm = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(2 * d, d),
            )
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any) -> Any:  # (B, T, N, d)
            z = xb.permute(0, 3, 1, 2)  # (B, d, T, N)
            z = F.pad(z, (0, 0, k - 1, 0))
            z = self.temporal(z).permute(0, 2, 3, 1)  # (B, T, N, d)
            z = torch.einsum("ij,btjd->btid", self.adj, z)
            xb = xb + self.drop(torch.relu(z))
            xb = xb + self.drop(self.ffn(self.norm(xb)))
            return xb

    class _STGCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(1, d)
            a = torch.tensor(adj_mat, dtype=torch.float32)
            self.blocks = nn.ModuleList([_STGCNBlock(a=a) for _ in range(blocks)])
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # (B, T, N)
            z = self.in_proj(xb.unsqueeze(-1))
            for blk in self.blocks:
                z = blk(z)
            yhat = self.head(z[:, -1, :, :]).transpose(1, 2)  # (B, h, N)
            return yhat

    cfg = _build_torch_multivariate_train_config(
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
        scheduler_restart_period=scheduler_restart_period,
        scheduler_restart_mult=scheduler_restart_mult,
        scheduler_pct_start=scheduler_pct_start,
        restore_best=restore_best,
        min_epochs=min_epochs,
        amp=amp,
        amp_dtype=amp_dtype,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        grad_accum_steps=grad_accum_steps,
        monitor=monitor,
        monitor_mode=monitor_mode,
        min_delta=min_delta,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        scheduler_patience=scheduler_patience,
        grad_clip_mode=grad_clip_mode,
        grad_clip_value=grad_clip_value,
        scheduler_plateau_factor=scheduler_plateau_factor,
        scheduler_plateau_threshold=scheduler_plateau_threshold,
        ema_decay=ema_decay,
        ema_warmup_epochs=ema_warmup_epochs,
        swa_start_epoch=swa_start_epoch,
        lookahead_steps=lookahead_steps,
        lookahead_alpha=lookahead_alpha,
        sam_rho=sam_rho,
        sam_adaptive=sam_adaptive,
        horizon_loss_decay=horizon_loss_decay,
        input_dropout=input_dropout,
        temporal_dropout=temporal_dropout,
        grad_noise_std=grad_noise_std,
        gc_mode=gc_mode,
        agc_clip_factor=agc_clip_factor,
        agc_eps=agc_eps,
        checkpoint_dir=checkpoint_dir,
        save_best_checkpoint=save_best_checkpoint,
        save_last_checkpoint=save_last_checkpoint,
        resume_checkpoint_path=resume_checkpoint_path,
        resume_checkpoint_strict=resume_checkpoint_strict,
    )
    model = _train_loop(_STGCN(), X, Y, cfg=cfg, device=str(device))

    return _finalize_torch_multivariate_forecast(
        model,
        torch_mod=torch,
        x_work=x_work,
        lag_count=lag_count,
        horizon=h,
        n_nodes=n_nodes,
        device=str(device),
        mean=mean,
        std=std,
        normalize=bool(normalize),
    )


def torch_graphwavenet_forecast(
    train: Any,
    horizon: int,
    *,
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
    Torch Graph WaveNet-style multivariate baseline (lite) on a wide `(T, N)` target matrix.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_2d_float_array(train)
    h, lag_count = _validated_multivariate_horizon_lags(horizon=horizon, lags=lags)
    k = _validated_graph_kernel_size(kernel_size)
    base = int(dilation_base)
    d, blocks, drop = _validated_torch_multivariate_model_dims(
        d_model=d_model,
        num_blocks=num_blocks,
        dropout=dropout,
    )
    if base <= 0:
        raise ValueError("dilation_base must be >= 1")
    if int(adj_emb_dim) <= 0:
        raise ValueError("adj_emb_dim must be >= 1")

    x_work, mean, std, X, Y, n_nodes = _prepare_torch_multivariate_training_data(
        x,
        lags=lag_count,
        horizon=h,
        normalize=bool(normalize),
    )
    static_adj = _resolve_adj_matrix(
        adj=adj,
        adj_path=str(adj_path),
        n_nodes=n_nodes,
        x_work=x_work,
        top_k=int(adj_top_k),
    )

    class _GWBlock(nn.Module):
        def __init__(self, *, dilation: int) -> None:
            super().__init__()
            self.dilation = int(dilation)
            self.filter = nn.Conv2d(
                d,
                d,
                kernel_size=(k, 1),
                dilation=(self.dilation, 1),
            )
            self.gate = nn.Conv2d(
                d,
                d,
                kernel_size=(k, 1),
                dilation=(self.dilation, 1),
            )
            self.res = nn.Conv2d(d, d, kernel_size=(1, 1))
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any, a: Any) -> Any:  # xb: (B, d, T, N), a: (N, N)
            pad_left = (k - 1) * int(self.dilation)
            z = F.pad(xb, (0, 0, pad_left, 0))
            z = torch.tanh(self.filter(z)) * torch.sigmoid(self.gate(z))
            z = z.permute(0, 2, 3, 1)  # (B, T, N, d)
            z = torch.einsum("ij,btjd->btid", a, z).permute(0, 3, 1, 2)  # (B, d, T, N)
            z = self.drop(torch.relu(z))
            xb = xb + self.res(z)
            xb = self.norm(xb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return xb

    class _GraphWaveNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(1, d)
            self.register_buffer("static_adj", torch.tensor(static_adj, dtype=torch.float32))
            self.blocks = nn.ModuleList([_GWBlock(dilation=(base**i)) for i in range(int(blocks))])
            self.adaptive_adj = bool(adaptive_adj)
            self.head = nn.Linear(d, h)
            if self.adaptive_adj:
                self.node_emb1 = nn.Parameter(
                    torch.randn((n_nodes, int(adj_emb_dim)), dtype=torch.float32) * 0.1
                )
                self.node_emb2 = nn.Parameter(
                    torch.randn((n_nodes, int(adj_emb_dim)), dtype=torch.float32) * 0.1
                )

        def _current_adj(self) -> Any:
            a = self.static_adj
            if not self.adaptive_adj:
                return a
            adapt = torch.softmax(
                torch.relu(self.node_emb1 @ self.node_emb2.transpose(0, 1)),
                dim=-1,
            )
            adapt = adapt + torch.eye(adapt.shape[0], device=adapt.device, dtype=adapt.dtype)
            mixed = a + adapt
            mixed = mixed / (mixed.sum(dim=-1, keepdim=True) + 1e-6)
            return mixed

        def forward(self, xb: Any) -> Any:  # (B, T, N)
            z = self.in_proj(xb.unsqueeze(-1)).permute(0, 3, 1, 2)  # (B, d, T, N)
            a = self._current_adj()
            for blk in self.blocks:
                z = blk(z, a=a)
            last = z[:, :, -1, :].transpose(1, 2)  # (B, N, d)
            yhat = self.head(last).transpose(1, 2)  # (B, h, N)
            return yhat

    cfg = _build_torch_multivariate_train_config(
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
        scheduler_restart_period=scheduler_restart_period,
        scheduler_restart_mult=scheduler_restart_mult,
        scheduler_pct_start=scheduler_pct_start,
        restore_best=restore_best,
        min_epochs=min_epochs,
        amp=amp,
        amp_dtype=amp_dtype,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        grad_accum_steps=grad_accum_steps,
        monitor=monitor,
        monitor_mode=monitor_mode,
        min_delta=min_delta,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        scheduler_patience=scheduler_patience,
        grad_clip_mode=grad_clip_mode,
        grad_clip_value=grad_clip_value,
        scheduler_plateau_factor=scheduler_plateau_factor,
        scheduler_plateau_threshold=scheduler_plateau_threshold,
        ema_decay=ema_decay,
        ema_warmup_epochs=ema_warmup_epochs,
        swa_start_epoch=swa_start_epoch,
        lookahead_steps=lookahead_steps,
        lookahead_alpha=lookahead_alpha,
        sam_rho=sam_rho,
        sam_adaptive=sam_adaptive,
        horizon_loss_decay=horizon_loss_decay,
        input_dropout=input_dropout,
        temporal_dropout=temporal_dropout,
        grad_noise_std=grad_noise_std,
        gc_mode=gc_mode,
        agc_clip_factor=agc_clip_factor,
        agc_eps=agc_eps,
        checkpoint_dir=checkpoint_dir,
        save_best_checkpoint=save_best_checkpoint,
        save_last_checkpoint=save_last_checkpoint,
        resume_checkpoint_path=resume_checkpoint_path,
        resume_checkpoint_strict=resume_checkpoint_strict,
    )
    model = _train_loop(_GraphWaveNet(), X, Y, cfg=cfg, device=str(device))

    return _finalize_torch_multivariate_forecast(
        model,
        torch_mod=torch,
        x_work=x_work,
        lag_count=lag_count,
        horizon=h,
        n_nodes=n_nodes,
        device=str(device),
        mean=mean,
        std=std,
        normalize=bool(normalize),
    )
