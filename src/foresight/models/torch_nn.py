from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("Training series must be non-empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("Series contains NaN/inf")
    return x


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'Torch models require PyTorch. Install with: pip install -e ".[torch]"'
        ) from e
    return torch


def _make_lagged_xy_multi(
    x: np.ndarray, *, lags: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.size)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if n < lag_count + h:
        raise ValueError(f"Need >= lags+horizon points (lags={lag_count}, horizon={h}), got {n}")

    rows = n - lag_count - h + 1
    X = np.empty((rows, lag_count), dtype=float)
    Y = np.empty((rows, h), dtype=float)
    for i in range(rows):
        t = i + lag_count
        X[i, :] = x[t - lag_count : t]
        Y[i, :] = x[t : t + h]
    return X, Y


def _normalize_series(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        std = 1.0
    return (x - mean) / std, mean, std


@dataclass(frozen=True)
class TorchTrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    seed: int
    patience: int
    loss: str = "mse"
    val_split: float = 0.0
    grad_clip_norm: float = 0.0
    optimizer: str = "adam"
    momentum: float = 0.9
    scheduler: str = "none"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    restore_best: bool = True


def _train_loop(
    model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cfg: TorchTrainConfig,
    device: str,
    loss_fn_override: Any | None = None,
) -> Any:
    torch = _require_torch()
    nn = torch.nn

    if cfg.epochs <= 0:
        raise ValueError("epochs must be >= 1")
    if cfg.lr <= 0.0:
        raise ValueError("lr must be > 0")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    if cfg.patience <= 0:
        raise ValueError("patience must be >= 1")
    if float(cfg.val_split) < 0.0 or float(cfg.val_split) >= 0.5:
        raise ValueError("val_split must be in [0, 0.5)")
    if float(cfg.grad_clip_norm) < 0.0:
        raise ValueError("grad_clip_norm must be >= 0")

    torch.manual_seed(int(cfg.seed))

    dev = torch.device(str(device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available")

    model = model.to(dev)

    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=dev)

    n = int(X_t.shape[0])
    val_n = 0
    if float(cfg.val_split) > 0.0 and n >= 5:
        val_n = max(1, int(round(float(cfg.val_split) * n)))
        val_n = min(val_n, n - 1)

    if val_n > 0:
        train_end = n - val_n
        X_train, Y_train = X_t[:train_end], Y_t[:train_end]
        X_val, Y_val = X_t[train_end:], Y_t[train_end:]
    else:
        X_train, Y_train = X_t, Y_t
        X_val, Y_val = None, None

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train),
        batch_size=int(cfg.batch_size),
        shuffle=True,
    )
    val_loader = (
        None
        if X_val is None
        else torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, Y_val),
            batch_size=int(cfg.batch_size),
            shuffle=False,
        )
    )

    opt_name = str(cfg.optimizer).lower().strip()
    if opt_name in {"adam", ""}:
        opt = torch.optim.Adam(
            model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
        )
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
        )
    elif opt_name == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.lr),
            momentum=float(cfg.momentum),
            weight_decay=float(cfg.weight_decay),
        )
    else:
        raise ValueError("optimizer must be one of: adam, adamw, sgd")

    loss_name = str(cfg.loss).lower().strip()
    if loss_name in {"mse", ""}:
        loss_fn = nn.MSELoss()
    elif loss_name in {"mae", "l1"}:
        loss_fn = nn.L1Loss()
    elif loss_name in {"huber", "smoothl1"}:
        loss_fn = nn.SmoothL1Loss()
    else:
        raise ValueError("loss must be one of: mse, mae, huber")

    if loss_fn_override is not None:
        loss_fn = loss_fn_override

    sched_name = str(cfg.scheduler).lower().strip()
    if sched_name in {"none", ""}:
        sched = None
    elif sched_name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg.epochs))
    elif sched_name == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=int(cfg.scheduler_step_size), gamma=float(cfg.scheduler_gamma)
        )
    else:
        raise ValueError("scheduler must be one of: none, cosine, step")

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    bad_epochs = 0

    for _epoch in range(int(cfg.epochs)):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if float(cfg.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(cfg.grad_clip_norm)
                )
            opt.step()

            total += float(loss.detach().cpu().item()) * int(xb.shape[0])
            count += int(xb.shape[0])

        train_loss = total / max(1, count)

        if val_loader is not None:
            model.eval()
            v_total = 0.0
            v_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    v_loss = loss_fn(pred, yb)
                    v_total += float(v_loss.detach().cpu().item()) * int(xb.shape[0])
                    v_count += int(xb.shape[0])
            monitor = v_total / max(1, v_count)
        else:
            monitor = train_loss

        if float(monitor) + 1e-12 < best_loss:
            best_loss = float(monitor)
            bad_epochs = 0
            if bool(cfg.restore_best):
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg.patience):
                break

        if sched is not None:
            sched.step()

    if bool(cfg.restore_best) and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def torch_mlp_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 24,
    hidden_sizes: Any = (64, 64),
    dropout: float = 0.0,
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
    Torch MLP direct multi-horizon forecast on lag windows.

    This is a lightweight neural baseline intended to be used behind optional
    dependencies (PyTorch). It trains per-series per-window, so keep settings small.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=int(lags), horizon=h)

    if isinstance(hidden_sizes, int):
        hs = (int(hidden_sizes),)
    elif isinstance(hidden_sizes, str):
        parts = [p.strip() for p in hidden_sizes.split(",") if p.strip()]
        hs = tuple(int(p) for p in parts)
    elif isinstance(hidden_sizes, list | tuple):
        hs = tuple(int(s) for s in hidden_sizes)
    else:
        hs = (int(hidden_sizes),)

    if not hs or any(int(s) <= 0 for s in hs):
        raise ValueError("hidden_sizes must be a non-empty sequence of positive ints")
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    layers: list[Any] = []
    in_dim = int(X.shape[1])
    for size in hs:
        layers.append(nn.Linear(in_dim, int(size)))
        layers.append(nn.ReLU())
        if drop > 0.0:
            layers.append(nn.Dropout(p=drop))
        in_dim = int(size)
    layers.append(nn.Linear(in_dim, h))
    model = nn.Sequential(*layers)

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

    feat = x_work[-int(lags) :].astype(float, copy=False).reshape(1, -1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_lstm_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
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
    Torch LSTM direct multi-horizon forecast on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if int(hidden_size) <= 0:
        raise ValueError("hidden_size must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=int(lags), horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    class _LSTMDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=(drop if int(num_layers) > 1 else 0.0),
                batch_first=True,
            )
            self.head = nn.Linear(int(hidden_size), h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            out, _ = self.lstm(xb)
            last = out[:, -1, :]
            return self.head(last)

    model = _LSTMDirect()

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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-int(lags) :].astype(float, copy=False).reshape(1, int(lags), 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_gru_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
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
    Torch GRU direct multi-horizon forecast on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if int(hidden_size) <= 0:
        raise ValueError("hidden_size must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=int(lags), horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    class _GRUDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=(drop if int(num_layers) > 1 else 0.0),
                batch_first=True,
            )
            self.head = nn.Linear(int(hidden_size), h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            out, _ = self.gru(xb)
            last = out[:, -1, :]
            return self.head(last)

    model = _GRUDirect()

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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-int(lags) :].astype(float, copy=False).reshape(1, int(lags), 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_tcn_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 24,
    channels: Any = (16, 16, 16),
    kernel_size: int = 3,
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
    Torch TCN (Temporal Convolutional Network) direct multi-horizon forecast.

    Uses causal dilated Conv1D blocks over lag windows.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if int(lags) <= 0:
        raise ValueError("lags must be >= 1")
    if int(kernel_size) <= 0:
        raise ValueError("kernel_size must be >= 1")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=int(lags), horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    if isinstance(channels, int):
        chans = (int(channels),)
    elif isinstance(channels, str):
        parts = [p.strip() for p in channels.split(",") if p.strip()]
        chans = tuple(int(p) for p in parts)
    elif isinstance(channels, list | tuple):
        chans = tuple(int(c) for c in channels)
    else:
        chans = (int(channels),)

    if not chans or any(int(c) <= 0 for c in chans):
        raise ValueError("channels must be a non-empty sequence of positive ints")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    k = int(kernel_size)

    class _TCNDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            in_ch = 1
            for i, out_ch in enumerate(chans):
                dilation = 2 ** int(i)
                pad = dilation * (k - 1)
                layers.append(nn.ConstantPad1d((pad, 0), 0.0))
                layers.append(
                    nn.Conv1d(
                        in_channels=int(in_ch),
                        out_channels=int(out_ch),
                        kernel_size=k,
                        dilation=int(dilation),
                    )
                )
                layers.append(nn.ReLU())
                if drop > 0.0:
                    layers.append(nn.Dropout(p=drop))
                in_ch = int(out_ch)

            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(int(in_ch), h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)
            xch = xb.transpose(1, 2)  # (B, 1, T)
            z = self.net(xch)
            last = z[:, :, -1]
            return self.head(last)

    model = _TCNDirect()

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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-int(lags) :].astype(float, copy=False).reshape(1, int(lags), 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_nbeats_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 48,
    num_blocks: int = 3,
    num_layers: int = 2,
    layer_width: int = 64,
    dropout: float = 0.0,
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
    Torch N-BEATS-style direct multi-horizon forecast.

    This is a simplified "generic" N-BEATS:
    - Each block outputs (backcast, forecast) via an MLP + linear projection.
    - Residual is updated by subtracting backcast; forecasts are summed.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if int(num_blocks) <= 0:
        raise ValueError("num_blocks must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")
    if int(layer_width) <= 0:
        raise ValueError("layer_width must be >= 1")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    class _NBeatsBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            in_dim = lag_count
            for _ in range(int(num_layers)):
                layers.append(nn.Linear(int(in_dim), int(layer_width)))
                layers.append(nn.ReLU())
                if drop > 0.0:
                    layers.append(nn.Dropout(p=drop))
                in_dim = int(layer_width)
            self.mlp = nn.Sequential(*layers)
            self.theta = nn.Linear(int(layer_width), int(lag_count + h))

        def forward(self, xb: Any) -> tuple[Any, Any]:
            z = self.mlp(xb)
            theta = self.theta(z)
            backcast = theta[:, :lag_count]
            forecast = theta[:, lag_count:]
            return backcast, forecast

    class _NBeats(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_NBeatsBlock() for _ in range(int(num_blocks))])

        def forward(self, xb: Any) -> Any:
            residual = xb
            forecast = torch.zeros((xb.shape[0], h), device=xb.device, dtype=xb.dtype)
            for block in self.blocks:
                backcast, f = block(residual)
                residual = residual - backcast
                forecast = forecast + f
            return forecast

    model = _NBeats()

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

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, -1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_nlinear_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 48,
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
    Torch NLinear-style baseline: linear mapping of (lags -> horizon) with last-value centering.
    """
    torch = _require_torch()
    nn = torch.nn

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

    class _NLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(lag_count, h)

        def forward(self, xb: Any) -> Any:
            last = xb[:, -1:].detach()
            xc = xb - last
            out = self.fc(xc) + last
            return out

    model = _NLinear()
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

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, -1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_dlinear_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 48,
    ma_window: int = 25,
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
    Torch DLinear-style baseline: moving-average decomposition + linear mapping on trend/seasonal parts.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if int(ma_window) <= 1:
        raise ValueError("ma_window must be >= 2")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)

    k = int(ma_window)
    left = k // 2
    right = k - 1 - left

    class _DLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc_trend = nn.Linear(lag_count, h)
            self.fc_seasonal = nn.Linear(lag_count, h)

        def forward(self, xb: Any) -> Any:
            x1 = xb.unsqueeze(1)  # (B, 1, T)
            xpad = F.pad(x1, (left, right), mode="replicate")
            weight = torch.ones((1, 1, k), device=xb.device, dtype=xb.dtype) / float(k)
            trend = F.conv1d(xpad, weight).squeeze(1)
            seasonal = xb - trend
            return self.fc_trend(trend) + self.fc_seasonal(seasonal)

    model = _DLinear()
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

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, -1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_transformer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
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
    Torch Transformer encoder direct multi-horizon forecast on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError("d_model must be >= 1")
    if heads <= 0:
        raise ValueError("nhead must be >= 1")
    if d % heads != 0:
        raise ValueError("d_model must be divisible by nhead")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")
    if int(dim_feedforward) <= 0:
        raise ValueError("dim_feedforward must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _TransformerDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=int(dim_feedforward),
                dropout=drop,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=int(num_layers))
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            z = self.embed(xb) + self.pos
            z = self.enc(z)
            last = z[:, -1, :]
            return self.head(last)

    model = _TransformerDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_patchtst_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
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
    PatchTST-style: patch lag windows and apply a Transformer encoder.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")

    p = int(patch_len)
    s = int(stride)
    if p <= 0:
        raise ValueError("patch_len must be >= 1")
    if s <= 0:
        raise ValueError("stride must be >= 1")
    if p > lag_count:
        raise ValueError("patch_len must be <= lags")

    n_patches = 1 + (lag_count - p) // s
    if n_patches <= 0:
        raise ValueError("Invalid patch configuration")

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError("d_model must be >= 1")
    if heads <= 0:
        raise ValueError("nhead must be >= 1")
    if d % heads != 0:
        raise ValueError("d_model must be divisible by nhead")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _PatchTSTDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = nn.Linear(p, d)
            self.pos = nn.Parameter(torch.zeros((1, n_patches, d), dtype=torch.float32))
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=int(dim_feedforward),
                dropout=drop,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=int(num_layers))
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            x0 = xb.squeeze(-1)  # (B, T)
            patches = x0.unfold(dimension=1, size=p, step=s)  # (B, P, p)
            z = self.patch_embed(patches) + self.pos
            z = self.enc(z)
            last = z[:, -1, :]
            return self.head(last)

    model = _PatchTSTDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_tsmixer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_blocks: int = 4,
    token_mixing_hidden: int = 128,
    channel_mixing_hidden: int = 128,
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
    Torch TSMixer-style direct multi-horizon forecast.

    Alternates token-mixing (across time) and channel-mixing (across features).
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if int(d_model) <= 0:
        raise ValueError("d_model must be >= 1")
    if int(num_blocks) <= 0:
        raise ValueError("num_blocks must be >= 1")
    if int(token_mixing_hidden) <= 0:
        raise ValueError("token_mixing_hidden must be >= 1")
    if int(channel_mixing_hidden) <= 0:
        raise ValueError("channel_mixing_hidden must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    d = int(d_model)

    class _MixerBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm_t = nn.LayerNorm(d)
            self.token_mlp = nn.Sequential(
                nn.Linear(lag_count, int(token_mixing_hidden)),
                nn.ReLU(),
                nn.Dropout(p=drop),
                nn.Linear(int(token_mixing_hidden), lag_count),
            )
            self.norm_c = nn.LayerNorm(d)
            self.channel_mlp = nn.Sequential(
                nn.Linear(d, int(channel_mixing_hidden)),
                nn.ReLU(),
                nn.Dropout(p=drop),
                nn.Linear(int(channel_mixing_hidden), d),
            )

        def forward(self, xb: Any) -> Any:  # (B, T, d)
            z = self.norm_t(xb)
            zt = self.token_mlp(z.transpose(1, 2)).transpose(1, 2)
            xb = xb + zt

            z = self.norm_c(xb)
            xb = xb + self.channel_mlp(z)
            return xb

    class _TSMixer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.blocks = nn.ModuleList([_MixerBlock() for _ in range(int(num_blocks))])
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            z = self.embed(xb)
            for blk in self.blocks:
                z = blk(z)
            last = z[:, -1, :]
            return self.head(last)

    model = _TSMixer()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_cnn_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 48,
    channels: Any = (32, 32, 32),
    kernel_size: int = 3,
    dropout: float = 0.1,
    pool: str = "last",
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
    Torch CNN direct multi-horizon forecast on lag windows.

    A simple Conv1D stack over the lag window, pooled to a fixed feature vector,
    then projected to the forecast horizon.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if int(kernel_size) <= 0:
        raise ValueError("kernel_size must be >= 1")

    if isinstance(channels, int):
        chans = (int(channels),)
    elif isinstance(channels, str):
        parts = [p.strip() for p in channels.split(",") if p.strip()]
        chans = tuple(int(p) for p in parts)
    elif isinstance(channels, list | tuple):
        chans = tuple(int(c) for c in channels)
    else:
        chans = (int(channels),)

    if not chans or any(int(c) <= 0 for c in chans):
        raise ValueError("channels must be a non-empty sequence of positive ints")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    pool_s = str(pool).lower().strip()
    if pool_s not in {"last", "mean", "max"}:
        raise ValueError("pool must be one of: last, mean, max")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    k = int(kernel_size)
    pad = k // 2

    class _CNNDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            in_ch = 1
            for out_ch in chans:
                layers.append(
                    nn.Conv1d(
                        in_channels=int(in_ch),
                        out_channels=int(out_ch),
                        kernel_size=k,
                        padding=int(pad),
                    )
                )
                layers.append(nn.ReLU())
                if drop > 0.0:
                    layers.append(nn.Dropout(p=drop))
                in_ch = int(out_ch)
            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(int(in_ch), h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)
            xch = xb.transpose(1, 2)  # (B, 1, T)
            z = self.net(xch)  # (B, C, T)
            if pool_s == "last":
                feat = z[:, :, -1]
            elif pool_s == "mean":
                feat = z.mean(dim=2)
            else:
                feat = z.amax(dim=2)
            return self.head(feat)

    model = _CNNDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_resnet1d_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    channels: int = 32,
    num_blocks: int = 4,
    kernel_size: int = 3,
    dropout: float = 0.1,
    pool: str = "last",
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
    Torch ResNet-1D direct multi-horizon forecast on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    c = int(channels)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if c <= 0:
        raise ValueError("channels must be >= 1")
    if int(num_blocks) <= 0:
        raise ValueError("num_blocks must be >= 1")
    if int(kernel_size) <= 0:
        raise ValueError("kernel_size must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    pool_s = str(pool).lower().strip()
    if pool_s not in {"last", "mean", "max"}:
        raise ValueError("pool must be one of: last, mean, max")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    k = int(kernel_size)
    pad = k // 2

    class _ResBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(c, c, kernel_size=k, padding=int(pad))
            self.conv2 = nn.Conv1d(c, c, kernel_size=k, padding=int(pad))
            self.act = nn.ReLU()
            self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

        def forward(self, xch: Any) -> Any:
            z = self.act(self.conv1(xch))
            z = self.drop(z)
            z = self.conv2(z)
            return self.act(xch + z)

    class _ResNet1DDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Conv1d(1, c, kernel_size=1)
            self.blocks = nn.ModuleList([_ResBlock() for _ in range(int(num_blocks))])
            self.head = nn.Linear(c, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)
            xch = xb.transpose(1, 2)  # (B, 1, T)
            z = self.in_proj(xch)
            for blk in self.blocks:
                z = blk(z)
            if pool_s == "last":
                feat = z[:, :, -1]
            elif pool_s == "mean":
                feat = z.mean(dim=2)
            else:
                feat = z.amax(dim=2)
            return self.head(feat)

    model = _ResNet1DDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_wavenet_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    channels: int = 32,
    num_layers: int = 6,
    kernel_size: int = 2,
    dropout: float = 0.0,
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
    Torch WaveNet-style gated dilated CNN direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    c = int(channels)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if c <= 0:
        raise ValueError("channels must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")
    if int(kernel_size) <= 0:
        raise ValueError("kernel_size must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    k = int(kernel_size)

    class _WaveNetDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Conv1d(1, c, kernel_size=1)
            self.filter_convs = nn.ModuleList()
            self.gate_convs = nn.ModuleList()
            self.res_convs = nn.ModuleList()
            self.skip_convs = nn.ModuleList()
            self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

            for i in range(int(num_layers)):
                dilation = 2 ** int(i)
                self.filter_convs.append(nn.Conv1d(c, c, kernel_size=k, dilation=int(dilation)))
                self.gate_convs.append(nn.Conv1d(c, c, kernel_size=k, dilation=int(dilation)))
                self.res_convs.append(nn.Conv1d(c, c, kernel_size=1))
                self.skip_convs.append(nn.Conv1d(c, c, kernel_size=1))

            self.out1 = nn.Conv1d(c, c, kernel_size=1)
            self.out2 = nn.Conv1d(c, c, kernel_size=1)
            self.head = nn.Linear(c, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)
            xch = xb.transpose(1, 2)  # (B, 1, T)
            xh = self.in_proj(xch)
            skip = None

            for i in range(int(num_layers)):
                dilation = 2 ** int(i)
                pad = int(dilation) * (k - 1)
                xpad = F.pad(xh, (pad, 0), mode="constant", value=0.0)
                f = torch.tanh(self.filter_convs[i](xpad))
                g = torch.sigmoid(self.gate_convs[i](xpad))
                z = self.drop(f * g)
                s = self.skip_convs[i](z)
                skip = s if skip is None else skip + s
                xh = self.res_convs[i](z) + xh

            out = torch.relu(skip)
            out = torch.relu(self.out1(out))
            out = self.out2(out)
            feat = out[:, :, -1]
            return self.head(feat)

    model = _WaveNetDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_bilstm_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
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
    Torch BiLSTM direct multi-horizon forecast on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if int(hidden_size) <= 0:
        raise ValueError("hidden_size must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")

    lag_count = int(lags)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    class _BiLSTMDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=(drop if int(num_layers) > 1 else 0.0),
                bidirectional=True,
                batch_first=True,
            )
            self.head = nn.Linear(2 * int(hidden_size), h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            _out, (hn, _cn) = self.lstm(xb)
            # last layer forward + backward hidden states
            last = torch.cat([hn[-2, :, :], hn[-1, :, :]], dim=1)
            return self.head(last)

    model = _BiLSTMDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_bigru_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
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
    Torch BiGRU direct multi-horizon forecast on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if int(hidden_size) <= 0:
        raise ValueError("hidden_size must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")

    lag_count = int(lags)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    class _BiGRUDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=(drop if int(num_layers) > 1 else 0.0),
                bidirectional=True,
                batch_first=True,
            )
            self.head = nn.Linear(2 * int(hidden_size), h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            _out, hn = self.gru(xb)
            last = torch.cat([hn[-2, :, :], hn[-1, :, :]], dim=1)
            return self.head(last)

    model = _BiGRUDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_attn_gru_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 48,
    hidden_size: int = 32,
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
    restore_best: bool = True,
) -> np.ndarray:
    """
    Torch GRU + attention pooling (direct multi-horizon) on lag windows.

    Uses a learned attention weight per timestep over the GRU hidden states.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if int(hidden_size) <= 0:
        raise ValueError("hidden_size must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _AttnGRUDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=(drop if int(num_layers) > 1 else 0.0),
                batch_first=True,
            )
            self.attn = nn.Linear(int(hidden_size), 1)
            self.head = nn.Linear(int(hidden_size), h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            out, _hn = self.gru(xb)  # (B, T, H)
            scores = self.attn(out).squeeze(-1)  # (B, T)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
            ctx = torch.sum(out * w, dim=1)  # (B, H)
            return self.head(ctx)

    model = _AttnGRUDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_fnet_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 4,
    dim_feedforward: int = 256,
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
    Torch FNet-style model: Fourier mixing instead of attention (direct multi-horizon).
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if int(d_model) <= 0:
        raise ValueError("d_model must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")
    if int(dim_feedforward) <= 0:
        raise ValueError("dim_feedforward must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    d = int(d_model)

    class _FNetLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(
                nn.Linear(d, int(dim_feedforward)),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(int(dim_feedforward), d),
                nn.Dropout(p=drop),
            )

        def forward(self, x: Any) -> Any:  # (B, T, d)
            z = self.norm1(x)
            z = torch.fft.fft(z, dim=1).real
            x = x + z
            z = self.norm2(x)
            x = x + self.ff(z)
            return x

    class _FNetDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
            self.layers = nn.ModuleList([_FNetLayer() for _ in range(int(num_layers))])
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            z = self.embed(xb) + self.pos
            for layer in self.layers:
                z = layer(z)
            last = z[:, -1, :]
            return self.head(last)

    model = _FNetDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_gmlp_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 4,
    ffn_dim: int = 128,
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
    Torch gMLP-style model (direct multi-horizon) on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    d = int(d_model)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if d <= 0:
        raise ValueError("d_model must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")
    if int(ffn_dim) <= 0:
        raise ValueError("ffn_dim must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _SpatialGatingUnit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(int(ffn_dim))
            self.proj = nn.Linear(lag_count, lag_count)

        def forward(self, v: Any) -> Any:  # v: (B, T, ffn_dim)
            z = self.norm(v)
            z = z.transpose(1, 2)  # (B, ffn_dim, T)
            z = self.proj(z)
            return z.transpose(1, 2)

    class _gMLPLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, 2 * int(ffn_dim))
            self.act = nn.GELU()
            self.sgu = _SpatialGatingUnit()
            self.fc2 = nn.Linear(int(ffn_dim), d)
            self.drop = nn.Dropout(p=drop)

        def forward(self, x: Any) -> Any:  # (B, T, d)
            z = self.norm(x)
            z = self.fc1(z)
            u, v = z.chunk(2, dim=-1)
            u = self.act(u)
            v = self.sgu(v)
            z = u * v
            z = self.fc2(self.drop(z))
            return x + self.drop(z)

    class _gMLPDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
            self.layers = nn.ModuleList([_gMLPLayer() for _ in range(int(num_layers))])
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            z = self.embed(xb) + self.pos
            for layer in self.layers:
                z = layer(z)
            last = z[:, -1, :]
            return self.head(last)

    model = _gMLPDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_nhits_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    pool_sizes: Any = (1, 2, 4),
    num_blocks: int = 6,
    num_layers: int = 2,
    layer_width: int = 128,
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
    Torch N-HiTS-style multi-rate residual MLP (direct multi-horizon).

    This is a simplified variant: each block downsamples the lag window by avg-pooling,
    predicts a low-resolution backcast and a horizon forecast, then upsamples backcast
    to update the residual.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if int(num_blocks) <= 0:
        raise ValueError("num_blocks must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")
    if int(layer_width) <= 0:
        raise ValueError("layer_width must be >= 1")

    if isinstance(pool_sizes, int):
        pools = (int(pool_sizes),)
    elif isinstance(pool_sizes, str):
        parts = [p.strip() for p in pool_sizes.split(",") if p.strip()]
        pools = tuple(int(p) for p in parts)
    elif isinstance(pool_sizes, list | tuple):
        pools = tuple(int(p) for p in pool_sizes)
    else:
        pools = (int(pool_sizes),)

    pools = tuple(int(p) for p in pools if int(p) > 0)
    if not pools:
        raise ValueError("pool_sizes must contain at least one positive int")
    if any(int(p) > lag_count for p in pools):
        raise ValueError("pool_sizes values must be <= lags")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)

    class _NHITSBlock(nn.Module):
        def __init__(self, pool: int) -> None:
            super().__init__()
            self.pool = int(pool)
            n_low = 1 + (lag_count - int(pool)) // int(pool)
            if n_low <= 0:
                raise ValueError("Invalid pool configuration")

            layers: list[Any] = []
            in_dim = n_low
            for _ in range(int(num_layers)):
                layers.append(nn.Linear(int(in_dim), int(layer_width)))
                layers.append(nn.ReLU())
                if drop > 0.0:
                    layers.append(nn.Dropout(p=drop))
                in_dim = int(layer_width)
            self.mlp = nn.Sequential(*layers)
            self.theta = nn.Linear(int(layer_width), int(n_low + h))

        def forward(self, xb: Any) -> tuple[Any, Any]:
            x1 = xb.unsqueeze(1)  # (B, 1, T)
            low = F.avg_pool1d(x1, kernel_size=int(self.pool), stride=int(self.pool)).squeeze(1)
            z = self.mlp(low)
            theta = self.theta(z)
            n_low = int(low.shape[1])
            back_low = theta[:, :n_low]
            forecast = theta[:, n_low:]

            back_up = F.interpolate(
                back_low.unsqueeze(1),
                size=lag_count,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            return back_up, forecast

    class _NHITS(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: list[Any] = []
            for i in range(int(num_blocks)):
                blocks.append(_NHITSBlock(pool=pools[int(i) % len(pools)]))
            self.blocks = nn.ModuleList(blocks)

        def forward(self, xb: Any) -> Any:
            residual = xb
            forecast = torch.zeros((xb.shape[0], h), device=xb.device, dtype=xb.dtype)
            for blk in self.blocks:
                back, f = blk(residual)
                residual = residual - back
                forecast = forecast + f
            return forecast

    model = _NHITS()
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

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, -1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_tide_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    hidden_size: int = 128,
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
    Torch TiDE-style encoder/decoder MLP (direct multi-horizon).

    Simplified TiDE: encode lag window into a context vector, then decode each
    horizon step using a learned step embedding + small MLP head.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    d = int(d_model)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if d <= 0:
        raise ValueError("d_model must be >= 1")
    if int(hidden_size) <= 0:
        raise ValueError("hidden_size must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)

    class _TiDEDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(lag_count, int(hidden_size)),
                nn.ReLU(),
                nn.Dropout(p=drop),
                nn.Linear(int(hidden_size), d),
            )
            self.step_emb = nn.Embedding(h, d)
            self.dec = nn.Sequential(
                nn.Linear(d, int(hidden_size)),
                nn.ReLU(),
                nn.Dropout(p=drop),
                nn.Linear(int(hidden_size), 1),
            )

        def forward(self, xb: Any) -> Any:  # xb: (B, T)
            ctx = self.enc(xb)  # (B, d)
            steps = self.step_emb(torch.arange(h, device=xb.device, dtype=torch.long))  # (h, d)
            z = ctx.unsqueeze(1) + steps.unsqueeze(0)  # (B, h, d)
            out = self.dec(z).squeeze(-1)  # (B, h)
            return out

    model = _TiDEDirect()
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

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, -1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_deepar_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 48,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
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
    Torch DeepAR-style Gaussian RNN trained for one-step prediction, forecasted recursively.

    Output is the predictive mean for each step (sigma is learned but not returned).
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if int(hidden_size) <= 0:
        raise ValueError("hidden_size must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=1)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)
    Y_next = Y.reshape(Y.shape[0], 1)

    class _DeepAR(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=(drop if int(num_layers) > 1 else 0.0),
                batch_first=True,
            )
            self.head = nn.Linear(int(hidden_size), 2)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            out, _ = self.rnn(xb)
            last = out[:, -1, :]
            return self.head(last)  # (B, 2)

    def _gaussian_nll(pred: Any, yb: Any) -> Any:
        mu = pred[:, 0:1]
        sigma = F.softplus(pred[:, 1:2]) + 1e-3
        z = (yb - mu) / sigma
        return (0.5 * math.log(2.0 * math.pi) + torch.log(sigma) + 0.5 * (z**2)).mean()

    model = _DeepAR()
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
    model = _train_loop(
        model, X_seq, Y_next, cfg=cfg, device=str(device), loss_fn_override=_gaussian_nll
    )

    hist = x_work[-lag_count:].astype(float, copy=True)
    preds: list[float] = []

    dev = torch.device(str(device))
    with torch.no_grad():
        for _ in range(h):
            feat = hist.reshape(1, lag_count, 1)
            feat_t = torch.tensor(feat, dtype=torch.float32, device=dev)
            out = model(feat_t)
            mu = float(out[:, 0].detach().cpu().item())
            preds.append(mu)
            hist = np.concatenate([hist[1:], np.array([mu], dtype=float)], axis=0)

    yhat = np.asarray(preds, dtype=float)
    if bool(normalize):
        yhat = yhat * std + mean
    return yhat


def torch_qrnn_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 48,
    q: float = 0.5,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
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
    Torch quantile-regression RNN (one-step), forecasted recursively.

    Trains with pinball loss at quantile q, returns that quantile as a point forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    q_f = float(q)
    if not (0.0 < q_f < 1.0):
        raise ValueError("q must be in (0, 1)")

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if int(hidden_size) <= 0:
        raise ValueError("hidden_size must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=1)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)
    Y_next = Y.reshape(Y.shape[0], 1)

    class _QRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=(drop if int(num_layers) > 1 else 0.0),
                batch_first=True,
            )
            self.head = nn.Linear(int(hidden_size), 1)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            out, _ = self.rnn(xb)
            last = out[:, -1, :]
            return self.head(last)  # (B, 1)

    def _pinball(pred: Any, yb: Any) -> Any:
        u = yb - pred
        return torch.mean(torch.maximum(q_f * u, (q_f - 1.0) * u))

    model = _QRNN()
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
    model = _train_loop(
        model, X_seq, Y_next, cfg=cfg, device=str(device), loss_fn_override=_pinball
    )

    hist = x_work[-lag_count:].astype(float, copy=True)
    preds: list[float] = []

    dev = torch.device(str(device))
    with torch.no_grad():
        for _ in range(h):
            feat = hist.reshape(1, lag_count, 1)
            feat_t = torch.tensor(feat, dtype=torch.float32, device=dev)
            out = model(feat_t)
            yq = float(out.reshape(-1)[0].detach().cpu().item())
            preds.append(yq)
            hist = np.concatenate([hist[1:], np.array([yq], dtype=float)], axis=0)

    yhat = np.asarray(preds, dtype=float)
    if bool(normalize):
        yhat = yhat * std + mean
    return yhat


def torch_linear_attention_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    dim_feedforward: int = 256,
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
    Torch Transformer-like encoder with linear attention (direct multi-horizon).

    Uses a linear attention kernel (ELU+1) to reduce attention complexity from O(T^2) to O(T).
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    d = int(d_model)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if d <= 0:
        raise ValueError("d_model must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")
    if int(dim_feedforward) <= 0:
        raise ValueError("dim_feedforward must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _LinearAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d)
            self.out = nn.Linear(d, d)
            self.drop = nn.Dropout(p=drop)

        def forward(self, x: Any) -> Any:  # (B, T, d)
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)

            q = F.elu(q) + 1.0
            k = F.elu(k) + 1.0

            kv = torch.einsum("btd,btm->bdm", k, v)  # (B, d, d)
            numer = torch.einsum("btd,bdm->btm", q, kv)  # (B, T, d)

            k_sum = k.sum(dim=1)  # (B, d)
            denom = (q * k_sum.unsqueeze(1)).sum(dim=-1, keepdim=True) + 1e-6  # (B, T, 1)
            out = numer / denom
            return self.out(self.drop(out))

    class _LinAttnLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.attn = _LinearAttention()
            self.norm2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(
                nn.Linear(d, int(dim_feedforward)),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(int(dim_feedforward), d),
                nn.Dropout(p=drop),
            )

        def forward(self, x: Any) -> Any:
            x = x + self.attn(self.norm1(x))
            x = x + self.ff(self.norm2(x))
            return x

    class _LinearAttnDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
            self.layers = nn.ModuleList([_LinAttnLayer() for _ in range(int(num_layers))])
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            z = self.embed(xb) + self.pos
            for layer in self.layers:
                z = layer(z)
            last = z[:, -1, :]
            return self.head(last)

    model = _LinearAttnDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_inception_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    channels: int = 32,
    num_blocks: int = 3,
    kernel_sizes: Any = (3, 5, 7),
    bottleneck_channels: int = 16,
    dropout: float = 0.1,
    pool: str = "last",
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
    Torch InceptionTime-style Conv1D model (direct multi-horizon).

    Uses parallel convolutions with multiple kernel sizes per block.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    c = int(channels)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if c <= 0:
        raise ValueError("channels must be >= 1")
    if int(num_blocks) <= 0:
        raise ValueError("num_blocks must be >= 1")
    if int(bottleneck_channels) <= 0:
        raise ValueError("bottleneck_channels must be >= 1")

    if isinstance(kernel_sizes, int):
        ks = (int(kernel_sizes),)
    elif isinstance(kernel_sizes, str):
        parts = [p.strip() for p in kernel_sizes.split(",") if p.strip()]
        ks = tuple(int(p) for p in parts)
    elif isinstance(kernel_sizes, list | tuple):
        ks = tuple(int(k) for k in kernel_sizes)
    else:
        ks = (int(kernel_sizes),)

    if not ks or any(int(k) <= 0 for k in ks):
        raise ValueError("kernel_sizes must be a non-empty sequence of positive ints")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    pool_s = str(pool).lower().strip()
    if pool_s not in {"last", "mean", "max"}:
        raise ValueError("pool must be one of: last, mean, max")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _InceptionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            b = int(bottleneck_channels)
            self.bottleneck = nn.Conv1d(c, b, kernel_size=1)
            self.convs = nn.ModuleList(
                [nn.Conv1d(b, c, kernel_size=int(k), padding=int(k) // 2) for k in ks]
            )
            self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            self.pool_conv = nn.Conv1d(c, c, kernel_size=1)
            self.proj = nn.Conv1d(c * (len(ks) + 1), c, kernel_size=1)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

        def forward(self, xch: Any) -> Any:  # (B, c, T)
            z0 = self.bottleneck(xch)
            outs = [conv(z0) for conv in self.convs]
            outs.append(self.pool_conv(self.pool(xch)))
            z = torch.cat(outs, dim=1)
            z = self.act(self.proj(z))
            return self.drop(z)

    class _InceptionDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Conv1d(1, c, kernel_size=1)
            self.blocks = nn.ModuleList([_InceptionBlock() for _ in range(int(num_blocks))])
            self.head = nn.Linear(c, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)
            xch = xb.transpose(1, 2)
            z = self.in_proj(xch)
            for blk in self.blocks:
                z = z + blk(z)
            if pool_s == "last":
                feat = z[:, :, -1]
            elif pool_s == "mean":
                feat = z.mean(dim=2)
            else:
                feat = z.amax(dim=2)
            return self.head(feat)

    model = _InceptionDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_mamba_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    conv_kernel: int = 3,
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
    Mamba-style selective SSM (lite) for direct multi-horizon forecasting.

    Notes:
      - Uses a causal depthwise Conv1D for short-range mixing
      - Uses an input-dependent exponential decay state update (per channel)
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")

    d = int(d_model)
    if d <= 0:
        raise ValueError("d_model must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")
    k = int(conv_kernel)
    if k <= 0:
        raise ValueError("conv_kernel must be >= 1")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _MambaBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.u_proj = nn.Linear(d, d)
            self.delta_proj = nn.Linear(d, d)
            self.gate_proj = nn.Linear(d, d)
            self.log_A = nn.Parameter(torch.zeros((d,), dtype=torch.float32))
            self.dwconv = nn.Conv1d(d, d, kernel_size=int(k), groups=d)
            self.out_proj = nn.Linear(d, d)
            self.drop = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

        def forward(self, xb: Any) -> Any:  # (B, T, d)
            z = self.norm(xb)
            u = self.u_proj(z)  # (B, T, d)

            # Causal depthwise conv on u (short-range mixing).
            u_ch = u.transpose(1, 2)  # (B, d, T)
            u_pad = F.pad(u_ch, (int(k) - 1, 0))
            u = self.dwconv(u_pad).transpose(1, 2)  # (B, T, d)

            delta = F.softplus(self.delta_proj(z))  # (B, T, d) in (0, +inf)
            a = torch.exp(-delta * F.softplus(self.log_A).reshape(1, 1, -1))  # (B, T, d)
            g = torch.sigmoid(self.gate_proj(z))  # (B, T, d)

            B = int(u.shape[0])
            T = int(u.shape[1])
            s = torch.zeros((B, d), device=u.device, dtype=u.dtype)
            outs: list[Any] = []
            for t in range(T):
                at = a[:, t, :]
                ut = u[:, t, :]
                s = at * s + (1.0 - at) * ut
                y = self.out_proj(s)
                outs.append(g[:, t, :] * y + (1.0 - g[:, t, :]) * ut)

            y_seq = torch.stack(outs, dim=1)
            y_seq = self.drop(y_seq)
            return xb + y_seq

    class _MambaDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
            self.layers = nn.ModuleList([_MambaBlock() for _ in range(int(num_layers))])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)
            z = self.embed(xb) + self.pos
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            last = z[:, -1, :]
            return self.head(last)

    model = _MambaDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_rwkv_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    ffn_dim: int = 128,
    dropout: float = 0.0,
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
    RWKV-style time-mix + channel-mix (lite) for direct multi-horizon forecasting.

    This is a minimal, fully self-contained implementation inspired by RWKV's
    recurrent attention (WKV) update, written with plain PyTorch ops.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")

    d = int(d_model)
    if d <= 0:
        raise ValueError("d_model must be >= 1")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")
    ffn = int(ffn_dim)
    if ffn <= 0:
        raise ValueError("ffn_dim must be >= 1")
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _RWKVBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.ln2 = nn.LayerNorm(d)

            self.time_mix_k = nn.Parameter(torch.randn((d,), dtype=torch.float32) * 0.02)
            self.time_mix_v = nn.Parameter(torch.randn((d,), dtype=torch.float32) * 0.02)
            self.time_mix_r = nn.Parameter(torch.randn((d,), dtype=torch.float32) * 0.02)

            self.time_decay = nn.Parameter(torch.zeros((d,), dtype=torch.float32))
            self.time_first = nn.Parameter(torch.zeros((d,), dtype=torch.float32))

            self.key = nn.Linear(d, d, bias=False)
            self.value = nn.Linear(d, d, bias=False)
            self.receptance = nn.Linear(d, d, bias=False)
            self.output = nn.Linear(d, d, bias=False)

            self.channel_mix_k = nn.Parameter(torch.randn((d,), dtype=torch.float32) * 0.02)
            self.channel_mix_r = nn.Parameter(torch.randn((d,), dtype=torch.float32) * 0.02)
            self.key_ffn = nn.Linear(d, ffn, bias=False)
            self.value_ffn = nn.Linear(ffn, d, bias=False)
            self.receptance_ffn = nn.Linear(d, d, bias=False)

            self.drop_time = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()
            self.drop_ffn = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

        def _time_mix(self, x: Any) -> Any:  # (B, T, d)
            z = self.ln1(x)
            B = int(z.shape[0])
            T = int(z.shape[1])

            mk = torch.sigmoid(self.time_mix_k).reshape(1, -1)
            mv = torch.sigmoid(self.time_mix_v).reshape(1, -1)
            mr = torch.sigmoid(self.time_mix_r).reshape(1, -1)

            td = (-F.softplus(self.time_decay)).reshape(1, -1)  # negative
            tf = self.time_first.reshape(1, -1)

            prev = torch.zeros((B, d), device=z.device, dtype=z.dtype)
            aa = torch.zeros((B, d), device=z.device, dtype=z.dtype)
            bb = torch.zeros((B, d), device=z.device, dtype=z.dtype)
            pp = torch.full((B, d), -1e9, device=z.device, dtype=z.dtype)

            outs: list[Any] = []
            for t in range(T):
                xt = z[:, t, :]
                xk = xt * mk + prev * (1.0 - mk)
                xv = xt * mv + prev * (1.0 - mv)
                xr = xt * mr + prev * (1.0 - mr)
                prev = xt

                k = self.key(xk)
                v = self.value(xv)
                r = torch.sigmoid(self.receptance(xr))

                ww = k + tf
                p = torch.maximum(pp, ww)
                e1 = torch.exp(pp - p)
                e2 = torch.exp(ww - p)
                a = e1 * aa + e2 * v
                b = e1 * bb + e2
                wkv = a / (b + 1e-9)
                aa = a
                bb = b
                pp = p + td

                outs.append(self.output(r * wkv))

            y = torch.stack(outs, dim=1)
            return self.drop_time(y)

        def _channel_mix(self, x: Any) -> Any:  # (B, T, d)
            z = self.ln2(x)
            B = int(z.shape[0])
            prev = torch.cat(
                [torch.zeros((B, 1, d), device=z.device, dtype=z.dtype), z[:, :-1, :]], dim=1
            )

            mk = torch.sigmoid(self.channel_mix_k).reshape(1, 1, -1)
            mr = torch.sigmoid(self.channel_mix_r).reshape(1, 1, -1)
            xk = z * mk + prev * (1.0 - mk)
            xr = z * mr + prev * (1.0 - mr)

            k = self.key_ffn(xk)
            k = F.relu(k) ** 2
            v = self.value_ffn(k)
            r = torch.sigmoid(self.receptance_ffn(xr))
            y = r * v
            return self.drop_ffn(y)

        def forward(self, x: Any) -> Any:
            x = x + self._time_mix(x)
            x = x + self._channel_mix(x)
            return x

    class _RWKVDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
            self.layers = nn.ModuleList([_RWKVBlock() for _ in range(int(num_layers))])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)
            z = self.embed(xb) + self.pos
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            last = z[:, -1, :]
            return self.head(last)

    model = _RWKVDirect()
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
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)
