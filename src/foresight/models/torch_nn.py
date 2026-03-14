from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

_HIDDEN_SIZE_MIN_MSG = "hidden_size must be >= 1"
_NUM_LAYERS_MIN_MSG = "num_layers must be >= 1"
_DROPOUT_RANGE_MSG = "dropout must be in [0, 1)"
_HORIZON_MIN_MSG = "horizon must be >= 1"
_LAGS_MIN_MSG = "lags must be >= 1"
_NUM_BLOCKS_MIN_MSG = "num_blocks must be >= 1"
_D_MODEL_MIN_MSG = "d_model must be >= 1"
_TOP_K_MIN_MSG = "top_k must be >= 1"
_NHEAD_MIN_MSG = "nhead must be >= 1"
_D_MODEL_HEAD_DIVISIBILITY_MSG = "d_model must be divisible by nhead"
_DIM_FEEDFORWARD_MIN_MSG = "dim_feedforward must be >= 1"
_PATCH_LEN_MIN_MSG = "patch_len must be >= 1"
_SEGMENT_LEN_MIN_MSG = "segment_len must be >= 1"
_SEGMENT_LEN_MAX_LAGS_MSG = "segment_len must be <= lags"
_INPUT_SIZE_MIN_MSG = "input_size must be >= 1"
_KERNEL_SIZE_MIN_MSG = "kernel_size must be >= 1"
_MA_WINDOW_MIN_MSG = "ma_window must be >= 2"
_STRIDE_MIN_MSG = "stride must be >= 1"
_POOL_MODE_MSG = "pool must be one of: last, mean, max"
_CHANNELS_MIN_MSG = "channels must be >= 1"
_FFN_DIM_MIN_MSG = "ffn_dim must be >= 1"


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


def _make_manual_gru_cell(*, input_size: int, hidden_size: int) -> Any:
    in_dim = int(input_size)
    hid = int(hidden_size)
    if in_dim <= 0:
        raise ValueError(_INPUT_SIZE_MIN_MSG)
    if hid <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)

    torch = _require_torch()
    nn = torch.nn

    class _ManualGRUCell(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(in_dim, 3 * hid, bias=True)
            self.h2h = nn.Linear(hid, 3 * hid, bias=False)

        def forward(self, x_t: Any, h_prev: Any) -> Any:
            gi = self.x2h(x_t)
            gh = self.h2h(h_prev)
            i_r, i_z, i_n = gi.chunk(3, dim=-1)
            h_r, h_z, h_n = gh.chunk(3, dim=-1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            return (1.0 - z) * n + z * h_prev

    return _ManualGRUCell()


def _make_manual_lstm_cell(*, input_size: int, hidden_size: int) -> Any:
    in_dim = int(input_size)
    hid = int(hidden_size)
    if in_dim <= 0:
        raise ValueError(_INPUT_SIZE_MIN_MSG)
    if hid <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)

    torch = _require_torch()
    nn = torch.nn

    class _ManualLSTMCell(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(in_dim, 4 * hid, bias=True)
            self.h2h = nn.Linear(hid, 4 * hid, bias=False)

        def forward(self, x_t: Any, state: tuple[Any, Any]) -> tuple[Any, Any]:
            h_prev, c_prev = state
            gates = self.x2h(x_t) + self.h2h(h_prev)
            i, f, g, o = gates.chunk(4, dim=-1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c_t = f * c_prev + i * g
            h_t = o * torch.tanh(c_t)
            return h_t, c_t

    return _ManualLSTMCell()


def _make_manual_gru(
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float = 0.0,
    bidirectional: bool = False,
) -> Any:
    in_dim = int(input_size)
    hid = int(hidden_size)
    layers = int(num_layers)
    drop = float(dropout)
    bidir = bool(bidirectional)
    if in_dim <= 0:
        raise ValueError(_INPUT_SIZE_MIN_MSG)
    if hid <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    class _GRULayer(nn.Module):
        def __init__(self, *, in_dim: int) -> None:
            super().__init__()
            self.x2h = nn.Linear(int(in_dim), 3 * hid, bias=True)
            self.h2h = nn.Linear(hid, 3 * hid, bias=False)

        def step(self, x_t: Any, h_prev: Any) -> Any:
            gi = self.x2h(x_t)
            gh = self.h2h(h_prev)
            i_r, i_z, i_n = gi.chunk(3, dim=-1)
            h_r, h_z, h_n = gh.chunk(3, dim=-1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            return (1.0 - z) * n + z * h_prev

    class _ManualGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_size = in_dim
            self.hidden_size = hid
            self.num_layers = layers
            self.dropout = drop
            self.bidirectional = bidir

            self.fwd = nn.ModuleList()
            self.bwd = nn.ModuleList() if self.bidirectional else None

            layer_in = int(self.input_size)
            for _layer in range(int(self.num_layers)):
                self.fwd.append(_GRULayer(in_dim=int(layer_in)))
                if self.bidirectional:
                    assert self.bwd is not None
                    self.bwd.append(_GRULayer(in_dim=int(layer_in)))
                    layer_in = 2 * int(self.hidden_size)
                else:
                    layer_in = int(self.hidden_size)

        def _init_h0(self, xb: Any, hx: Any | None) -> list[Any]:
            B = int(xb.shape[0])
            dirs = 2 if self.bidirectional else 1
            if hx is None:
                return [
                    xb.new_zeros((B, int(self.hidden_size)))
                    for _ in range(int(self.num_layers) * dirs)
                ]
            h0 = hx
            expected = (int(self.num_layers) * dirs, B, int(self.hidden_size))
            if tuple(h0.shape) != expected:
                raise ValueError(f"Expected h0 shape {expected}, got {tuple(h0.shape)}")
            return [h0[i] for i in range(expected[0])]

        def forward(self, xb: Any, hx: Any | None = None) -> tuple[Any, Any]:
            if xb.ndim != 3:
                raise ValueError("Expected xb with shape (B, T, C)")
            if int(xb.shape[2]) != int(self.input_size):
                raise ValueError(
                    f"Expected input_size={int(self.input_size)}, got C={int(xb.shape[2])}"
                )

            B, T, _C = xb.shape
            if int(T) <= 0:
                raise ValueError("Sequence length T must be >= 1")

            dirs = 2 if self.bidirectional else 1
            h0_list = self._init_h0(xb, hx)

            x = xb
            h_n_out: list[Any] = []
            for layer in range(int(self.num_layers)):
                fwd_layer = self.fwd[layer]
                h_f = h0_list[layer * dirs + 0]
                outs_f: list[Any] = []
                for t in range(int(T)):
                    h_f = fwd_layer.step(x[:, t, :], h_f)
                    outs_f.append(h_f)
                out_f = torch.stack(outs_f, dim=1)

                if self.bidirectional:
                    assert self.bwd is not None
                    bwd_layer = self.bwd[layer]
                    h_b = h0_list[layer * dirs + 1]
                    outs_b_rev: list[Any] = []
                    for t in range(int(T) - 1, -1, -1):
                        h_b = bwd_layer.step(x[:, t, :], h_b)
                        outs_b_rev.append(h_b)
                    outs_b_rev.reverse()
                    out_b = torch.stack(outs_b_rev, dim=1)
                    out = torch.cat([out_f, out_b], dim=-1)
                    h_n_out.extend([h_f, h_b])
                else:
                    out = out_f
                    h_n_out.append(h_f)

                if layer < int(self.num_layers) - 1 and float(self.dropout) > 0.0 and self.training:
                    out = F.dropout(out, p=float(self.dropout), training=True)
                x = out

            h_n = torch.stack(h_n_out, dim=0)
            return x, h_n

    return _ManualGRU()


def _make_manual_lstm(
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float = 0.0,
    bidirectional: bool = False,
) -> Any:
    in_dim = int(input_size)
    hid = int(hidden_size)
    layers = int(num_layers)
    drop = float(dropout)
    bidir = bool(bidirectional)
    if in_dim <= 0:
        raise ValueError(_INPUT_SIZE_MIN_MSG)
    if hid <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    class _LSTMLayer(nn.Module):
        def __init__(self, *, in_dim: int) -> None:
            super().__init__()
            self.x2h = nn.Linear(int(in_dim), 4 * hid, bias=True)
            self.h2h = nn.Linear(hid, 4 * hid, bias=False)

        def step(self, x_t: Any, state: tuple[Any, Any]) -> tuple[Any, Any]:
            h_prev, c_prev = state
            gates = self.x2h(x_t) + self.h2h(h_prev)
            i, f, g, o = gates.chunk(4, dim=-1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c_t = f * c_prev + i * g
            h_t = o * torch.tanh(c_t)
            return h_t, c_t

    class _ManualLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_size = in_dim
            self.hidden_size = hid
            self.num_layers = layers
            self.dropout = drop
            self.bidirectional = bidir

            self.fwd = nn.ModuleList()
            self.bwd = nn.ModuleList() if self.bidirectional else None

            layer_in = int(self.input_size)
            for _layer in range(int(self.num_layers)):
                self.fwd.append(_LSTMLayer(in_dim=int(layer_in)))
                if self.bidirectional:
                    assert self.bwd is not None
                    self.bwd.append(_LSTMLayer(in_dim=int(layer_in)))
                    layer_in = 2 * int(self.hidden_size)
                else:
                    layer_in = int(self.hidden_size)

        def _init_state(self, xb: Any, hx: tuple[Any, Any] | None) -> tuple[list[Any], list[Any]]:
            B = int(xb.shape[0])
            dirs = 2 if self.bidirectional else 1
            if hx is None:
                h0 = [
                    xb.new_zeros((B, int(self.hidden_size)))
                    for _ in range(int(self.num_layers) * dirs)
                ]
                c0 = [
                    xb.new_zeros((B, int(self.hidden_size)))
                    for _ in range(int(self.num_layers) * dirs)
                ]
                return h0, c0

            h0_t, c0_t = hx
            expected = (int(self.num_layers) * dirs, B, int(self.hidden_size))
            if tuple(h0_t.shape) != expected:
                raise ValueError(f"Expected h0 shape {expected}, got {tuple(h0_t.shape)}")
            if tuple(c0_t.shape) != expected:
                raise ValueError(f"Expected c0 shape {expected}, got {tuple(c0_t.shape)}")
            h0 = [h0_t[i] for i in range(expected[0])]
            c0 = [c0_t[i] for i in range(expected[0])]
            return h0, c0

        def forward(
            self, xb: Any, hx: tuple[Any, Any] | None = None
        ) -> tuple[Any, tuple[Any, Any]]:
            if xb.ndim != 3:
                raise ValueError("Expected xb with shape (B, T, C)")
            if int(xb.shape[2]) != int(self.input_size):
                raise ValueError(
                    f"Expected input_size={int(self.input_size)}, got C={int(xb.shape[2])}"
                )

            B, T, _C = xb.shape
            if int(T) <= 0:
                raise ValueError("Sequence length T must be >= 1")

            dirs = 2 if self.bidirectional else 1
            h0_list, c0_list = self._init_state(xb, hx)

            x = xb
            h_n_out: list[Any] = []
            c_n_out: list[Any] = []
            for layer in range(int(self.num_layers)):
                fwd_layer = self.fwd[layer]
                h_f = h0_list[layer * dirs + 0]
                c_f = c0_list[layer * dirs + 0]
                outs_f: list[Any] = []
                for t in range(int(T)):
                    h_f, c_f = fwd_layer.step(x[:, t, :], (h_f, c_f))
                    outs_f.append(h_f)
                out_f = torch.stack(outs_f, dim=1)

                if self.bidirectional:
                    assert self.bwd is not None
                    bwd_layer = self.bwd[layer]
                    h_b = h0_list[layer * dirs + 1]
                    c_b = c0_list[layer * dirs + 1]
                    outs_b_rev: list[Any] = []
                    for t in range(int(T) - 1, -1, -1):
                        h_b, c_b = bwd_layer.step(x[:, t, :], (h_b, c_b))
                        outs_b_rev.append(h_b)
                    outs_b_rev.reverse()
                    out_b = torch.stack(outs_b_rev, dim=1)
                    out = torch.cat([out_f, out_b], dim=-1)
                    h_n_out.extend([h_f, h_b])
                    c_n_out.extend([c_f, c_b])
                else:
                    out = out_f
                    h_n_out.append(h_f)
                    c_n_out.append(c_f)

                if layer < int(self.num_layers) - 1 and float(self.dropout) > 0.0 and self.training:
                    out = F.dropout(out, p=float(self.dropout), training=True)
                x = out

            h_n = torch.stack(h_n_out, dim=0)
            c_n = torch.stack(c_n_out, dim=0)
            return x, (h_n, c_n)

    return _ManualLSTM()


def _make_lagged_xy_multi(
    x: np.ndarray, *, lags: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.size)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
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


def _as_2d_float_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/inf")
    return arr


def _normalize_exog_pair(
    train_exog: np.ndarray, future_exog: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_exog, axis=0, dtype=float)
    std = np.std(train_exog, axis=0, dtype=float)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        (train_exog - mean.reshape(1, -1)) / std.reshape(1, -1),
        (future_exog - mean.reshape(1, -1)) / std.reshape(1, -1),
    )


def _make_lagged_xy_exog_seq(
    y: np.ndarray,
    exog: np.ndarray,
    *,
    lags: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(y.size)
    h = int(horizon)
    lag_count = int(lags)
    if exog.ndim != 2:
        raise ValueError(f"exog must be 2D, got shape {exog.shape}")
    if int(exog.shape[0]) != n:
        raise ValueError("exog rows must match target length")
    if n < lag_count + h:
        raise ValueError(f"Need >= lags+horizon points (lags={lag_count}, horizon={h}), got {n}")

    rows = n - lag_count - h + 1
    x_dim = int(exog.shape[1])
    X = np.empty((rows, lag_count + h, 1 + x_dim), dtype=float)
    Y = np.empty((rows, h), dtype=float)
    for i in range(rows):
        t = i + lag_count
        y_past = y[t - lag_count : t]
        x_past = exog[t - lag_count : t, :]
        x_future = exog[t : t + h, :]
        y_feat = np.concatenate([y_past, np.zeros((h,), dtype=float)], axis=0).reshape(-1, 1)
        x_feat = np.concatenate([x_past, x_future], axis=0)
        X[i] = np.concatenate([y_feat, x_feat], axis=1)
        Y[i] = y[t : t + h]
    return X, Y


def _make_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((int(seq_len), int(d_model)), dtype=float)
    position = np.arange(int(seq_len), dtype=float).reshape(-1, 1)
    div_term = np.exp(
        np.arange(0, int(d_model), 2, dtype=float) * (-math.log(10000.0) / float(d_model))
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


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


def _moving_average_1d(xb: Any, *, window: int) -> Any:
    torch = _require_torch()
    F = torch.nn.functional

    k = int(window)
    if k <= 0:
        raise ValueError("window must be >= 1")

    left = k // 2
    right = k - 1 - left
    x1 = xb.unsqueeze(1)
    xpad = F.pad(x1, (left, right), mode="replicate")
    weight = torch.ones((1, 1, k), device=xb.device, dtype=xb.dtype) / float(k)
    return F.conv1d(xpad, weight).squeeze(1)


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


def _fit_encoder_direct_model(
    train: Any,
    horizon: int,
    *,
    lags: int,
    build_model: Callable[[int, int], Any],
    normalize: bool,
    device: str,
    cfg: TorchTrainConfig,
) -> np.ndarray:
    torch = _require_torch()

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_model(lag_count, h)
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def _make_grn(d_in: int, d_hidden: int, d_out: int | None = None, dropout: float = 0.0) -> Any:
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    d_in_i = int(d_in)
    d_hidden_i = int(d_hidden)
    d_out_i = int(d_in_i) if d_out is None else int(d_out)

    class _GRN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d_in_i, d_hidden_i)
            self.fc2 = nn.Linear(d_hidden_i, d_out_i)
            self.gate = nn.Linear(d_out_i, d_out_i)
            self.dropout = nn.Dropout(float(dropout))
            self.skip = nn.Identity() if d_in_i == d_out_i else nn.Linear(d_in_i, d_out_i)
            self.norm = nn.LayerNorm(d_out_i)

        def forward(self, x: Any) -> Any:
            h = F.elu(self.fc1(x))
            h = self.fc2(h)
            h = self.dropout(h)
            g = torch.sigmoid(self.gate(h))
            return self.norm(self.skip(x) + g * h)

    return _GRN()


def _make_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((int(seq_len), int(d_model)), dtype=float)
    pos = np.arange(int(seq_len), dtype=float).reshape(-1, 1)
    div = np.exp(np.arange(0, int(d_model), 2, dtype=float) * (-math.log(10000.0) / float(d_model)))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


def _moving_average_1d(y: Any, *, window: int) -> Any:
    torch = _require_torch()
    F = torch.nn.functional

    k = int(window)
    if k <= 0:
        raise ValueError("window must be >= 1")

    left = k // 2
    right = k - 1 - left
    y_in = y.unsqueeze(1)
    y_pad = F.pad(y_in, (left, right), mode="replicate")
    return F.avg_pool1d(y_pad, kernel_size=k, stride=1).squeeze(1)


def _parse_int_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        items = (int(value),)
    elif isinstance(value, str):
        parts = [p.strip() for p in str(value).split(",") if p.strip()]
        items = tuple(int(p) for p in parts)
    elif isinstance(value, list | tuple):
        items = tuple(int(v) for v in value)
    else:
        items = (int(value),)

    if not items or any(int(v) <= 0 for v in items):
        raise ValueError(f"{name} must be a non-empty sequence of positive ints")
    return items


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
        raise ValueError(_HORIZON_MIN_MSG)

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
        raise ValueError(_DROPOUT_RANGE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if int(hidden_size) <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=int(lags), horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    class _LSTMDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rnn_drop = drop if int(num_layers) > 1 else 0.0
            self.lstm = _make_manual_lstm(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=float(rnn_drop),
                bidirectional=False,
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
        raise ValueError(_HORIZON_MIN_MSG)
    if int(hidden_size) <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=int(lags), horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    class _GRUDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rnn_drop = drop if int(num_layers) > 1 else 0.0
            self.gru = _make_manual_gru(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=float(rnn_drop),
                bidirectional=False,
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
        raise ValueError(_HORIZON_MIN_MSG)
    if int(lags) <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if int(kernel_size) <= 0:
        raise ValueError(_KERNEL_SIZE_MIN_MSG)

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
        raise ValueError(_DROPOUT_RANGE_MSG)

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
    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if int(num_blocks) <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(layer_width) <= 0:
        raise ValueError("layer_width must be >= 1")

    torch = _require_torch()
    nn = torch.nn

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if int(ma_window) <= 1:
        raise ValueError(_MA_WINDOW_MIN_MSG)

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
    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    torch = _require_torch()
    nn = torch.nn

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


def torch_informer_direct_forecast(
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
    Torch Informer-style encoder (lite) direct multi-horizon forecast on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn

    d = int(d_model)
    heads = int(nhead)
    layers = int(num_layers)
    ff_dim = int(dim_feedforward)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if ff_dim <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

    def _build_model(lag_count: int, h: int) -> Any:
        pe = _make_positional_encoding(lag_count, d)

        class _InformerDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.register_buffer("pe", torch.tensor(pe, dtype=torch.float32), persistent=False)
                layer = nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=heads,
                    dim_feedforward=ff_dim,
                    dropout=drop,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.enc = nn.TransformerEncoder(layer, num_layers=layers)
                self.head = nn.Linear(d, h)

            def forward(self, xb: Any) -> Any:
                z = self.in_proj(xb) + self.pe.unsqueeze(0)
                z = self.enc(z)
                return self.head(z[:, -1, :])

        return _InformerDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_autoformer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    ma_window: int = 7,
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
    Torch Autoformer-style decomposition encoder (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    d = int(d_model)
    heads = int(nhead)
    layers = int(num_layers)
    ff_dim = int(dim_feedforward)
    ma = int(ma_window)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if ff_dim <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)
    if ma <= 1:
        raise ValueError(_MA_WINDOW_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

    def _build_model(lag_count: int, h: int) -> Any:
        pe = _make_positional_encoding(lag_count, d)

        class _AutoformerDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.register_buffer("pe", torch.tensor(pe, dtype=torch.float32), persistent=False)
                layer = nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=heads,
                    dim_feedforward=ff_dim,
                    dropout=drop,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.enc = nn.TransformerEncoder(layer, num_layers=layers)
                self.seasonal_head = nn.Linear(d, h)
                self.trend_proj = nn.Linear(lag_count, h)

            def forward(self, xb: Any) -> Any:
                y_ctx = xb[:, :, 0]
                trend = _moving_average_1d(y_ctx, window=ma)
                seasonal = y_ctx - trend
                z = self.in_proj(seasonal.unsqueeze(-1)) + self.pe.unsqueeze(0)
                z = self.enc(z)
                seasonal_hat = self.seasonal_head(z[:, -1, :])
                trend_hat = self.trend_proj(trend)
                return seasonal_hat + trend_hat

        return _AutoformerDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_nonstationary_transformer_direct_forecast(
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
    Torch Non-stationary Transformer-style model (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    d = int(d_model)
    heads = int(nhead)
    layers = int(num_layers)
    ff_dim = int(dim_feedforward)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if ff_dim <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    head_dim = d // heads
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

    def _build_model(lag_count: int, h: int) -> Any:
        pe = _make_positional_encoding(lag_count, d)

        class _DSFactors(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                hidden = max(8, d)
                self.mlp = nn.Sequential(
                    nn.Linear(2, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 2 * heads),
                )

            def forward(self, mu: Any, sigma: Any) -> tuple[Any, Any]:
                feats = torch.stack([mu, sigma], dim=-1)
                out = self.mlp(feats)
                tau_raw, delta = out.chunk(2, dim=-1)
                tau = F.softplus(tau_raw) + 1e-3
                return tau, delta

        class _NSAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qkv = nn.Linear(d, 3 * d)
                self.out = nn.Linear(d, d)
                self.drop = nn.Dropout(p=drop)

            def forward(self, xb: Any, tau: Any, delta: Any) -> Any:
                qkv = self.qkv(xb)
                q, k, v = qkv.chunk(3, dim=-1)

                def _reshape(z: Any) -> Any:
                    return z.reshape(z.shape[0], z.shape[1], heads, head_dim).permute(0, 2, 1, 3)

                qh = _reshape(q)
                kh = _reshape(k)
                vh = _reshape(v)
                scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(float(head_dim))
                scores = scores * tau.reshape(tau.shape[0], heads, 1, 1)
                scores = scores + delta.reshape(delta.shape[0], heads, 1, 1)
                attn = torch.softmax(scores, dim=-1)
                attn = self.drop(attn)
                out = attn @ vh
                out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], d)
                return self.out(out)

        class _NSTBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(d)
                self.attn = _NSAttention()
                self.norm2 = nn.LayerNorm(d)
                self.ffn = nn.Sequential(
                    nn.Linear(d, ff_dim),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(ff_dim, d),
                )
                self.drop = nn.Dropout(p=drop)

            def forward(self, xb: Any, tau: Any, delta: Any) -> Any:
                xb = xb + self.drop(self.attn(self.norm1(xb), tau, delta))
                xb = xb + self.drop(self.ffn(self.norm2(xb)))
                return xb

        class _NSTDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.register_buffer("pe", torch.tensor(pe, dtype=torch.float32), persistent=False)
                self.ds = _DSFactors()
                self.blocks = nn.ModuleList([_NSTBlock() for _ in range(layers)])
                self.head = nn.Linear(d, h)

            def forward(self, xb: Any) -> Any:
                y_ctx = xb[:, :, 0]
                mu = torch.mean(y_ctx, dim=1)
                sigma = torch.sqrt(torch.mean((y_ctx - mu.unsqueeze(1)) ** 2, dim=1) + 1e-6)
                x_norm = ((y_ctx - mu.unsqueeze(1)) / sigma.unsqueeze(1)).unsqueeze(-1)
                tau, delta = self.ds(mu, sigma)
                z = self.in_proj(x_norm) + self.pe.unsqueeze(0)
                for blk in self.blocks:
                    z = blk(z, tau, delta)
                yhat_norm = self.head(z[:, -1, :])
                return yhat_norm * sigma.unsqueeze(1) + mu.unsqueeze(1)

        return _NSTDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_fedformer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    ffn_dim: int = 256,
    modes: int = 16,
    ma_window: int = 7,
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
    Torch FEDformer-style decomposition + frequency-mixing model (lite) direct forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    d = int(d_model)
    layers = int(num_layers)
    ff_dim = int(ffn_dim)
    mode_count = int(modes)
    ma = int(ma_window)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if ff_dim <= 0:
        raise ValueError(_FFN_DIM_MIN_MSG)
    if mode_count <= 0:
        raise ValueError("modes must be >= 1")
    if ma <= 1:
        raise ValueError(_MA_WINDOW_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

    def _build_model(lag_count: int, h: int) -> Any:
        pe = _make_positional_encoding(lag_count, d)

        class _FourierMix(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w_re = nn.Parameter(torch.randn(mode_count, d, dtype=torch.float32) * 0.02)
                self.w_im = nn.Parameter(torch.randn(mode_count, d, dtype=torch.float32) * 0.02)

            def forward(self, xb: Any) -> Any:
                xf = torch.fft.rfft(xb, dim=1)
                lf = int(xf.shape[1])
                m = min(mode_count, lf)
                w_c = torch.complex(self.w_re[:m, :], self.w_im[:m, :])
                out_f = torch.zeros_like(xf)
                out_f[:, :m, :] = xf[:, :m, :] * w_c.unsqueeze(0)
                return torch.fft.irfft(out_f, n=int(xb.shape[1]), dim=1)

        class _FEDformerBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(d)
                self.mix = _FourierMix()
                self.norm2 = nn.LayerNorm(d)
                self.ffn = nn.Sequential(
                    nn.Linear(d, ff_dim),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(ff_dim, d),
                )
                self.drop = nn.Dropout(p=drop)

            def forward(self, xb: Any) -> Any:
                xb = xb + self.drop(self.mix(self.norm1(xb)))
                xb = xb + self.drop(self.ffn(self.norm2(xb)))
                return xb

        class _FEDformerDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.register_buffer("pe", torch.tensor(pe, dtype=torch.float32), persistent=False)
                self.blocks = nn.ModuleList([_FEDformerBlock() for _ in range(layers)])
                self.seasonal_head = nn.Linear(d, h)
                self.trend_proj = nn.Linear(lag_count, h)

            def forward(self, xb: Any) -> Any:
                y_ctx = xb[:, :, 0]
                trend = _moving_average_1d(y_ctx, window=ma)
                seasonal = y_ctx - trend
                z = self.in_proj(seasonal.unsqueeze(-1)) + self.pe.unsqueeze(0)
                for blk in self.blocks:
                    z = blk(z)
                seasonal_hat = self.seasonal_head(z[:, -1, :])
                trend_hat = self.trend_proj(trend)
                return seasonal_hat + trend_hat

        return _FEDformerDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_itransformer_direct_forecast(
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
    Torch iTransformer-style inverted-token encoder (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    d = int(d_model)
    heads = int(nhead)
    layers = int(num_layers)
    ff_dim = int(dim_feedforward)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if ff_dim <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

    def _build_model(lag_count: int, h: int) -> Any:
        class _ITransformerDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(lag_count, d)
                self.token_pos = nn.Parameter(torch.zeros((1, 1, d), dtype=torch.float32))
                layer = nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=heads,
                    dim_feedforward=ff_dim,
                    dropout=drop,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.enc = nn.TransformerEncoder(layer, num_layers=layers)
                self.out = nn.Linear(d, h)

            def forward(self, xb: Any) -> Any:
                xinv = xb.transpose(1, 2)
                z = self.in_proj(xinv) + self.token_pos
                z = self.enc(z)
                return self.out(z[:, 0, :])

        return _ITransformerDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_timesnet_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    top_k: int = 3,
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
    Torch TimesNet-style period-mixing Conv2D model (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    d = int(d_model)
    layers = int(num_layers)
    k = int(top_k)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if k <= 0:
        raise ValueError(_TOP_K_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

    def _build_model(lag_count: int, h: int) -> Any:
        def _detect_periods(y_ctx: Any) -> tuple[list[int], Any]:
            amp = torch.fft.rfft(y_ctx, dim=1).abs().mean(dim=0)
            if int(amp.numel()) <= 1:
                w = torch.ones((1,), dtype=torch.float32, device=y_ctx.device)
                return [1], w

            amp = amp.to(dtype=torch.float32).clone()
            amp[0] = 0.0
            k_eff = min(k, int(amp.numel() - 1))
            vals, idx = torch.topk(amp, k=k_eff, largest=True)

            period_to_val: dict[int, float] = {}
            for f_i, v_i in zip(idx.tolist(), vals.tolist(), strict=True):
                f = int(f_i)
                if f <= 0:
                    continue
                p = int(round(float(lag_count) / float(f)))
                p = max(1, min(p, lag_count))
                prev = period_to_val.get(p)
                if prev is None or float(v_i) > float(prev):
                    period_to_val[p] = float(v_i)

            if not period_to_val:
                w = torch.ones((1,), dtype=torch.float32, device=y_ctx.device)
                return [1], w

            items = sorted(period_to_val.items(), key=lambda kv: kv[1], reverse=True)
            periods = [int(p) for p, _v in items]
            vals_t = torch.tensor(
                [float(v) for _p, v in items], dtype=torch.float32, device=y_ctx.device
            )
            weights = torch.softmax(vals_t, dim=0)
            return periods, weights

        class _TimesBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(d, d, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Conv2d(d, d, kernel_size=3, padding=1),
                    nn.Dropout(p=drop),
                )
                self.norm = nn.LayerNorm(d)

            def forward(self, z: Any, periods: list[int], weights: Any) -> Any:
                bsz, seq_len, _dim = z.shape
                out = torch.zeros_like(z)
                for w, p in zip(weights, periods, strict=True):
                    pp = int(p)
                    if pp <= 0:
                        continue
                    pad_len = (pp - (int(seq_len) % pp)) % pp
                    if pad_len:
                        pad = z[:, -1:, :].expand(-1, int(pad_len), -1)
                        z_pad = torch.cat([z, pad], dim=1)
                    else:
                        z_pad = z

                    full_len = int(z_pad.shape[1])
                    z2 = z_pad.reshape(int(bsz), full_len // pp, pp, d).permute(0, 3, 1, 2)
                    z2 = self.conv(z2)
                    z2 = z2.permute(0, 2, 3, 1).reshape(int(bsz), full_len, d)
                    out = out + w * z2[:, : int(seq_len), :]
                return self.norm(z + out)

        class _TimesNetDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.blocks = nn.ModuleList([_TimesBlock() for _ in range(layers)])
                self.head = nn.Linear(d, h)

            def forward(self, xb: Any) -> Any:
                y_ctx = xb[:, :, 0]
                with torch.no_grad():
                    periods, weights = _detect_periods(y_ctx)
                z = self.in_proj(xb)
                for blk in self.blocks:
                    z = blk(z, periods, weights)
                return self.head(z[:, -1, :])

        return _TimesNetDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_tft_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    lstm_layers: int = 1,
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
    Torch TFT-style (lite) direct multi-horizon forecast on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn

    d = int(d_model)
    heads = int(nhead)
    rnn_layers = int(lstm_layers)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if rnn_layers <= 0:
        raise ValueError("lstm_layers must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

    def _build_model(_lag_count: int, h: int) -> Any:
        rnn_drop = drop if rnn_layers > 1 else 0.0

        class _TFTDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.pre_grn = _make_grn(d, max(8, d), d, dropout=drop)
                self.lstm = _make_manual_lstm(
                    input_size=d,
                    hidden_size=d,
                    num_layers=rnn_layers,
                    dropout=rnn_drop,
                    bidirectional=False,
                )
                self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
                self.post_grn = _make_grn(d, max(8, d), d, dropout=drop)
                self.out = nn.Linear(d, h)

            def forward(self, xb: Any) -> Any:
                h0 = self.pre_grn(self.in_proj(xb))
                h1, _ = self.lstm(h0)
                attn, _ = self.attn(h1, h1, h1, need_weights=False)
                h2 = self.post_grn(h1 + attn)
                return self.out(h2[:, -1, :])

        return _TFTDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_timemixer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_blocks: int = 4,
    multiscale_factors: Any = (1, 2, 4),
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
    Torch TimeMixer-style (lite) direct multi-horizon forecast on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    d = int(d_model)
    blocks = int(num_blocks)
    token_hidden = int(token_mixing_hidden)
    channel_hidden = int(channel_mixing_hidden)
    scales = _parse_int_tuple(multiscale_factors, name="multiscale_factors")
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if blocks <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if token_hidden <= 0:
        raise ValueError("token_mixing_hidden must be >= 1")
    if channel_hidden <= 0:
        raise ValueError("channel_mixing_hidden must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

    def _build_model(lag_count: int, h: int) -> Any:
        class _MixerBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm_t = nn.LayerNorm(d)
                self.token_mlp = nn.Sequential(
                    nn.Linear(lag_count, token_hidden),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(token_hidden, lag_count),
                )
                self.norm_c = nn.LayerNorm(d)
                self.channel_mlp = nn.Sequential(
                    nn.Linear(d, channel_hidden),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(channel_hidden, d),
                )

            def forward(self, xb: Any) -> Any:
                z = self.norm_t(xb)
                xb = xb + self.token_mlp(z.transpose(1, 2)).transpose(1, 2)
                z = self.norm_c(xb)
                xb = xb + self.channel_mlp(z)
                return xb

        class _TimeMixerDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.scale_proj = nn.ModuleList([nn.Linear(d, d) for _ in scales])
                self.fuse = nn.Linear(d * len(scales), d)
                self.blocks = nn.ModuleList([_MixerBlock() for _ in range(blocks)])
                self.out = nn.Linear(d, h)

            def _smooth_scale(self, z: Any, factor: int) -> Any:
                if int(factor) <= 1:
                    return z
                left = int(factor) // 2
                right = int(factor) - 1 - left
                zc = z.transpose(1, 2)
                z_pad = F.pad(zc, (left, right), mode="replicate")
                return F.avg_pool1d(z_pad, kernel_size=int(factor), stride=1).transpose(1, 2)

            def forward(self, xb: Any) -> Any:
                base = self.in_proj(xb)
                multiscale = []
                for factor, proj in zip(scales, self.scale_proj, strict=True):
                    multiscale.append(proj(self._smooth_scale(base, int(factor))))
                z = self.fuse(torch.cat(multiscale, dim=-1))
                for blk in self.blocks:
                    z = blk(z)
                return self.out(z[:, -1, :])

        return _TimeMixerDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_sparsetsf_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    period_len: int = 24,
    d_model: int = 64,
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
    Torch SparseTSF-style (lite) direct multi-horizon forecast on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    period = int(period_len)
    d = int(d_model)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if period <= 0:
        raise ValueError("period_len must be >= 1")
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    start = lag_count % period
    sparse_idx = np.arange(start, lag_count, period, dtype=int)
    if sparse_idx.size == 0:
        sparse_idx = np.array([lag_count - 1], dtype=int)

    class _SparseTSFDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(int(sparse_idx.size), d)
            self.out = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:
            xs = xb[:, sparse_idx]
            z = F.gelu(self.proj(xs))
            return self.out(z)

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
    model = _train_loop(_SparseTSFDirect(), X, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, -1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_lightts_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    chunk_len: int = 12,
    d_model: int = 64,
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
    Torch LightTS-style (lite) dual-sampling MLP on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    chunk = int(chunk_len)
    d = int(d_model)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if chunk <= 0:
        raise ValueError("chunk_len must be >= 1")
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    n_chunks = int(math.ceil(float(lag_count) / float(chunk)))
    pad_len = int(n_chunks * chunk - lag_count)

    class _LightTSDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cont_proj = nn.Linear(n_chunks, d)
            self.inter_proj = nn.Linear(chunk, d)
            self.fuse = nn.Linear(2 * d, d)
            self.dropout = nn.Dropout(p=drop)
            self.out = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:
            if pad_len > 0:
                pad = xb[:, -1:].expand(-1, pad_len)
                xb = torch.cat([xb, pad], dim=1)
            chunks = xb.reshape(xb.shape[0], n_chunks, chunk)
            cont = chunks.mean(dim=2)
            inter = chunks.transpose(1, 2).mean(dim=2)
            zc = F.gelu(self.cont_proj(cont))
            zi = F.gelu(self.inter_proj(inter))
            z = self.dropout(F.gelu(self.fuse(torch.cat([zc, zi], dim=1))))
            return self.out(z)

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
    model = _train_loop(_LightTSDirect(), X, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, -1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_frets_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    top_k_freqs: int = 8,
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
    Torch FreTS-style (lite) frequency-domain MLP on lag windows.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    d = int(d_model)
    layers = int(num_layers)
    k = int(top_k_freqs)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if k <= 0:
        raise ValueError("top_k_freqs must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)

    class _FreTSDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k_eff = min(k, max(1, lag_count // 2))
            in_dim = 2 * self.k_eff
            layers_list: list[Any] = []
            cur = in_dim
            for _ in range(layers):
                layers_list.extend(
                    [
                        nn.Linear(cur, d),
                        nn.GELU(),
                        nn.Dropout(p=drop),
                    ]
                )
                cur = d
            self.backbone = nn.Sequential(*layers_list)
            self.out = nn.Linear(cur, h)

        def _freq_features(self, xb: Any) -> Any:
            xf = torch.fft.rfft(xb, dim=1)
            mag = xf.abs()
            if int(mag.shape[1]) <= 1:
                mag = torch.ones_like(mag)
            mag = mag.clone()
            mag[:, 0] = 0.0
            k_eff = min(self.k_eff, int(mag.shape[1] - 1))
            vals, idx = torch.topk(mag[:, 1:], k=k_eff, dim=1, largest=True)
            del vals
            idx = idx + 1
            batch = torch.arange(xb.shape[0], device=xb.device).unsqueeze(1)
            picked = xf[batch, idx]
            feat = torch.cat([picked.real, picked.imag], dim=1)
            if k_eff < self.k_eff:
                pad = feat.new_zeros((feat.shape[0], 2 * (self.k_eff - k_eff)))
                feat = torch.cat([feat, pad], dim=1)
            return feat

        def forward(self, xb: Any) -> Any:
            feat = self._freq_features(xb)
            z = self.backbone(feat)
            return self.out(z)

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
    model = _train_loop(_FreTSDirect(), X, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, -1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_film_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    ma_window: int = 7,
    kernel_size: int = 7,
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
    Torch FiLM-style decomposition + long-filter mixer (lite) direct forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    d = int(d_model)
    layers = int(num_layers)
    ma = int(ma_window)
    k = int(kernel_size)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if ma <= 1:
        raise ValueError(_MA_WINDOW_MIN_MSG)
    if k <= 0:
        raise ValueError(_KERNEL_SIZE_MIN_MSG)
    if k % 2 == 0:
        raise ValueError("kernel_size must be odd for same-length filtering")
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

    def _build_model(lag_count: int, h: int) -> Any:
        class _FiLMBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(d)
                self.dwconv = nn.Conv1d(d, d, kernel_size=k, padding=k // 2, groups=d)
                self.pwconv = nn.Conv1d(d, d, kernel_size=1)
                self.norm2 = nn.LayerNorm(d)
                self.ffn = nn.Sequential(
                    nn.Linear(d, 2 * d),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(2 * d, d),
                )
                self.drop = nn.Dropout(p=drop)

            def forward(self, xb: Any) -> Any:
                z = self.norm1(xb).transpose(1, 2)
                z = self.pwconv(self.dwconv(z)).transpose(1, 2)
                xb = xb + self.drop(z)
                xb = xb + self.drop(self.ffn(self.norm2(xb)))
                return xb

        class _FiLMDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.blocks = nn.ModuleList([_FiLMBlock() for _ in range(layers)])
                self.out = nn.Linear(d, h)
                self.trend_proj = nn.Linear(lag_count, h)

            def forward(self, xb: Any) -> Any:
                y_ctx = xb[:, :, 0]
                trend = _moving_average_1d(y_ctx, window=ma)
                seasonal = y_ctx - trend
                z = self.in_proj(seasonal.unsqueeze(-1))
                for blk in self.blocks:
                    z = blk(z)
                pooled = z.mean(dim=1)
                seasonal_hat = self.out(pooled)
                trend_hat = self.trend_proj(trend)
                return seasonal_hat + trend_hat

        return _FiLMDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_micn_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    kernel_sizes: Any = (3, 5, 7),
    ma_window: int = 7,
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
    Torch MICN-style multiscale convolutional decomposition model (lite) direct forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    d = int(d_model)
    layers = int(num_layers)
    ma = int(ma_window)
    ks = _parse_int_tuple(kernel_sizes, name="kernel_sizes")
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if ma <= 1:
        raise ValueError(_MA_WINDOW_MIN_MSG)
    if any(int(v) % 2 == 0 for v in ks):
        raise ValueError("kernel_sizes must be odd for same-length convolutions")
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

    def _build_model(lag_count: int, h: int) -> Any:
        class _MICNBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(d)
                self.branches = nn.ModuleList(
                    [nn.Conv1d(d, d, kernel_size=int(v), padding=int(v) // 2) for v in ks]
                )
                self.proj = nn.Conv1d(d * len(ks), d, kernel_size=1)
                self.norm2 = nn.LayerNorm(d)
                self.ffn = nn.Sequential(
                    nn.Linear(d, 2 * d),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(2 * d, d),
                )
                self.drop = nn.Dropout(p=drop)

            def forward(self, xb: Any) -> Any:
                z = self.norm1(xb).transpose(1, 2)
                merged = torch.cat([branch(z) for branch in self.branches], dim=1)
                merged = self.proj(merged).transpose(1, 2)
                xb = xb + self.drop(merged)
                xb = xb + self.drop(self.ffn(self.norm2(xb)))
                return xb

        class _MICNDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.blocks = nn.ModuleList([_MICNBlock() for _ in range(layers)])
                self.out = nn.Linear(d, h)
                self.trend_proj = nn.Linear(lag_count, h)

            def forward(self, xb: Any) -> Any:
                y_ctx = xb[:, :, 0]
                trend = _moving_average_1d(y_ctx, window=ma)
                seasonal = y_ctx - trend
                z = self.in_proj(seasonal.unsqueeze(-1))
                for blk in self.blocks:
                    z = blk(z)
                pooled = z.mean(dim=1)
                seasonal_hat = self.out(pooled)
                trend_hat = self.trend_proj(trend)
                return seasonal_hat + trend_hat

        return _MICNDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_koopa_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    latent_dim: int = 32,
    num_blocks: int = 2,
    ma_window: int = 7,
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
    Torch Koopa-style decomposition + latent linear dynamics model (lite) direct forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    d = int(d_model)
    latent = int(latent_dim)
    blocks = int(num_blocks)
    ma = int(ma_window)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if latent <= 0:
        raise ValueError("latent_dim must be >= 1")
    if blocks <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if ma <= 1:
        raise ValueError(_MA_WINDOW_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

    def _build_model(lag_count: int, h: int) -> Any:
        class _KoopaBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(d)
                self.ffn = nn.Sequential(
                    nn.Linear(d, 2 * d),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(2 * d, d),
                )
                self.drop = nn.Dropout(p=drop)

            def forward(self, xb: Any) -> Any:
                return xb + self.drop(self.ffn(self.norm(xb)))

        class _KoopaDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.blocks = nn.ModuleList([_KoopaBlock() for _ in range(blocks)])
                self.to_latent = nn.Linear(d, latent)
                self.transition = nn.Parameter(torch.randn(latent, latent) * 0.02)
                self.decoder = nn.Linear(latent, 1)
                self.trend_proj = nn.Linear(lag_count, h)

            def forward(self, xb: Any) -> Any:
                y_ctx = xb[:, :, 0]
                trend = _moving_average_1d(y_ctx, window=ma)
                seasonal = y_ctx - trend
                z = self.in_proj(seasonal.unsqueeze(-1))
                for blk in self.blocks:
                    z = blk(z)
                state = self.to_latent(z.mean(dim=1))
                trans = torch.eye(latent, device=state.device, dtype=state.dtype) + torch.tanh(
                    self.transition
                ) / math.sqrt(float(latent))
                outs: list[Any] = []
                cur = state
                for _ in range(h):
                    cur = cur @ trans
                    outs.append(self.decoder(cur).squeeze(-1))
                seasonal_hat = torch.stack(outs, dim=1)
                trend_hat = self.trend_proj(trend)
                return seasonal_hat + trend_hat

        return _KoopaDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_samformer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
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
    restore_best: bool = True,
) -> np.ndarray:
    """
    Torch SAMformer-style linear-attention + adaptive mixing model (lite) direct forecast.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    d = int(d_model)
    heads = int(nhead)
    layers = int(num_layers)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    head_dim = d // heads
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

    def _build_model(_lag_count: int, h: int) -> Any:
        class _SAMBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(d)
                self.qkv = nn.Linear(d, 3 * d)
                self.out = nn.Linear(d, d)
                self.mix_gate = nn.Linear(d, d)
                self.norm2 = nn.LayerNorm(d)
                self.ffn = nn.Sequential(
                    nn.Linear(d, 2 * d),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(2 * d, d),
                )
                self.drop = nn.Dropout(p=drop)

            def _linear_attention(self, xb: Any) -> Any:
                qkv = self.qkv(xb)
                q, k, v = qkv.chunk(3, dim=-1)

                def _reshape(z: Any) -> Any:
                    return z.reshape(z.shape[0], z.shape[1], heads, head_dim).permute(0, 2, 1, 3)

                qh = F.elu(_reshape(q)) + 1.0
                kh = F.elu(_reshape(k)) + 1.0
                vh = _reshape(v)
                kv = torch.einsum("bhtd,bhte->bhde", kh, vh)
                numer = torch.einsum("bhtd,bhde->bhte", qh, kv)
                denom = torch.einsum("bhtd,bhd->bht", qh, kh.sum(dim=2)) + 1e-6
                out = numer / denom.unsqueeze(-1)
                out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], d)
                return self.out(out)

            def forward(self, xb: Any) -> Any:
                attn = self._linear_attention(self.norm1(xb))
                gate = torch.sigmoid(self.mix_gate(xb.mean(dim=1))).unsqueeze(1)
                xb = xb + self.drop(gate * attn)
                xb = xb + self.drop(self.ffn(self.norm2(xb)))
                return xb

        class _SAMformerDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                self.blocks = nn.ModuleList([_SAMBlock() for _ in range(layers)])
                self.head = nn.Linear(d, h)

            def forward(self, xb: Any) -> Any:
                z = self.in_proj(xb)
                for blk in self.blocks:
                    z = blk(z)
                return self.head(z[:, -1, :])

        return _SAMformerDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_retnet_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
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
    Torch RetNet-style retention network (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    d = int(d_model)
    heads = int(nhead)
    layers = int(num_layers)
    hidden = int(ffn_dim)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if hidden <= 0:
        raise ValueError(_FFN_DIM_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    head_dim = d // heads
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

    def _build_model(lag_count: int, h: int) -> Any:
        class _RetentionBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(d)
                self.q_proj = nn.Linear(d, d)
                self.k_proj = nn.Linear(d, d)
                self.v_proj = nn.Linear(d, d)
                self.out_proj = nn.Linear(d, d)
                self.decay_logits = nn.Parameter(
                    torch.linspace(-1.25, 1.25, steps=heads, dtype=torch.float32)
                )
                self.norm2 = nn.LayerNorm(d)
                self.ffn = nn.Sequential(
                    nn.Linear(d, hidden),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(hidden, d),
                )
                self.drop = nn.Dropout(p=drop)

            def _retention(self, xb: Any) -> Any:
                batch, seq_len, _dim = xb.shape

                def _reshape(z: Any) -> Any:
                    return z.reshape(batch, seq_len, heads, head_dim).permute(0, 2, 1, 3)

                q = F.elu(_reshape(self.q_proj(xb))) + 1.0
                k = F.elu(_reshape(self.k_proj(xb))) + 1.0
                v = _reshape(self.v_proj(xb))

                decay = torch.sigmoid(self.decay_logits).view(1, heads, 1, 1)
                state = torch.zeros(
                    (batch, heads, head_dim, head_dim), device=xb.device, dtype=xb.dtype
                )
                key_state = torch.zeros((batch, heads, head_dim), device=xb.device, dtype=xb.dtype)
                outs: list[Any] = []
                for idx in range(seq_len):
                    k_t = k[:, :, idx, :]
                    v_t = v[:, :, idx, :]
                    state = decay * state + torch.einsum("bhd,bhe->bhde", k_t, v_t)
                    key_state = decay.squeeze(-1) * key_state + k_t
                    numer = torch.einsum("bhd,bhde->bhe", q[:, :, idx, :], state)
                    denom = torch.einsum("bhd,bhd->bh", q[:, :, idx, :], key_state).unsqueeze(-1)
                    outs.append(numer / (denom + 1e-6))

                out = torch.stack(outs, dim=2)
                out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, d)
                return self.out_proj(out / math.sqrt(float(head_dim)))

            def forward(self, xb: Any) -> Any:
                xb = xb + self.drop(self._retention(self.norm1(xb)))
                xb = xb + self.drop(self.ffn(self.norm2(xb)))
                return xb

        class _RetNetDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, d)
                pos = _make_positional_encoding(lag_count, d)
                self.register_buffer(
                    "positional_encoding",
                    torch.tensor(pos, dtype=torch.float32).unsqueeze(0),
                )
                self.blocks = nn.ModuleList([_RetentionBlock() for _ in range(layers)])
                self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))

            def forward(self, xb: Any) -> Any:
                z = self.in_proj(xb) + self.positional_encoding[:, : xb.shape[1], :]
                for blk in self.blocks:
                    z = blk(z)
                return self.head(z[:, -1, :])

        return _RetNetDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=int(lags),
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


def torch_retnet_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
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
    Torch RetNet-style retention network trained for one-step prediction, forecasted recursively.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    d = int(d_model)
    heads = int(nhead)
    layers = int(num_layers)
    hidden = int(ffn_dim)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if hidden <= 0:
        raise ValueError(_FFN_DIM_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    head_dim = d // heads

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=1)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)
    Y_next = Y.reshape(Y.shape[0], 1)

    class _RetentionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.out_proj = nn.Linear(d, d)
            self.decay_logits = nn.Parameter(
                torch.linspace(-1.25, 1.25, steps=heads, dtype=torch.float32)
            )
            self.norm2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(hidden, d),
            )
            self.drop = nn.Dropout(p=drop)

        def _retention(self, xb: Any) -> Any:
            batch, seq_len, _dim = xb.shape

            def _reshape(z: Any) -> Any:
                return z.reshape(batch, seq_len, heads, head_dim).permute(0, 2, 1, 3)

            q = F.elu(_reshape(self.q_proj(xb))) + 1.0
            k = F.elu(_reshape(self.k_proj(xb))) + 1.0
            v = _reshape(self.v_proj(xb))

            decay = torch.sigmoid(self.decay_logits).view(1, heads, 1, 1)
            state = torch.zeros(
                (batch, heads, head_dim, head_dim), device=xb.device, dtype=xb.dtype
            )
            key_state = torch.zeros((batch, heads, head_dim), device=xb.device, dtype=xb.dtype)
            outs: list[Any] = []
            for idx in range(seq_len):
                k_t = k[:, :, idx, :]
                v_t = v[:, :, idx, :]
                state = decay * state + torch.einsum("bhd,bhe->bhde", k_t, v_t)
                key_state = decay.squeeze(-1) * key_state + k_t
                numer = torch.einsum("bhd,bhde->bhe", q[:, :, idx, :], state)
                denom = torch.einsum("bhd,bhd->bh", q[:, :, idx, :], key_state).unsqueeze(-1)
                outs.append(numer / (denom + 1e-6))

            out = torch.stack(outs, dim=2)
            out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, d)
            return self.out_proj(out / math.sqrt(float(head_dim)))

        def forward(self, xb: Any) -> Any:
            xb = xb + self.drop(self._retention(self.norm1(xb)))
            xb = xb + self.drop(self.ffn(self.norm2(xb)))
            return xb

    class _RetNetRecursive(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(1, d)
            pos = _make_positional_encoding(lag_count, d)
            self.register_buffer(
                "positional_encoding",
                torch.tensor(pos, dtype=torch.float32).unsqueeze(0),
            )
            self.blocks = nn.ModuleList([_RetentionBlock() for _ in range(layers)])
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        def forward(self, xb: Any) -> Any:
            z = self.in_proj(xb) + self.positional_encoding[:, : xb.shape[1], :]
            for blk in self.blocks:
                z = blk(z)
            return self.head(z[:, -1, :])

    model = _RetNetRecursive()
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
    model = _train_loop(model, X_seq, Y_next, cfg=cfg, device=str(device))

    hist = x_work[-lag_count:].astype(float, copy=True)
    preds: list[float] = []
    dev = torch.device(str(device))
    with torch.no_grad():
        for _ in range(h):
            feat = hist.reshape(1, lag_count, 1)
            feat_t = torch.tensor(feat, dtype=torch.float32, device=dev)
            mu = float(model(feat_t).detach().cpu().reshape(-1)[0])
            preds.append(mu)
            hist = np.concatenate([hist[1:], np.array([mu], dtype=float)], axis=0)

    yhat = np.asarray(preds, dtype=float)
    if bool(normalize):
        yhat = yhat * std + mean
    return yhat


def torch_timexer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    x_cols: Any = (),
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
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
    restore_best: bool = True,
    train_exog: Any | None = None,
    future_exog: Any | None = None,
) -> np.ndarray:
    """
    Torch TimeXer-style exogenous-aware transformer (lite) direct multi-horizon forecast.
    """
    torch = _require_torch()
    nn = torch.nn

    y = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    d = int(d_model)
    heads = int(nhead)
    layers = int(num_layers)
    drop = float(dropout)
    _x_cols = x_cols
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)
    if train_exog is None or future_exog is None:
        raise ValueError("torch-timexer-direct requires train_exog and future_exog")

    train_x_raw = _as_2d_float_array(train_exog, name="train_exog")
    future_x_raw = _as_2d_float_array(future_exog, name="future_exog")
    if int(train_x_raw.shape[0]) != int(y.size):
        raise ValueError("train_exog rows must match train length")
    if int(future_x_raw.shape[0]) != h:
        raise ValueError("future_exog rows must have horizon rows")

    y_work = y
    y_mean = 0.0
    y_std = 1.0
    if bool(normalize):
        y_work, y_mean, y_std = _normalize_series(y_work)
    train_x, future_x = _normalize_exog_pair(train_x_raw, future_x_raw)
    X, Y = _make_lagged_xy_exog_seq(y_work, train_x, lags=lag_count, horizon=h)

    x_dim = int(train_x.shape[1])
    past_dim = 1 + x_dim
    future_dim = x_dim

    class _CrossBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm_q = nn.LayerNorm(d)
            self.cross = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
            self.norm_ffn = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(2 * d, d),
            )
            self.drop = nn.Dropout(p=drop)

        def forward(self, q: Any, mem: Any) -> Any:
            attn, _ = self.cross(self.norm_q(q), mem, mem, need_weights=False)
            q = q + self.drop(attn)
            q = q + self.drop(self.ffn(self.norm_ffn(q)))
            return q

    class _TimeXerDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.past_proj = nn.Linear(past_dim, d)
            self.future_proj = nn.Linear(future_dim, d)
            self.past_pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
            self.future_pos = nn.Parameter(torch.zeros((1, h, d), dtype=torch.float32))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=2 * d,
                dropout=drop,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.blocks = nn.ModuleList([_CrossBlock() for _ in range(layers)])
            self.out = nn.Linear(d, 1)

        def forward(self, xb: Any) -> Any:
            past = xb[:, :lag_count, :]
            future_x = xb[:, lag_count:, 1:]
            mem = self.enc(self.past_proj(past) + self.past_pos)
            q = self.future_proj(future_x) + self.future_pos
            for blk in self.blocks:
                q = blk(q, mem)
            return self.out(q).squeeze(-1)

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
    model = _train_loop(_TimeXerDirect(), X, Y, cfg=cfg, device=str(device))

    future_tokens = np.concatenate(
        [np.zeros((h, 1), dtype=float), future_x.astype(float, copy=False)],
        axis=1,
    )
    feat = np.concatenate(
        [
            np.concatenate([y_work[-lag_count:].reshape(-1, 1), train_x[-lag_count:, :]], axis=1),
            future_tokens,
        ],
        axis=0,
    ).reshape(1, lag_count + h, 1 + x_dim)

    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * y_std + y_mean
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
    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    p = int(patch_len)
    s = int(stride)
    if p <= 0:
        raise ValueError(_PATCH_LEN_MIN_MSG)
    if s <= 0:
        raise ValueError(_STRIDE_MIN_MSG)
    if p > lag_count:
        raise ValueError("patch_len must be <= lags")

    n_patches = 1 + (lag_count - p) // s
    if n_patches <= 0:
        raise ValueError("Invalid patch configuration")

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    torch = _require_torch()
    nn = torch.nn

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


def torch_crossformer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    segment_len: int = 16,
    stride: int = 16,
    num_scales: int = 3,
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
    Crossformer-style (lite): multi-scale segmented tokens + Transformer encoder (direct multi-horizon).

    Notes:
      - This is a lightweight approximation inspired by Crossformer ideas (segmentation + cross-scale mixing).
      - For each scale i, we segment the lag window into length `segment_len * 2^i` tokens and concatenate
        all scale tokens into a single Transformer encoder sequence.
    """
    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    base_seg = int(segment_len)
    base_stride = int(stride)
    n_scales_req = int(num_scales)
    if base_seg <= 0:
        raise ValueError(_SEGMENT_LEN_MIN_MSG)
    if base_stride <= 0:
        raise ValueError(_STRIDE_MIN_MSG)
    if n_scales_req <= 0:
        raise ValueError("num_scales must be >= 1")
    if base_seg > lag_count:
        raise ValueError(_SEGMENT_LEN_MAX_LAGS_MSG)

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    torch = _require_torch()
    nn = torch.nn

    # Build scale configs (skip scales that don't fit).
    scales: list[tuple[int, int, int]] = []  # (seg_len, step, n_tokens)
    for i in range(int(n_scales_req)):
        seg_i = int(base_seg * (2**i))
        step_i = int(base_stride * (2**i))
        if seg_i > lag_count:
            break
        if step_i <= 0:
            continue
        n_tokens_i = 1 + (lag_count - seg_i) // step_i
        if n_tokens_i <= 0:
            continue
        scales.append((seg_i, step_i, int(n_tokens_i)))

    if not scales:
        raise ValueError("Invalid (segment_len, stride, num_scales) configuration for given lags")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _CrossFormerDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale_proj = nn.ModuleList([nn.Linear(int(seg), d) for seg, _step, _nt in scales])
            self.scale_pos = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros((1, int(nt), d), dtype=torch.float32))
                    for _seg, _step, nt in scales
                ]
            )
            self.scale_emb = nn.Embedding(int(len(scales)), d)
            self.drop = nn.Dropout(p=drop)

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
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            x0 = xb.squeeze(-1)  # (B, T)
            tokens: list[Any] = []
            for si, (seg_i, step_i, nt_i) in enumerate(scales):
                segs = x0.unfold(dimension=1, size=int(seg_i), step=int(step_i))  # (B, nt, seg_i)
                if segs.shape[1] != int(nt_i):
                    # Should not happen given precomputed nt_i, but guard against shape drift.
                    segs = segs[:, : int(nt_i), :]
                z = self.scale_proj[si](segs)
                z = z + self.scale_pos[si] + self.scale_emb.weight[int(si)].reshape(1, 1, d)
                tokens.append(self.drop(z))

            zcat = torch.cat(tokens, dim=1)
            zcat = self.enc(zcat)
            pooled = self.norm(zcat.mean(dim=1))
            return self.head(pooled)

    model = _CrossFormerDirect()
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


def torch_pyraformer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    segment_len: int = 16,
    stride: int = 16,
    num_levels: int = 3,
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
    Pyraformer-style (lite): pyramid pooling over segment tokens + Transformer encoder (direct multi-horizon).

    Notes:
      - This is a lightweight approximation inspired by Pyraformer ideas (hierarchical multi-resolution tokens).
      - Level 0 uses segmented tokens from the lag window; higher levels are built via pooling (factor 2).
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    seg = int(segment_len)
    step = int(stride)
    levels_req = int(num_levels)
    if seg <= 0:
        raise ValueError(_SEGMENT_LEN_MIN_MSG)
    if step <= 0:
        raise ValueError(_STRIDE_MIN_MSG)
    if levels_req <= 0:
        raise ValueError("num_levels must be >= 1")
    if seg > lag_count:
        raise ValueError(_SEGMENT_LEN_MAX_LAGS_MSG)

    n0 = 1 + (lag_count - seg) // step
    if n0 <= 0:
        raise ValueError("Invalid segment configuration: produces no tokens")

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    # Determine actual pyramid sizes.
    level_sizes: list[int] = [int(n0)]
    for _i in range(int(levels_req) - 1):
        nxt = int(level_sizes[-1] // 2)
        if nxt <= 0:
            break
        level_sizes.append(nxt)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _PyraFormerDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_proj = nn.Linear(int(seg), d)
            self.level_proj = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(p=drop))
                    for _ in level_sizes[1:]
                ]
            )
            self.pos = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros((1, int(sz), d), dtype=torch.float32))
                    for sz in level_sizes
                ]
            )
            self.level_emb = nn.Embedding(int(len(level_sizes)), d)
            self.drop = nn.Dropout(p=drop)

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
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            x0 = xb.squeeze(-1)  # (B, T)
            segs = x0.unfold(dimension=1, size=int(seg), step=int(step))  # (B, n0, seg)
            segs = segs[:, : int(level_sizes[0]), :]
            z0 = self.patch_proj(segs)
            z0 = z0 + self.pos[0] + self.level_emb.weight[0].reshape(1, 1, d)
            z0 = self.drop(z0)

            tokens: list[Any] = [z0]
            z_prev = z0
            for li in range(1, int(len(level_sizes))):
                n_prev = int(z_prev.shape[1])
                n_even = n_prev - (n_prev % 2)
                if n_even <= 0:
                    break
                z_pool = (
                    z_prev[:, :n_even, :].reshape(z_prev.shape[0], n_even // 2, 2, d).mean(dim=2)
                )
                z_pool = self.level_proj[int(li - 1)](z_pool)
                z_pool = z_pool[:, : int(level_sizes[li]), :]
                z_pool = z_pool + self.pos[li] + self.level_emb.weight[int(li)].reshape(1, 1, d)
                z_pool = self.drop(z_pool)
                tokens.append(z_pool)
                z_prev = z_pool

            zcat = torch.cat(tokens, dim=1)
            zcat = self.enc(zcat)
            pooled = self.norm(zcat.mean(dim=1))
            return self.head(pooled)

    model = _PyraFormerDirect()
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


def torch_perceiver_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    d_model: int = 64,
    latent_len: int = 32,
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
    Perceiver-style (lite): learnable latent array + cross-attention + latent Transformer (direct multi-horizon).

    This design can scale to long inputs by keeping the latent length fixed while attending to the input tokens.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)

    lat = int(latent_len)
    if lat <= 0:
        raise ValueError("latent_len must be >= 1")

    heads = int(nhead)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _PerceiverDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
            self.latents = nn.Parameter(torch.randn((1, lat, d), dtype=torch.float32) * 0.02)

            self.cross_norm_q = nn.LayerNorm(d)
            self.cross_norm_kv = nn.LayerNorm(d)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d,
                num_heads=heads,
                dropout=drop,
                batch_first=True,
            )
            self.drop = nn.Dropout(p=drop)

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
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            z = self.in_proj(xb) + self.pos  # (B,T,d)
            B = int(z.shape[0])
            lat_b = self.latents.expand(B, -1, -1)
            upd, _w = self.cross_attn(
                self.cross_norm_q(lat_b),
                self.cross_norm_kv(z),
                self.cross_norm_kv(z),
                need_weights=False,
            )
            lat_b = lat_b + self.drop(upd)
            lat_b = self.enc(lat_b)
            pooled = self.norm(lat_b.mean(dim=1))
            return self.head(pooled)

    model = _PerceiverDirect()
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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if int(d_model) <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_blocks) <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if int(token_mixing_hidden) <= 0:
        raise ValueError("token_mixing_hidden must be >= 1")
    if int(channel_mixing_hidden) <= 0:
        raise ValueError("channel_mixing_hidden must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if int(kernel_size) <= 0:
        raise ValueError(_KERNEL_SIZE_MIN_MSG)

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
        raise ValueError(_DROPOUT_RANGE_MSG)

    pool_s = str(pool).lower().strip()
    if pool_s not in {"last", "mean", "max"}:
        raise ValueError(_POOL_MODE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if c <= 0:
        raise ValueError(_CHANNELS_MIN_MSG)
    if int(num_blocks) <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if int(kernel_size) <= 0:
        raise ValueError(_KERNEL_SIZE_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    pool_s = str(pool).lower().strip()
    if pool_s not in {"last", "mean", "max"}:
        raise ValueError(_POOL_MODE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if c <= 0:
        raise ValueError(_CHANNELS_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(kernel_size) <= 0:
        raise ValueError(_KERNEL_SIZE_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if int(hidden_size) <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)

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
        raise ValueError(_DROPOUT_RANGE_MSG)

    class _BiLSTMDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rnn_drop = drop if int(num_layers) > 1 else 0.0
            self.lstm = _make_manual_lstm(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=float(rnn_drop),
                bidirectional=True,
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
        raise ValueError(_HORIZON_MIN_MSG)
    if int(hidden_size) <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)

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
        raise ValueError(_DROPOUT_RANGE_MSG)

    class _BiGRUDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rnn_drop = drop if int(num_layers) > 1 else 0.0
            self.gru = _make_manual_gru(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=float(rnn_drop),
                bidirectional=True,
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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if int(hidden_size) <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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
            rnn_drop = drop if int(num_layers) > 1 else 0.0
            self.gru = _make_manual_gru(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=float(rnn_drop),
                bidirectional=False,
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


def torch_segrnn_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    segment_len: int = 12,
    d_model: int = 64,
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
    restore_best: bool = True,
) -> np.ndarray:
    """
    Torch SegRNN-style segmented recurrent model (direct multi-horizon).

    The lag window is chunked into fixed-size segments, embedded segment-wise,
    encoded with a GRU, and decoded via horizon-query embeddings.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    seg_len = int(segment_len)
    d = int(d_model)
    hidden = int(hidden_size)
    layers = int(num_layers)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if seg_len <= 0:
        raise ValueError(_SEGMENT_LEN_MIN_MSG)
    if seg_len > lag_count:
        raise ValueError(_SEGMENT_LEN_MAX_LAGS_MSG)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if hidden <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    n_segments = int(math.ceil(lag_count / seg_len))
    total_len = int(n_segments * seg_len)
    pad_left = int(total_len - lag_count)

    def _segmentize_windows(windows: np.ndarray) -> np.ndarray:
        if pad_left > 0:
            windows = np.pad(windows, ((0, 0), (pad_left, 0)), mode="edge")
        return windows.reshape(int(windows.shape[0]), n_segments, seg_len)

    X_seg = _segmentize_windows(X)

    class _SegRNNDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seg_proj = nn.Linear(seg_len, d)
            self.seg_norm = nn.LayerNorm(d)
            self.pos = nn.Parameter(torch.zeros((1, n_segments, d), dtype=torch.float32))
            rnn_drop = drop if layers > 1 else 0.0
            self.gru = _make_manual_gru(
                input_size=d,
                hidden_size=hidden,
                num_layers=layers,
                dropout=float(rnn_drop),
                bidirectional=False,
            )
            self.horizon_emb = nn.Embedding(h, hidden)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(p=drop) if drop > 0.0 else nn.Identity(),
                nn.Linear(hidden, 1),
            )

        def forward(self, xb: Any) -> Any:  # xb: (B, S, segment_len)
            z = self.seg_norm(self.seg_proj(xb)) + self.pos
            out, _ = self.gru(z)
            ctx = out[:, -1, :]
            steps = self.horizon_emb(torch.arange(h, device=xb.device, dtype=torch.long))
            z_out = ctx.unsqueeze(1) + steps.unsqueeze(0)
            return self.head(z_out).squeeze(-1)

    model = _SegRNNDirect()
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
    model = _train_loop(model, X_seg, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count)
    feat_seg = _segmentize_windows(feat)
    with torch.no_grad():
        feat_t = torch.tensor(feat_seg, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_moderntcn_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    patch_len: int = 8,
    d_model: int = 64,
    num_blocks: int = 3,
    expansion_factor: float = 2.0,
    kernel_size: int = 9,
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
    Torch ModernTCN-style patchified convolutional mixer (direct multi-horizon).

    The lag window is grouped into fixed-size patches, embedded as tokens, mixed
    by large-kernel depthwise convolutions, and refined with channel MLP blocks.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    patch = int(patch_len)
    d = int(d_model)
    blocks = int(num_blocks)
    expand = float(expansion_factor)
    k = int(kernel_size)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if patch <= 0:
        raise ValueError(_PATCH_LEN_MIN_MSG)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if blocks <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if expand <= 0.0:
        raise ValueError("expansion_factor must be > 0")
    if k <= 0 or k % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    n_patches = int(math.ceil(lag_count / patch))
    total_len = int(n_patches * patch)
    pad_left = int(total_len - lag_count)

    def _patchify_windows(windows: np.ndarray) -> np.ndarray:
        if pad_left > 0:
            windows = np.pad(windows, ((0, 0), (pad_left, 0)), mode="edge")
        return windows.reshape(int(windows.shape[0]), n_patches, patch)

    X_patch = _patchify_windows(X)
    hidden_dim = max(1, int(round(d * expand)))

    class _ModernTCNBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm_conv = nn.LayerNorm(d)
            self.dwconv = nn.Conv1d(
                in_channels=d,
                out_channels=d,
                kernel_size=k,
                padding=k // 2,
                groups=d,
            )
            self.norm_ffn = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(hidden_dim, d),
            )
            self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

        def forward(self, xb: Any) -> Any:  # xb: (B, P, d)
            z = self.norm_conv(xb).transpose(1, 2)
            z = self.dwconv(z).transpose(1, 2)
            xb = xb + self.drop(z)
            z = self.ffn(self.norm_ffn(xb))
            return xb + self.drop(z)

    class _ModernTCNDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_proj = nn.Linear(patch, d)
            self.patch_norm = nn.LayerNorm(d)
            self.pos = nn.Parameter(torch.zeros((1, n_patches, d), dtype=torch.float32))
            self.blocks = nn.ModuleList([_ModernTCNBlock() for _ in range(blocks)])
            self.out_norm = nn.LayerNorm(2 * d)
            self.head = nn.Linear(2 * d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, P, patch_len)
            z = self.patch_norm(self.patch_proj(xb)) + self.pos
            for blk in self.blocks:
                z = blk(z)
            feat = torch.cat([z[:, -1, :], z.mean(dim=1)], dim=1)
            feat = self.out_norm(feat)
            return self.head(feat)

    model = _ModernTCNDirect()
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
    model = _train_loop(model, X_patch, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count)
    feat_patch = _patchify_windows(feat)
    with torch.no_grad():
        feat_t = torch.tensor(feat_patch, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_basisformer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    patch_len: int = 8,
    d_model: int = 64,
    num_bases: int = 16,
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
    Torch Basisformer-style learned basis routing model (direct multi-horizon).

    The lag window is patchified into tokens, projected onto a learned basis bank,
    encoded with a compact Transformer, and decoded through horizon queries.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    patch = int(patch_len)
    d = int(d_model)
    bases = int(num_bases)
    heads = int(nhead)
    layers = int(num_layers)
    ff = int(dim_feedforward)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if patch <= 0:
        raise ValueError(_PATCH_LEN_MIN_MSG)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if bases <= 0:
        raise ValueError("num_bases must be >= 1")
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if ff <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    n_patches = int(math.ceil(lag_count / patch))
    total_len = int(n_patches * patch)
    pad_left = int(total_len - lag_count)

    def _patchify_windows(windows: np.ndarray) -> np.ndarray:
        if pad_left > 0:
            windows = np.pad(windows, ((0, 0), (pad_left, 0)), mode="edge")
        return windows.reshape(int(windows.shape[0]), n_patches, patch)

    X_patch = _patchify_windows(X)

    class _BasisformerDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_proj = nn.Linear(patch, d)
            self.patch_norm = nn.LayerNorm(d)
            self.pos = nn.Parameter(torch.zeros((1, n_patches, d), dtype=torch.float32))
            self.coeff_norm = nn.LayerNorm(d)
            self.coeff_proj = nn.Linear(d, bases)
            self.basis_bank = nn.Parameter(torch.randn((bases, d), dtype=torch.float32) * 0.02)
            self.basis_gate = nn.Linear(d, d)
            self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=ff,
                dropout=drop,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.horizon_queries = nn.Embedding(h, d)
            self.decoder_attn = nn.MultiheadAttention(
                embed_dim=d,
                num_heads=heads,
                dropout=drop,
                batch_first=True,
            )
            self.out_norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)

        def forward(self, xb: Any) -> Any:  # xb: (B, P, patch_len)
            z = self.patch_norm(self.patch_proj(xb)) + self.pos
            coeff = torch.softmax(self.coeff_proj(self.coeff_norm(z)), dim=-1)
            basis_ctx = torch.matmul(coeff, self.basis_bank)
            gate = torch.sigmoid(self.basis_gate(z))
            z = z + self.drop(gate * basis_ctx)
            z = self.encoder(z)

            B = int(z.shape[0])
            q = self.horizon_queries.weight.unsqueeze(0).expand(B, -1, -1)
            q, _ = self.decoder_attn(q, z, z, need_weights=False)
            q = self.out_norm(q)
            return self.head(q).squeeze(-1)

    model = _BasisformerDirect()
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
    model = _train_loop(model, X_patch, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count)
    feat_patch = _patchify_windows(feat)
    with torch.no_grad():
        feat_t = torch.tensor(feat_patch, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_witran_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    grid_cols: int = 12,
    d_model: int = 64,
    hidden_size: int = 64,
    nhead: int = 4,
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
    restore_best: bool = True,
) -> np.ndarray:
    """
    Torch WITRAN-style 2D grid recurrent mixer (direct multi-horizon).

    The lag window is reshaped into a row-column grid, updated by coupled row/column
    recurrent states, and decoded with compact horizon-query attention.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    cols = int(grid_cols)
    d = int(d_model)
    hidden = int(hidden_size)
    heads = int(nhead)
    layers = int(num_layers)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if cols <= 0:
        raise ValueError("grid_cols must be >= 1")
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if hidden <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    rows = int(math.ceil(lag_count / cols))
    total_len = int(rows * cols)
    pad_left = int(total_len - lag_count)

    def _gridify_windows(windows: np.ndarray) -> np.ndarray:
        if pad_left > 0:
            windows = np.pad(windows, ((0, 0), (pad_left, 0)), mode="edge")
        return windows.reshape(int(windows.shape[0]), rows, cols)

    X_grid = _gridify_windows(X)

    class _GridBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            in_dim = d + 2 * hidden
            self.norm = nn.LayerNorm(d)
            self.row_gate = nn.Linear(in_dim, hidden)
            self.row_cand = nn.Linear(in_dim, hidden)
            self.col_gate = nn.Linear(in_dim, hidden)
            self.col_cand = nn.Linear(in_dim, hidden)
            self.out_proj = nn.Linear(2 * hidden, d)
            self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

        def forward(self, xg: Any) -> Any:  # xg: (B, R, C, d)
            z = self.norm(xg)
            B = int(z.shape[0])
            row_states = [z.new_zeros((B, hidden)) for _ in range(rows)]
            col_states = [z.new_zeros((B, hidden)) for _ in range(cols)]
            row_tokens: list[Any] = []
            for i in range(rows):
                col_tokens: list[Any] = []
                for j in range(cols):
                    xij = z[:, i, j, :]
                    joint = torch.cat([xij, row_states[i], col_states[j]], dim=-1)
                    row_gate = torch.sigmoid(self.row_gate(joint))
                    row_cand = torch.tanh(self.row_cand(joint))
                    col_gate = torch.sigmoid(self.col_gate(joint))
                    col_cand = torch.tanh(self.col_cand(joint))
                    row_states[i] = row_gate * row_states[i] + (1.0 - row_gate) * row_cand
                    col_states[j] = col_gate * col_states[j] + (1.0 - col_gate) * col_cand
                    cell = self.out_proj(torch.cat([row_states[i], col_states[j]], dim=-1))
                    col_tokens.append(cell)
                row_tokens.append(torch.stack(col_tokens, dim=1))
            upd = torch.stack(row_tokens, dim=1)
            return xg + self.drop(upd)

    class _WITRANDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cell_proj = nn.Linear(1, d)
            self.token_norm = nn.LayerNorm(d)
            self.row_pos = nn.Parameter(torch.zeros((1, rows, 1, d), dtype=torch.float32))
            self.col_pos = nn.Parameter(torch.zeros((1, 1, cols, d), dtype=torch.float32))
            self.blocks = nn.ModuleList([_GridBlock() for _ in range(layers)])
            self.horizon_queries = nn.Embedding(h, d)
            self.decoder_attn = nn.MultiheadAttention(
                embed_dim=d,
                num_heads=heads,
                dropout=drop,
                batch_first=True,
            )
            self.out_norm = nn.LayerNorm(d)
            self.head = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Dropout(p=drop) if drop > 0.0 else nn.Identity(),
                nn.Linear(d, 1),
            )

        def forward(self, xb: Any) -> Any:  # xb: (B, R, C)
            z = self.token_norm(self.cell_proj(xb.unsqueeze(-1))) + self.row_pos + self.col_pos
            for blk in self.blocks:
                z = blk(z)
            B = int(z.shape[0])
            tokens = z.reshape(B, rows * cols, d)
            q = self.horizon_queries.weight.unsqueeze(0).expand(B, -1, -1)
            q, _ = self.decoder_attn(q, tokens, tokens, need_weights=False)
            q = self.out_norm(q)
            return self.head(q).squeeze(-1)

    model = _WITRANDirect()
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
    model = _train_loop(model, X_grid, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count)
    feat_grid = _gridify_windows(feat)
    with torch.no_grad():
        feat_t = torch.tensor(feat_grid, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_crossgnn_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    d_model: int = 64,
    num_blocks: int = 3,
    top_k: int = 8,
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
    Torch CrossGNN-style lag-graph mixer (direct multi-horizon).

    Lag positions are treated as graph nodes with a learned sparse adjacency, then
    mixed through graph propagation and channel MLP updates across multiple blocks.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    d = int(d_model)
    blocks = int(num_blocks)
    k = int(top_k)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if blocks <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if k <= 0:
        raise ValueError(_TOP_K_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)
    topk = min(k, lag_count)

    class _GraphBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm_g = nn.LayerNorm(d)
            self.msg_proj = nn.Linear(d, d)
            self.gate = nn.Linear(2 * d, d)
            self.update = nn.Linear(2 * d, d)
            self.norm_ffn = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(2 * d, d),
            )
            self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

        def forward(self, z: Any, adj: Any) -> Any:  # z: (B, T, d), adj: (T, T)
            zn = self.norm_g(z)
            msg = torch.einsum("ij,bjd->bid", adj, self.msg_proj(zn))
            joint = torch.cat([zn, msg], dim=-1)
            gated = torch.sigmoid(self.gate(joint)) * torch.tanh(self.update(joint))
            z = z + self.drop(gated)
            z = z + self.drop(self.ffn(self.norm_ffn(z)))
            return z

    class _CrossGNNDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
            self.node_emb = nn.Parameter(torch.randn((lag_count, d), dtype=torch.float32) * 0.02)
            self.blocks = nn.ModuleList([_GraphBlock() for _ in range(blocks)])
            self.out_norm = nn.LayerNorm(2 * d)
            self.head = nn.Linear(2 * d, h)

        def _make_adj(self) -> Any:
            scores = torch.matmul(self.node_emb, self.node_emb.transpose(0, 1))
            if topk < lag_count:
                vals, idx = torch.topk(scores, k=topk, dim=-1)
                mask = torch.full_like(scores, float("-inf"))
                mask.scatter_(1, idx, vals)
                scores = mask
            adj = torch.softmax(scores / math.sqrt(float(d)), dim=-1)
            eye = torch.eye(lag_count, device=adj.device, dtype=adj.dtype)
            adj = 0.9 * adj + 0.1 * eye
            adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return adj

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            z = self.in_proj(xb) + self.pos
            adj = self._make_adj()
            for blk in self.blocks:
                z = blk(z, adj)
            feat = torch.cat([z[:, -1, :], z.mean(dim=1)], dim=-1)
            feat = self.out_norm(feat)
            return self.head(feat)

    model = _CrossGNNDirect()
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


def torch_pathformer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    d_model: int = 64,
    expert_patch_lens: Any = (4, 8, 16),
    num_blocks: int = 3,
    top_k: int = 2,
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
    Torch Pathformer-style multi-scale expert routing model (direct multi-horizon).

    Multiple patch-size experts summarize the lag window at different scales, and a
    lightweight router selects the most relevant experts block-by-block.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    d = int(d_model)
    blocks = int(num_blocks)
    k = int(top_k)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if blocks <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if k <= 0:
        raise ValueError(_TOP_K_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    if isinstance(expert_patch_lens, int):
        patch_lens = (int(expert_patch_lens),)
    elif isinstance(expert_patch_lens, str):
        parts = [p.strip() for p in expert_patch_lens.split(",") if p.strip()]
        patch_lens = tuple(int(p) for p in parts)
    elif isinstance(expert_patch_lens, list | tuple):
        patch_lens = tuple(int(p) for p in expert_patch_lens)
    else:
        patch_lens = (int(expert_patch_lens),)
    if not patch_lens or any(p <= 0 for p in patch_lens):
        raise ValueError("expert_patch_lens must be a non-empty sequence of positive ints")
    topk = min(k, len(patch_lens))

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _ScaleExpert(nn.Module):
        def __init__(self, patch_len: int) -> None:
            super().__init__()
            self.patch_len = int(patch_len)
            self.proj = nn.Linear(int(patch_len), d)
            self.norm = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(2 * d, d),
            )
            self.out_norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

        def forward(self, xflat: Any) -> Any:  # xflat: (B, T)
            B = int(xflat.shape[0])
            pad_left = (-lag_count) % self.patch_len
            if pad_left > 0:
                left = xflat[:, :1].expand(-1, pad_left)
                xflat = torch.cat([left, xflat], dim=1)
            num_tokens = int(xflat.shape[1] // self.patch_len)
            patches = xflat.reshape(B, num_tokens, self.patch_len)
            z = self.proj(patches)
            z = z + self.drop(self.ffn(self.norm(z)))
            return self.out_norm(z.mean(dim=1))

    class _RoutingBlock(nn.Module):
        def __init__(self, num_experts: int) -> None:
            super().__init__()
            self.router = nn.Linear(d, num_experts)
            self.mix = nn.Sequential(
                nn.LayerNorm(2 * d),
                nn.Linear(2 * d, 2 * d),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(2 * d, d),
            )
            self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

        def forward(self, ctx: Any, expert_stack: Any) -> Any:  # ctx: (B,d), expert_stack: (B,E,d)
            logits = self.router(ctx)
            if topk < int(expert_stack.shape[1]):
                vals, idx = torch.topk(logits, k=topk, dim=-1)
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(1, idx, vals)
                logits = masked
            weights = torch.softmax(logits, dim=-1)
            routed = torch.einsum("be,bed->bd", weights, expert_stack)
            upd = self.mix(torch.cat([ctx, routed], dim=-1))
            return ctx + self.drop(upd)

    class _PathformerDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_proj = nn.Linear(lag_count, d)
            self.experts = nn.ModuleList([_ScaleExpert(p) for p in patch_lens])
            self.blocks = nn.ModuleList([_RoutingBlock(len(patch_lens)) for _ in range(blocks)])
            self.out_norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            xflat = xb.squeeze(-1)
            ctx = self.context_proj(xflat)
            expert_feats = [expert(xflat) for expert in self.experts]
            expert_stack = torch.stack(expert_feats, dim=1)
            for blk in self.blocks:
                ctx = blk(ctx, expert_stack)
            ctx = self.out_norm(ctx)
            return self.head(ctx)

    model = _PathformerDirect()
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


def torch_timesmamba_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    patch_len: int = 8,
    d_model: int = 64,
    state_size: int = 64,
    num_blocks: int = 3,
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
    Torch TimesMamba-style patch state-space mixer (direct multi-horizon).

    The lag window is patchified into tokens, mixed with lightweight state-space
    recurrences over patch tokens, and decoded from the final contextual summary.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    patch = int(patch_len)
    d = int(d_model)
    state = int(state_size)
    blocks = int(num_blocks)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if patch <= 0:
        raise ValueError(_PATCH_LEN_MIN_MSG)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if state <= 0:
        raise ValueError("state_size must be >= 1")
    if blocks <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    n_patches = int(math.ceil(lag_count / patch))
    total_len = int(n_patches * patch)
    pad_left = int(total_len - lag_count)

    def _patchify_windows(windows: np.ndarray) -> np.ndarray:
        if pad_left > 0:
            windows = np.pad(windows, ((0, 0), (pad_left, 0)), mode="edge")
        return windows.reshape(int(windows.shape[0]), n_patches, patch)

    X_patch = _patchify_windows(X)

    class _SSMBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(d, state)
            self.state_proj = nn.Linear(state, state)
            self.out_proj = nn.Linear(state, d)
            self.gate = nn.Linear(d, state)
            self.norm = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(2 * d, d),
            )
            self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

        def forward(self, z: Any) -> Any:  # z: (B, P, d)
            B, P, _ = z.shape
            zn = self.norm(z)
            s = z.new_zeros((B, state))
            outs: list[Any] = []
            for t in range(int(P)):
                x_t = zn[:, t, :]
                gate = torch.sigmoid(self.gate(x_t))
                cand = torch.tanh(self.in_proj(x_t) + self.state_proj(s))
                s = gate * cand + (1.0 - gate) * s
                outs.append(self.out_proj(s))
            upd = torch.stack(outs, dim=1)
            z = z + self.drop(upd)
            z = z + self.drop(self.ffn(self.norm(z)))
            return z

    class _TimesMambaDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_proj = nn.Linear(patch, d)
            self.patch_norm = nn.LayerNorm(d)
            self.pos = nn.Parameter(torch.zeros((1, n_patches, d), dtype=torch.float32))
            self.blocks = nn.ModuleList([_SSMBlock() for _ in range(blocks)])
            self.out_norm = nn.LayerNorm(2 * d)
            self.head = nn.Linear(2 * d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, P, patch_len)
            z = self.patch_norm(self.patch_proj(xb)) + self.pos
            for blk in self.blocks:
                z = blk(z)
            feat = torch.cat([z[:, -1, :], z.mean(dim=1)], dim=-1)
            feat = self.out_norm(feat)
            return self.head(feat)

    model = _TimesMambaDirect()
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
    model = _train_loop(model, X_patch, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count)
    feat_patch = _patchify_windows(feat)
    with torch.no_grad():
        feat_t = torch.tensor(feat_patch, dtype=torch.float32, device=torch.device(str(device)))
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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if int(d_model) <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(ffn_dim) <= 0:
        raise ValueError(_FFN_DIM_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if int(num_blocks) <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
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
        raise ValueError(_DROPOUT_RANGE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(hidden_size) <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if int(hidden_size) <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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
            rnn_drop = drop if int(num_layers) > 1 else 0.0
            self.rnn = _make_manual_gru(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=float(rnn_drop),
                bidirectional=False,
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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if int(hidden_size) <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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
            rnn_drop = drop if int(num_layers) > 1 else 0.0
            self.rnn = _make_manual_gru(
                input_size=1,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=float(rnn_drop),
                bidirectional=False,
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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if c <= 0:
        raise ValueError(_CHANNELS_MIN_MSG)
    if int(num_blocks) <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if int(bottleneck_channels) <= 0:
        raise ValueError("bottleneck_channels must be at least 1")

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
        raise ValueError(_DROPOUT_RANGE_MSG)

    pool_s = str(pool).lower().strip()
    if pool_s not in {"last", "mean", "max"}:
        raise ValueError(_POOL_MODE_MSG)

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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)
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
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    ffn = int(ffn_dim)
    if ffn <= 0:
        raise ValueError(_FFN_DIM_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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


def torch_hyena_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    ffn_dim: int = 128,
    kernel_size: int = 64,
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
    Hyena-style long convolution sequence model (lite) for direct multi-horizon forecasting.

    The core mixing operator is a depthwise causal Conv1D (per-channel long kernel) with gating,
    followed by a channel-mixing FFN.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    hidden = int(ffn_dim)
    if hidden <= 0:
        raise ValueError(_FFN_DIM_MIN_MSG)
    k = int(kernel_size)
    if k <= 0:
        raise ValueError(_KERNEL_SIZE_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _HyenaBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.in_proj = nn.Linear(d, 2 * d)
            self.dwconv = nn.Conv1d(d, d, kernel_size=int(k), groups=d)
            self.out_proj = nn.Linear(d, d)
            self.drop1 = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

            self.norm2 = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, hidden)
            self.fc2 = nn.Linear(hidden, d)
            self.drop2 = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

        def forward(self, xb: Any) -> Any:  # (B, T, d)
            z = self.norm1(xb)
            g, v = self.in_proj(z).chunk(2, dim=-1)
            g = torch.sigmoid(g)

            v_ch = v.transpose(1, 2)  # (B, d, T)
            v_pad = F.pad(v_ch, (int(k) - 1, 0))
            y = self.dwconv(v_pad).transpose(1, 2)  # (B, T, d)
            y = self.out_proj(g * y)
            xb = xb + self.drop1(y)

            z2 = self.norm2(xb)
            y2 = F.gelu(self.fc1(z2))
            y2 = self.fc2(self.drop2(y2))
            xb = xb + self.drop2(y2)
            return xb

    class _HyenaDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
            self.blocks = nn.ModuleList([_HyenaBlock() for _ in range(int(num_layers))])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)
            z = self.embed(xb) + self.pos
            for blk in self.blocks:
                z = blk(z)
            z = self.norm(z)
            return self.head(z[:, -1, :])

    model = _HyenaDirect()
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


def torch_dilated_rnn_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    cell: str = "gru",
    hidden_size: int = 64,
    num_layers: int = 3,
    dilation_base: int = 2,
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
    Dilated RNN (lite) for direct multi-horizon forecasting on lag windows.

    Each recurrent layer uses a fixed dilation (1, dilation_base, dilation_base^2, ...),
    which increases the receptive field without increasing sequence length.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    cell_s = str(cell).lower().strip()
    if cell_s not in {"gru", "lstm"}:
        raise ValueError("cell must be one of: gru, lstm")

    d = int(hidden_size)
    if d <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    L = int(num_layers)
    if L <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    base = int(dilation_base)
    if base <= 1:
        raise ValueError("dilation_base must be >= 2")
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _DilatedRNNDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.drop_in = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

            dilations = [int(base**i) for i in range(L)]
            self.dilations = dilations
            if cell_s == "gru":
                self.cells = nn.ModuleList(
                    [_make_manual_gru_cell(input_size=d, hidden_size=d) for _ in dilations]
                )
            else:
                self.cells = nn.ModuleList(
                    [_make_manual_lstm_cell(input_size=d, hidden_size=d) for _ in dilations]
                )

            self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in dilations])
            self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)
            z = self.drop_in(self.embed(xb))  # (B, T, d)
            B = int(z.shape[0])
            T = int(z.shape[1])

            for layer, dil in enumerate(self.dilations):
                cell_mod = self.cells[layer]
                norm = self.norms[layer]

                if cell_s == "gru":
                    h_states: list[Any] = []
                    zeros = torch.zeros((B, d), device=z.device, dtype=z.dtype)
                    for t in range(T):
                        h_prev = h_states[t - dil] if t >= dil else zeros
                        ht = cell_mod(z[:, t, :], h_prev)
                        h_states.append(ht)
                    z = torch.stack(h_states, dim=1)
                else:
                    h_states = []
                    c_states: list[Any] = []
                    zeros_h = torch.zeros((B, d), device=z.device, dtype=z.dtype)
                    zeros_c = torch.zeros((B, d), device=z.device, dtype=z.dtype)
                    for t in range(T):
                        if t >= dil:
                            h_prev = h_states[t - dil]
                            c_prev = c_states[t - dil]
                        else:
                            h_prev = zeros_h
                            c_prev = zeros_c
                        ht, ct = cell_mod(z[:, t, :], (h_prev, c_prev))
                        h_states.append(ht)
                        c_states.append(ct)
                    z = torch.stack(h_states, dim=1)

                z = norm(z)
                z = self.drop(F.gelu(z))

            last = z[:, -1, :]
            return self.head(last)

    model = _DilatedRNNDirect()
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


def torch_kan_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    grid_size: int = 16,
    grid_range: float = 2.0,
    dropout: float = 0.1,
    linear_skip: bool = True,
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
    KAN (Kolmogorov-Arnold Network) style spline MLP (lite) for direct forecasting.

    Implements per-edge univariate functions via a shared triangular spline basis over a
    fixed grid. This is a lightweight, self-contained approximation that is still fully
    implemented (no external KAN packages).
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    L = int(num_layers)
    if L <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    K = int(grid_size)
    if K < 4:
        raise ValueError("grid_size must be >= 4")
    grid_r = float(grid_range)
    if grid_r <= 0.0:
        raise ValueError("grid_range must be > 0")
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)

    class _KANSplineLayer(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            knots = torch.linspace(-grid_r, grid_r, int(K), dtype=torch.float32)
            self.register_buffer("knots", knots, persistent=False)
            if int(K) < 2:
                raise ValueError("grid_size must be >= 2")
            delta = float((2.0 * grid_r) / float(int(K) - 1))
            self.register_buffer(
                "inv_delta",
                torch.tensor(1.0 / max(delta, 1e-6), dtype=torch.float32),
                persistent=False,
            )
            self.coeff = nn.Parameter(torch.empty((out_features, in_features, int(K))))
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=torch.float32))
            self.linear = nn.Linear(in_features, out_features) if bool(linear_skip) else None
            nn.init.normal_(self.coeff, mean=0.0, std=0.02)

        def forward(self, xb: Any) -> Any:  # (B, in_features)
            xk = xb.unsqueeze(-1)  # (B, in, 1)
            basis = torch.relu(1.0 - torch.abs(xk - self.knots.reshape(1, 1, -1)) * self.inv_delta)
            y = torch.einsum("bik,oik->bo", basis, self.coeff)
            if self.linear is not None:
                y = y + self.linear(xb)
            return y + self.bias

    class _KANDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            in_dim = int(lag_count)
            for _i in range(int(L)):
                layers.append(_KANSplineLayer(in_dim, d))
                layers.append(nn.LayerNorm(d))
                layers.append(nn.Dropout(p=drop) if drop > 0.0 else nn.Identity())
                in_dim = d
            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:
            z = self.net(xb)
            z = F.gelu(z)
            return self.head(z)

    model = _KANDirect()
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


def torch_scinet_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    num_stages: int = 3,
    conv_kernel: int = 5,
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
    SCINet-style sample-convolution interaction network (lite) for direct forecasting.

    This is a lightweight variant that repeatedly splits the sequence into even/odd
    subsequences, applies convolutional mixing, and merges back (with residual FFN).
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    stages = int(num_stages)
    if stages <= 0:
        raise ValueError("num_stages must be >= 1")
    k = int(conv_kernel)
    if k <= 0:
        raise ValueError("conv_kernel must be >= 1")
    hidden = int(ffn_dim)
    if hidden <= 0:
        raise ValueError(_FFN_DIM_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    pad_left = (k - 1) // 2
    pad_right = (k - 1) - pad_left

    class _SCIBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.even_conv = nn.Conv1d(d, d, kernel_size=int(k), padding=0)
            self.odd_conv = nn.Conv1d(d, d, kernel_size=int(k), padding=0)
            self.cross_even = nn.Linear(d, d)
            self.cross_odd = nn.Linear(d, d)
            self.out_proj = nn.Linear(d, d)
            self.drop1 = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

            self.norm2 = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, hidden)
            self.fc2 = nn.Linear(hidden, d)
            self.drop2 = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

        def _same_conv(self, conv: Any, xbt: Any) -> Any:
            # xbt: (B, T, d)
            x_ch = xbt.transpose(1, 2)  # (B, d, T)
            x_pad = F.pad(x_ch, (int(pad_left), int(pad_right)))
            y = conv(x_pad).transpose(1, 2)
            return y

        def _pad_to(self, xbt: Any, target_len: int) -> Any:
            if int(xbt.shape[1]) == int(target_len):
                return xbt
            if int(xbt.shape[1]) > int(target_len):
                return xbt[:, : int(target_len), :]
            pad_n = int(target_len) - int(xbt.shape[1])
            return F.pad(xbt, (0, 0, 0, pad_n))

        def forward(self, xb: Any) -> Any:  # (B, T, d)
            z = self.norm1(xb)
            even = z[:, ::2, :]
            odd = z[:, 1::2, :]

            even_h = F.gelu(self._same_conv(self.even_conv, even))
            odd_h = F.gelu(self._same_conv(self.odd_conv, odd))

            even_len = int(even.shape[1])
            odd_len = int(odd.shape[1])
            odd_to_even = self._pad_to(odd_h, even_len)
            even_to_odd = self._pad_to(even_h, odd_len)

            even2 = even + self.cross_even(odd_to_even)
            odd2 = odd + self.cross_odd(even_to_odd)

            # Merge (interleave) back.
            out = torch.zeros_like(z)
            out[:, ::2, :] = even2
            out[:, 1::2, :] = odd2

            xb = xb + self.drop1(self.out_proj(out))

            z2 = self.norm2(xb)
            y2 = F.gelu(self.fc1(z2))
            y2 = self.fc2(self.drop2(y2))
            xb = xb + self.drop2(y2)
            return xb

    class _SCINetDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.zeros((1, lag_count, d), dtype=torch.float32))
            self.blocks = nn.ModuleList([_SCIBlock() for _ in range(int(stages))])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)
            z = self.embed(xb) + self.pos
            for blk in self.blocks:
                z = blk(z)
            z = self.norm(z)
            return self.head(z[:, -1, :])

    model = _SCINetDirect()
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


def torch_etsformer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    alpha_init: float = 0.3,
    beta_init: float = 0.1,
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
    ETSformer-style exponential smoothing + Transformer residual model (lite).

    - Compute a Holt-style baseline forecast from the context window (learned alpha/beta).
    - Feed residual tokens into a Transformer encoder to predict horizon residual adjustments.
    - Output = baseline + residual_adjustment.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_HEAD_DIVISIBILITY_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    alpha0 = float(alpha_init)
    beta0 = float(beta_init)
    if not (0.0 < alpha0 < 1.0):
        raise ValueError("alpha_init must be in (0,1)")
    if not (0.0 < beta0 < 1.0):
        raise ValueError("beta_init must be in (0,1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    def _logit(p: float) -> float:
        p2 = min(max(float(p), 1e-4), 1.0 - 1e-4)
        return math.log(p2 / (1.0 - p2))

    alpha_logit_init = _logit(alpha0)
    beta_logit_init = _logit(beta0)

    class _ETSformerDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init, dtype=torch.float32))
            self.beta_logit = nn.Parameter(torch.tensor(beta_logit_init, dtype=torch.float32))
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
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)
            y = xb.squeeze(-1)  # (B, T)
            B = int(y.shape[0])
            T = int(y.shape[1])

            alpha = torch.sigmoid(self.alpha_logit)
            beta = torch.sigmoid(self.beta_logit)

            level = y[:, 0]
            if T >= 2:
                b = y[:, 1] - y[:, 0]
            else:
                b = torch.zeros((B,), device=y.device, dtype=y.dtype)

            levels: list[Any] = [level]
            trends: list[Any] = [b]
            for t in range(1, T):
                yt = y[:, t]
                level_new = alpha * yt + (1.0 - alpha) * (level + b)
                b_new = beta * (level_new - level) + (1.0 - beta) * b
                level, b = level_new, b_new
                levels.append(level)
                trends.append(b)

            level_seq = torch.stack(levels, dim=1)  # (B, T)
            trend_seq = torch.stack(trends, dim=1)  # (B, T)

            fitted = torch.empty_like(y)
            fitted[:, 0] = level_seq[:, 0]
            if T > 1:
                fitted[:, 1:] = level_seq[:, :-1] + trend_seq[:, :-1]

            resid = (y - fitted).unsqueeze(-1)  # (B, T, 1)

            steps = torch.arange(1, int(h) + 1, device=y.device, dtype=y.dtype).reshape(1, -1)
            baseline = level_seq[:, -1].unsqueeze(1) + steps * trend_seq[:, -1].unsqueeze(1)

            z = self.embed(resid) + self.pos
            z = self.enc(z)
            delta = self.head(z[:, -1, :])
            return baseline + delta

    model = _ETSformerDirect()
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


def torch_esrnn_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    cell: str = "gru",
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    alpha_init: float = 0.3,
    beta_init: float = 0.1,
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
    ESRNN-style hybrid (lite): exponential smoothing baseline + RNN residual model.

    - Holt-style baseline forecast from the context window (learned alpha/beta).
    - RNN (GRU/LSTM) reads residual tokens to predict horizon residual adjustments.
    - Output = baseline + residual_adjustment.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)

    cell_s = str(cell).lower().strip()
    if cell_s not in {"gru", "lstm"}:
        raise ValueError("cell must be one of: gru, lstm")

    d = int(hidden_size)
    if d <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    L = int(num_layers)
    if L <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    alpha0 = float(alpha_init)
    beta0 = float(beta_init)
    if not (0.0 < alpha0 < 1.0):
        raise ValueError("alpha_init must be in (0,1)")
    if not (0.0 < beta0 < 1.0):
        raise ValueError("beta_init must be in (0,1)")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    def _logit(p: float) -> float:
        p2 = min(max(float(p), 1e-4), 1.0 - 1e-4)
        return math.log(p2 / (1.0 - p2))

    alpha_logit_init = _logit(alpha0)
    beta_logit_init = _logit(beta0)

    class _ESRNNDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init, dtype=torch.float32))
            self.beta_logit = nn.Parameter(torch.tensor(beta_logit_init, dtype=torch.float32))

            if cell_s == "gru":
                rnn_drop = float(drop) if int(L) > 1 else 0.0
                self.rnn = _make_manual_gru(
                    input_size=1,
                    hidden_size=d,
                    num_layers=int(L),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
            else:
                rnn_drop = float(drop) if int(L) > 1 else 0.0
                self.rnn = _make_manual_lstm(
                    input_size=1,
                    hidden_size=d,
                    num_layers=int(L),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
            if xb.ndim == 2:
                xb = xb.unsqueeze(-1)

            y = xb.squeeze(-1)  # (B, T)
            B = int(y.shape[0])
            T = int(y.shape[1])

            alpha = torch.sigmoid(self.alpha_logit)
            beta = torch.sigmoid(self.beta_logit)

            level = y[:, 0]
            if T >= 2:
                b = y[:, 1] - y[:, 0]
            else:
                b = torch.zeros((B,), device=y.device, dtype=y.dtype)

            levels: list[Any] = [level]
            trends: list[Any] = [b]
            for t in range(1, T):
                yt = y[:, t]
                level_new = alpha * yt + (1.0 - alpha) * (level + b)
                b_new = beta * (level_new - level) + (1.0 - beta) * b
                level, b = level_new, b_new
                levels.append(level)
                trends.append(b)

            level_seq = torch.stack(levels, dim=1)  # (B, T)
            trend_seq = torch.stack(trends, dim=1)  # (B, T)

            fitted = torch.empty_like(y)
            fitted[:, 0] = level_seq[:, 0]
            if T > 1:
                fitted[:, 1:] = level_seq[:, :-1] + trend_seq[:, :-1]

            resid = (y - fitted).unsqueeze(-1)  # (B, T, 1)

            steps = torch.arange(1, int(h) + 1, device=y.device, dtype=y.dtype).reshape(1, -1)
            baseline = level_seq[:, -1].unsqueeze(1) + steps * trend_seq[:, -1].unsqueeze(1)

            out, _st = self.rnn(resid)
            last = self.norm(out[:, -1, :])
            delta = self.head(last)
            return baseline + delta

    model = _ESRNNDirect()
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
