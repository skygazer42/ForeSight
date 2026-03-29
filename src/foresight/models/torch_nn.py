from __future__ import annotations

import contextvars
import copy
import json
import math
import time
import warnings
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from ..cli_runtime import compact_log_payload, emit_cli_event
from ..optional_deps import dependency_install_hint, require_dependency

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
_LOW_FREQ_BINS_MIN_MSG = "low_freq_bins must be >= 1"
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
_MIN_EPOCHS_MIN_MSG = "min_epochs must be >= 1"
_MIN_EPOCHS_MAX_EPOCHS_MSG = "min_epochs must be <= epochs"
_GRAD_ACCUM_STEPS_MIN_MSG = "grad_accum_steps must be >= 1"
_AMP_DTYPE_OPTIONS_MSG = "amp_dtype must be one of: auto, float16, bfloat16"
_AMP_REQUIRES_CUDA_MSG = "amp=True requires device='cuda'"
_MONITOR_OPTIONS_MSG = "monitor must be one of: auto, train_loss, val_loss"
_MONITOR_MODE_OPTIONS_MSG = "monitor_mode must be one of: min, max"
_MIN_DELTA_MIN_MSG = "min_delta must be >= 0"
_WARMUP_EPOCHS_MIN_MSG = "warmup_epochs must be >= 0"
_WARMUP_EPOCHS_MAX_EPOCHS_MSG = "warmup_epochs must be <= epochs"
_WARMUP_EPOCHS_ONECYCLE_MSG = "warmup_epochs requires scheduler != onecycle"
_EMA_DECAY_RANGE_MSG = "ema_decay must be in [0, 1)"
_EMA_WARMUP_EPOCHS_MIN_MSG = "ema_warmup_epochs must be >= 0"
_EMA_WARMUP_EPOCHS_MAX_EPOCHS_MSG = "ema_warmup_epochs must be <= epochs"
_SWA_START_EPOCH_MIN_MSG = "swa_start_epoch must be >= -1"
_SWA_START_EPOCH_MAX_EPOCHS_MSG = "swa_start_epoch must be <= epochs"
_EMA_SWA_CONFLICT_MSG = "ema_decay and swa_start_epoch cannot both be enabled"
_LOOKAHEAD_STEPS_MIN_MSG = "lookahead_steps must be >= 0"
_LOOKAHEAD_ALPHA_RANGE_MSG = "lookahead_alpha must be in (0, 1]"
_SAM_RHO_MIN_MSG = "sam_rho must be >= 0"
_SAM_REQUIRES_SINGLE_ACCUM_MSG = "sam_rho requires grad_accum_steps == 1"
_SAM_REQUIRES_AMP_DISABLED_MSG = "sam_rho requires amp=False"
_HORIZON_LOSS_DECAY_POSITIVE_MSG = "horizon_loss_decay must be > 0"
_INPUT_DROPOUT_RANGE_MSG = "input_dropout must be in [0, 1)"
_TEMPORAL_DROPOUT_RANGE_MSG = "temporal_dropout must be in [0, 1)"
_GRAD_NOISE_STD_MIN_MSG = "grad_noise_std must be >= 0"
_GC_MODE_OPTIONS_MSG = "gc_mode must be one of: off, all, conv_only"
_AGC_CLIP_FACTOR_MIN_MSG = "agc_clip_factor must be >= 0"
_AGC_EPS_POSITIVE_MSG = "agc_eps must be > 0"
_MIN_LR_MIN_MSG = "min_lr must be >= 0"
_NUM_WORKERS_MIN_MSG = "num_workers must be >= 0"
_PERSISTENT_WORKERS_REQUIRES_NUM_WORKERS_MSG = "persistent_workers requires num_workers >= 1"
_SCHEDULER_PATIENCE_MIN_MSG = "scheduler_patience must be >= 1"
_GRAD_CLIP_MODE_OPTIONS_MSG = "grad_clip_mode must be one of: norm, value"
_GRAD_CLIP_VALUE_MIN_MSG = "grad_clip_value must be >= 0"
_SCHEDULER_RESTART_PERIOD_MIN_MSG = "scheduler_restart_period must be >= 1"
_SCHEDULER_RESTART_MULT_MIN_MSG = "scheduler_restart_mult must be >= 1"
_SCHEDULER_PCT_START_RANGE_MSG = "scheduler_pct_start must be in (0, 1)"
_SCHEDULER_PLATEAU_FACTOR_RANGE_MSG = "scheduler_plateau_factor must be in (0, 1)"
_SCHEDULER_PLATEAU_THRESHOLD_MIN_MSG = "scheduler_plateau_threshold must be >= 0"
_VAL_MONITOR_REQUIRES_VAL_SPLIT_MSG = "monitor='val_loss' requires val_split > 0"
_CHECKPOINT_DIR_REQUIRED_MSG = "checkpoint_dir is required when checkpoint saving is enabled"
_RESUME_CHECKPOINT_PATH_MISSING_MSG = "resume_checkpoint_path does not exist"
_TENSORBOARD_FLUSH_SECS_MIN_MSG = "tensorboard_flush_secs must be >= 1"
_WANDB_MODE_OPTIONS_MSG = "wandb_mode must be one of: online, offline, disabled"
_SCHEDULER_OPTIONS_MSG = (
    "scheduler must be one of: none, cosine, step, plateau, onecycle, cosine_restarts"
)
_MLFLOW_DEFAULT_EXPERIMENT_NAME = "ForeSight"


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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The pynvml package is deprecated.*",
            category=FutureWarning,
        )
        try:
            return require_dependency("torch", install_hint='pip install -e ".[torch]"')
        except ImportError as e:
            raise ImportError(
                f"Torch models require PyTorch. Install with: {dependency_install_hint('torch')}"
            ) from e


def _make_transformer_encoder(
    *,
    nn: Any,
    layer: Any,
    num_layers: int,
) -> Any:
    try:
        return nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=int(num_layers))


def _require_mlflow() -> Any:
    try:
        import mlflow  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "MLflow tracking requires `mlflow`; install `mlflow` to enable tracking"
        ) from e
    return mlflow


def _require_wandb() -> Any:
    try:
        import wandb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "Weights & Biases tracking requires `wandb`; install `wandb` to enable tracking"
        ) from e
    return wandb


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

            _, T, _C = xb.shape
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

            _, T, _C = xb.shape
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
    scheduler_restart_period: int = 10
    scheduler_restart_mult: int = 1
    scheduler_pct_start: float = 0.3
    restore_best: bool = True
    min_epochs: int = 1
    amp: bool = False
    amp_dtype: str = "auto"
    warmup_epochs: int = 0
    min_lr: float = 0.0
    grad_accum_steps: int = 1
    monitor: str = "auto"
    monitor_mode: str = "min"
    min_delta: float = 0.0
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    scheduler_patience: int = 5
    grad_clip_mode: str = "norm"
    grad_clip_value: float = 0.0
    scheduler_plateau_factor: float = 0.1
    scheduler_plateau_threshold: float = 1e-4
    ema_decay: float = 0.0
    ema_warmup_epochs: int = 0
    swa_start_epoch: int = -1
    lookahead_steps: int = 0
    lookahead_alpha: float = 0.5
    sam_rho: float = 0.0
    sam_adaptive: bool = False
    horizon_loss_decay: float = 1.0
    input_dropout: float = 0.0
    temporal_dropout: float = 0.0
    grad_noise_std: float = 0.0
    gc_mode: str = "off"
    agc_clip_factor: float = 0.0
    agc_eps: float = 1e-3
    checkpoint_dir: str = ""
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    resume_checkpoint_path: str = ""
    resume_checkpoint_strict: bool = True
    tensorboard_log_dir: str = ""
    tensorboard_run_name: str = ""
    tensorboard_flush_secs: int = 10
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = ""
    mlflow_run_name: str = ""
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_dir: str = ""
    wandb_mode: str = "online"


@dataclass(frozen=True)
class TorchCheckpointResumeState:
    start_epoch: int = 0
    last_monitor: float | None = None
    best_monitor: float | None = None
    bad_epochs: int = 0
    best_epoch: int = -1
    best_state: dict[str, Any] | None = None
    base_lrs: tuple[float, ...] | None = None
    ema_state: dict[str, Any] | None = None
    swa_state: dict[str, Any] | None = None
    swa_n_averaged: int = 0
    lookahead_state: dict[str, Any] | None = None
    lookahead_step: int = 0


@dataclass
class TorchTrackingSession:
    backend: str
    handle: Any
    run_dir: str = ""
    run_name: str = ""
    metadata: dict[str, Any] | None = None


TorchBatchPredictFn = Callable[..., Any]
TorchOptimizerFactory = Callable[..., Any]
TorchSchedulerFactory = Callable[..., tuple[Any, str]]
_TORCH_RUNTIME_OVERRIDE_VAR: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "foresight_torch_runtime_override",
    default={},
)


@contextmanager
def torch_train_config_override(overrides: Mapping[str, Any] | None) -> Any:
    payload = {
        str(key): value
        for key, value in dict(overrides or {}).items()
        if value is not None and value != ""
    }
    token = _TORCH_RUNTIME_OVERRIDE_VAR.set(payload)
    try:
        yield
    finally:
        _TORCH_RUNTIME_OVERRIDE_VAR.reset(token)


def _merge_torch_runtime_override(cfg: TorchTrainConfig) -> TorchTrainConfig:
    overrides = _TORCH_RUNTIME_OVERRIDE_VAR.get()
    if not overrides:
        return cfg
    merged = dict(vars(cfg))
    merged.update(dict(overrides))
    return TorchTrainConfig(**merged)


def _validate_torch_train_config(cfg: TorchTrainConfig) -> None:
    if cfg.epochs <= 0:
        raise ValueError("epochs must be >= 1")
    if cfg.lr <= 0.0:
        raise ValueError("lr must be > 0")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    if int(cfg.min_epochs) <= 0:
        raise ValueError(_MIN_EPOCHS_MIN_MSG)
    if int(cfg.min_epochs) > int(cfg.epochs):
        raise ValueError(_MIN_EPOCHS_MAX_EPOCHS_MSG)
    if cfg.patience <= 0:
        raise ValueError("patience must be >= 1")
    if float(cfg.val_split) < 0.0 or float(cfg.val_split) >= 0.5:
        raise ValueError("val_split must be in [0, 0.5)")
    if float(cfg.grad_clip_norm) < 0.0:
        raise ValueError("grad_clip_norm must be >= 0")
    amp_dtype = str(cfg.amp_dtype).lower().strip()
    if amp_dtype not in {"auto", "float16", "bfloat16"}:
        raise ValueError(_AMP_DTYPE_OPTIONS_MSG)
    if int(cfg.warmup_epochs) < 0:
        raise ValueError(_WARMUP_EPOCHS_MIN_MSG)
    if int(cfg.warmup_epochs) > int(cfg.epochs):
        raise ValueError(_WARMUP_EPOCHS_MAX_EPOCHS_MSG)
    if str(cfg.scheduler).lower().strip() == "onecycle" and int(cfg.warmup_epochs) > 0:
        raise ValueError(_WARMUP_EPOCHS_ONECYCLE_MSG)
    if not (0.0 <= float(cfg.ema_decay) < 1.0):
        raise ValueError(_EMA_DECAY_RANGE_MSG)
    if int(cfg.ema_warmup_epochs) < 0:
        raise ValueError(_EMA_WARMUP_EPOCHS_MIN_MSG)
    if int(cfg.ema_warmup_epochs) > int(cfg.epochs):
        raise ValueError(_EMA_WARMUP_EPOCHS_MAX_EPOCHS_MSG)
    if int(cfg.swa_start_epoch) < -1:
        raise ValueError(_SWA_START_EPOCH_MIN_MSG)
    if int(cfg.swa_start_epoch) > int(cfg.epochs):
        raise ValueError(_SWA_START_EPOCH_MAX_EPOCHS_MSG)
    if float(cfg.ema_decay) > 0.0 and int(cfg.swa_start_epoch) >= 0:
        raise ValueError(_EMA_SWA_CONFLICT_MSG)
    if int(cfg.lookahead_steps) < 0:
        raise ValueError(_LOOKAHEAD_STEPS_MIN_MSG)
    if not (0.0 < float(cfg.lookahead_alpha) <= 1.0):
        raise ValueError(_LOOKAHEAD_ALPHA_RANGE_MSG)
    if float(cfg.sam_rho) < 0.0:
        raise ValueError(_SAM_RHO_MIN_MSG)
    if float(cfg.sam_rho) > 0.0 and int(cfg.grad_accum_steps) != 1:
        raise ValueError(_SAM_REQUIRES_SINGLE_ACCUM_MSG)
    if float(cfg.sam_rho) > 0.0 and bool(cfg.amp):
        raise ValueError(_SAM_REQUIRES_AMP_DISABLED_MSG)
    if float(cfg.horizon_loss_decay) <= 0.0:
        raise ValueError(_HORIZON_LOSS_DECAY_POSITIVE_MSG)
    if not (0.0 <= float(cfg.input_dropout) < 1.0):
        raise ValueError(_INPUT_DROPOUT_RANGE_MSG)
    if not (0.0 <= float(cfg.temporal_dropout) < 1.0):
        raise ValueError(_TEMPORAL_DROPOUT_RANGE_MSG)
    if float(cfg.grad_noise_std) < 0.0:
        raise ValueError(_GRAD_NOISE_STD_MIN_MSG)
    gc_mode = str(cfg.gc_mode).lower().strip()
    if gc_mode not in {"off", "all", "conv_only"}:
        raise ValueError(_GC_MODE_OPTIONS_MSG)
    if float(cfg.agc_clip_factor) < 0.0:
        raise ValueError(_AGC_CLIP_FACTOR_MIN_MSG)
    if float(cfg.agc_eps) <= 0.0:
        raise ValueError(_AGC_EPS_POSITIVE_MSG)
    if float(cfg.min_lr) < 0.0:
        raise ValueError(_MIN_LR_MIN_MSG)
    if int(cfg.scheduler_restart_period) <= 0:
        raise ValueError(_SCHEDULER_RESTART_PERIOD_MIN_MSG)
    if int(cfg.scheduler_restart_mult) <= 0:
        raise ValueError(_SCHEDULER_RESTART_MULT_MIN_MSG)
    if not (0.0 < float(cfg.scheduler_pct_start) < 1.0):
        raise ValueError(_SCHEDULER_PCT_START_RANGE_MSG)
    if int(cfg.grad_accum_steps) <= 0:
        raise ValueError(_GRAD_ACCUM_STEPS_MIN_MSG)
    monitor = str(cfg.monitor).lower().strip()
    if monitor not in {"auto", "train_loss", "val_loss"}:
        raise ValueError(_MONITOR_OPTIONS_MSG)
    monitor_mode = str(cfg.monitor_mode).lower().strip()
    if monitor_mode not in {"min", "max"}:
        raise ValueError(_MONITOR_MODE_OPTIONS_MSG)
    if float(cfg.min_delta) < 0.0:
        raise ValueError(_MIN_DELTA_MIN_MSG)
    if int(cfg.num_workers) < 0:
        raise ValueError(_NUM_WORKERS_MIN_MSG)
    if bool(cfg.persistent_workers) and int(cfg.num_workers) <= 0:
        raise ValueError(_PERSISTENT_WORKERS_REQUIRES_NUM_WORKERS_MSG)
    if int(cfg.scheduler_patience) <= 0:
        raise ValueError(_SCHEDULER_PATIENCE_MIN_MSG)
    if str(cfg.grad_clip_mode).lower().strip() not in {"norm", "value"}:
        raise ValueError(_GRAD_CLIP_MODE_OPTIONS_MSG)
    if float(cfg.grad_clip_value) < 0.0:
        raise ValueError(_GRAD_CLIP_VALUE_MIN_MSG)
    if not (0.0 < float(cfg.scheduler_plateau_factor) < 1.0):
        raise ValueError(_SCHEDULER_PLATEAU_FACTOR_RANGE_MSG)
    if float(cfg.scheduler_plateau_threshold) < 0.0:
        raise ValueError(_SCHEDULER_PLATEAU_THRESHOLD_MIN_MSG)
    if monitor == "val_loss" and float(cfg.val_split) <= 0.0:
        raise ValueError(_VAL_MONITOR_REQUIRES_VAL_SPLIT_MSG)
    checkpoint_dir = str(cfg.checkpoint_dir).strip()
    if (bool(cfg.save_best_checkpoint) or bool(cfg.save_last_checkpoint)) and not checkpoint_dir:
        raise ValueError(_CHECKPOINT_DIR_REQUIRED_MSG)
    resume_checkpoint_path = str(cfg.resume_checkpoint_path).strip()
    if resume_checkpoint_path and not Path(resume_checkpoint_path).is_file():
        raise ValueError(_RESUME_CHECKPOINT_PATH_MISSING_MSG)
    if int(cfg.tensorboard_flush_secs) <= 0:
        raise ValueError(_TENSORBOARD_FLUSH_SECS_MIN_MSG)
    wandb_mode = str(cfg.wandb_mode).lower().strip()
    if wandb_mode and wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError(_WANDB_MODE_OPTIONS_MSG)


def _validate_torch_train_config_kwargs(params: Mapping[str, Any]) -> None:
    cfg_kwargs = {
        field.name: params[field.name]
        for field in fields(TorchTrainConfig)
        if field.name in params
    }
    _validate_torch_train_config(TorchTrainConfig(**cfg_kwargs))


def _make_torch_dataloader(
    torch: Any,
    dataset: Any,
    *,
    cfg: TorchTrainConfig,
    shuffle: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "batch_size": int(cfg.batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.pin_memory),
    }
    if int(cfg.num_workers) > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
    return torch.utils.data.DataLoader(dataset, **kwargs)


def _make_torch_scheduler(
    torch: Any,
    opt: Any,
    *,
    cfg: TorchTrainConfig,
    steps_per_epoch: int | None = None,
) -> tuple[Any, str]:
    sched_name = str(cfg.scheduler).lower().strip()
    if sched_name in {"none", ""}:
        return None, "none"
    if sched_name == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=max(1, int(cfg.epochs) - int(cfg.warmup_epochs)),
                eta_min=float(cfg.min_lr),
            ),
            sched_name,
        )
    if sched_name == "step":
        return (
            torch.optim.lr_scheduler.StepLR(
                opt,
                step_size=int(cfg.scheduler_step_size),
                gamma=float(cfg.scheduler_gamma),
            ),
            sched_name,
        )
    if sched_name == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode=str(cfg.monitor_mode).lower().strip(),
                patience=int(cfg.scheduler_patience),
                factor=float(cfg.scheduler_plateau_factor),
                threshold=float(cfg.scheduler_plateau_threshold),
                min_lr=float(cfg.min_lr),
            ),
            sched_name,
        )
    if sched_name == "cosine_restarts":
        return (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt,
                T_0=int(cfg.scheduler_restart_period),
                T_mult=int(cfg.scheduler_restart_mult),
                eta_min=float(cfg.min_lr),
            ),
            sched_name,
        )
    if sched_name == "onecycle":
        if steps_per_epoch is None or int(steps_per_epoch) <= 0:
            raise ValueError("onecycle scheduler requires steps_per_epoch >= 1")
        final_div_factor = 1e4
        if float(cfg.min_lr) > 0.0:
            final_div_factor = max(1.0, float(cfg.lr) / (25.0 * float(cfg.min_lr)))
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=float(cfg.lr),
                epochs=int(cfg.epochs),
                steps_per_epoch=int(steps_per_epoch),
                pct_start=float(cfg.scheduler_pct_start),
                div_factor=25.0,
                final_div_factor=float(final_div_factor),
            ),
            sched_name,
        )
    raise ValueError(_SCHEDULER_OPTIONS_MSG)


def _resolve_torch_amp_dtype(torch: Any, *, cfg: TorchTrainConfig, dev: Any) -> Any:
    amp_dtype = str(cfg.amp_dtype).lower().strip()
    if amp_dtype == "auto":
        if dev.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported"):
            if bool(torch.cuda.is_bf16_supported()):
                return torch.bfloat16
        return torch.float16
    if amp_dtype == "float16":
        return torch.float16
    return torch.bfloat16


def _make_torch_amp_scaler(torch: Any, *, enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def _make_torch_autocast_context(
    torch: Any,
    *,
    enabled: bool,
    dev: Any,
    dtype: Any | None,
) -> Any:
    if not bool(enabled):
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=str(dev.type), dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


def _make_torch_amp_state(torch: Any, *, cfg: TorchTrainConfig, dev: Any) -> tuple[bool, Any, Any]:
    amp_enabled = bool(cfg.amp)
    if not amp_enabled:
        return False, None, None
    if dev.type != "cuda":
        raise ValueError(_AMP_REQUIRES_CUDA_MSG)
    amp_dtype = _resolve_torch_amp_dtype(torch, cfg=cfg, dev=dev)
    scaler_enabled = amp_dtype == torch.float16
    scaler = _make_torch_amp_scaler(torch, enabled=scaler_enabled)
    return True, amp_dtype, scaler


def _apply_torch_warmup(
    opt: Any,
    *,
    cfg: TorchTrainConfig,
    epoch_idx: int,
    base_lrs: tuple[float, ...],
) -> None:
    warmup_epochs = int(cfg.warmup_epochs)
    if warmup_epochs <= 0 or int(epoch_idx) >= warmup_epochs:
        return
    scale = float(int(epoch_idx) + 1) / float(warmup_epochs)
    min_lr = float(cfg.min_lr)
    for group, base_lr in zip(opt.param_groups, base_lrs, strict=True):
        group["lr"] = max(min_lr, float(base_lr) * scale)


def _clamp_torch_optimizer_min_lr(opt: Any, *, cfg: TorchTrainConfig) -> None:
    min_lr = float(cfg.min_lr)
    if min_lr <= 0.0:
        return
    for group in opt.param_groups:
        group["lr"] = max(min_lr, float(group["lr"]))


def _apply_torch_train_input_dropout(torch: Any, xb: Any, *, cfg: TorchTrainConfig) -> Any:
    p = float(cfg.input_dropout)
    if p <= 0.0:
        return xb
    if hasattr(xb, "is_floating_point") and not bool(xb.is_floating_point()):
        return xb
    return torch.nn.functional.dropout(xb, p=p, training=True)


def _apply_torch_train_temporal_dropout(torch: Any, xb: Any, *, cfg: TorchTrainConfig) -> Any:
    p = float(cfg.temporal_dropout)
    if p <= 0.0:
        return xb
    if hasattr(xb, "is_floating_point") and not bool(xb.is_floating_point()):
        return xb
    if int(getattr(xb, "ndim", 0)) < 2:
        return xb
    keep = 1.0 - p
    mask_shape = (int(xb.shape[0]), int(xb.shape[1])) + (1,) * max(0, int(xb.ndim) - 2)
    mask = (torch.rand(mask_shape, device=xb.device) >= p).to(dtype=xb.dtype)
    if keep > 0.0:
        mask = mask / keep
    return xb * mask


def _reduce_torch_horizon_loss(torch: Any, loss: Any, *, cfg: TorchTrainConfig) -> Any:
    if not hasattr(loss, "ndim"):
        return loss
    if int(loss.ndim) == 0:
        return loss
    if int(loss.ndim) < 2:
        return loss.mean()
    if int(loss.shape[1]) <= 1:
        return loss.mean()

    decay = float(cfg.horizon_loss_decay)
    if abs(decay - 1.0) < 1e-12:
        return loss.mean()

    steps = torch.arange(int(loss.shape[1]), device=loss.device, dtype=loss.dtype)
    weights = torch.pow(torch.as_tensor(decay, device=loss.device, dtype=loss.dtype), steps)
    weights = weights / weights.mean()
    view_shape = (1, int(loss.shape[1])) + (1,) * max(0, int(loss.ndim) - 2)
    return (loss * weights.reshape(view_shape)).mean()


def _make_torch_unreduced_loss_fn(nn: Any, loss: str) -> Any:
    loss_name = str(loss).lower().strip()
    if loss_name in {"mse", ""}:
        return nn.MSELoss(reduction="none")
    if loss_name in {"mae", "l1"}:
        return nn.L1Loss(reduction="none")
    if loss_name in {"huber", "smoothl1"}:
        return nn.SmoothL1Loss(reduction="none")
    raise ValueError("loss must be one of: mse, mae, huber")


def _make_torch_loss_fn(
    torch: Any,
    nn: Any,
    *,
    cfg: TorchTrainConfig,
    loss_fn_override: Any | None = None,
) -> Any:
    raw_loss_fn = (
        loss_fn_override
        if loss_fn_override is not None
        else _make_torch_unreduced_loss_fn(nn, cfg.loss)
    )

    def _loss(pred: Any, yb: Any) -> Any:
        loss = raw_loss_fn(pred, yb)
        return _reduce_torch_horizon_loss(torch, loss, cfg=cfg)

    return _loss


def _apply_torch_gradient_clipping(torch: Any, model: Any, *, cfg: TorchTrainConfig) -> None:
    _apply_torch_gradient_centralization(torch, model, cfg=cfg)
    _apply_torch_gradient_noise(torch, model, cfg=cfg)
    _apply_torch_agc(torch, model, cfg=cfg)

    mode = str(cfg.grad_clip_mode).lower().strip()
    if mode == "value":
        clip_value = float(cfg.grad_clip_value)
        if clip_value > 0.0:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
        return

    max_norm = float(cfg.grad_clip_norm)
    if max_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def _apply_torch_gradient_noise(torch: Any, model: Any, *, cfg: TorchTrainConfig) -> None:
    std = float(cfg.grad_noise_std)
    if std <= 0.0:
        return

    with torch.no_grad():
        for param in model.parameters():
            grad = getattr(param, "grad", None)
            if grad is None or bool(getattr(grad, "is_sparse", False)):
                continue
            if hasattr(grad, "is_floating_point") and not bool(grad.is_floating_point()):
                continue
            grad.add_(torch.randn_like(grad) * std)


def _torch_scheduler_steps_per_batch(sched_name: str) -> bool:
    return str(sched_name).lower().strip() == "onecycle"


def _select_torch_monitor_value(
    cfg: TorchTrainConfig,
    *,
    train_loss: float,
    val_loss: float | None,
) -> float:
    monitor = str(cfg.monitor).lower().strip()
    if monitor == "train_loss":
        return float(train_loss)
    if monitor == "val_loss":
        if val_loss is None:
            raise ValueError(_VAL_MONITOR_REQUIRES_VAL_SPLIT_MSG)
        return float(val_loss)
    if val_loss is not None:
        return float(val_loss)
    return float(train_loss)


def _torch_monitor_improved(*, value: float, best: float, cfg: TorchTrainConfig) -> bool:
    min_delta = float(cfg.min_delta)
    if str(cfg.monitor_mode).lower().strip() == "max":
        return float(value) > float(best) + min_delta
    return float(value) < float(best) - min_delta


def _clone_torch_checkpoint_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    return value


def _clone_torch_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        if hasattr(value, "detach"):
            out[key] = value.detach().clone()
        else:
            out[key] = copy.deepcopy(value)
    return out


def _clone_torch_state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key: _clone_torch_checkpoint_value(value) for key, value in state_dict.items()}


def _make_torch_ema_model(model: Any, *, cfg: TorchTrainConfig) -> Any | None:
    if float(cfg.ema_decay) <= 0.0:
        return None
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    return ema_model


def _torch_ema_active_for_epoch(*, cfg: TorchTrainConfig, epoch_idx: int) -> bool:
    return float(cfg.ema_decay) > 0.0 and int(epoch_idx) >= int(cfg.ema_warmup_epochs)


def _update_torch_ema_model(torch: Any, *, ema_model: Any, model: Any, cfg: TorchTrainConfig) -> None:
    decay = float(cfg.ema_decay)
    if decay <= 0.0:
        return

    ema_params = dict(ema_model.named_parameters())
    ema_buffers = dict(ema_model.named_buffers())
    with torch.no_grad():
        for name, param in model.named_parameters():
            ema_params[name].lerp_(param.detach(), weight=1.0 - decay)
        for name, buf in model.named_buffers():
            ema_buf = ema_buffers[name]
            src = buf.detach()
            if torch.is_floating_point(src) or torch.is_complex(src):
                ema_buf.lerp_(src, weight=1.0 - decay)
            else:
                ema_buf.copy_(src)


def _make_torch_swa_model(model: Any, *, cfg: TorchTrainConfig) -> Any | None:
    if int(cfg.swa_start_epoch) < 0:
        return None
    swa_model = copy.deepcopy(model)
    swa_model.requires_grad_(False)
    swa_model.eval()
    return swa_model


def _torch_swa_active_for_epoch(*, cfg: TorchTrainConfig, epoch_idx: int) -> bool:
    return int(cfg.swa_start_epoch) >= 0 and int(epoch_idx) >= int(cfg.swa_start_epoch)


def _update_torch_swa_model(
    torch: Any,
    *,
    swa_model: Any,
    model: Any,
    n_averaged: int,
) -> int:
    next_n_averaged = int(n_averaged) + 1
    update_weight = 1.0 / float(next_n_averaged)
    swa_params = dict(swa_model.named_parameters())
    swa_buffers = dict(swa_model.named_buffers())
    with torch.no_grad():
        for name, param in model.named_parameters():
            swa_param = swa_params[name]
            src = param.detach()
            if torch.is_floating_point(src) or torch.is_complex(src):
                swa_param.lerp_(src, weight=update_weight)
            else:
                swa_param.copy_(src)
        for name, buf in model.named_buffers():
            swa_buf = swa_buffers[name]
            src = buf.detach()
            if torch.is_floating_point(src) or torch.is_complex(src):
                swa_buf.lerp_(src, weight=update_weight)
            else:
                swa_buf.copy_(src)
    return next_n_averaged


def _make_torch_lookahead_model(model: Any, *, cfg: TorchTrainConfig) -> Any | None:
    if int(cfg.lookahead_steps) <= 0:
        return None
    lookahead_model = copy.deepcopy(model)
    lookahead_model.requires_grad_(False)
    lookahead_model.eval()
    return lookahead_model


def _torch_lookahead_active(*, cfg: TorchTrainConfig, lookahead_step: int) -> bool:
    return int(cfg.lookahead_steps) > 0 and int(lookahead_step) >= int(cfg.lookahead_steps)


def _torch_sam_active(*, cfg: TorchTrainConfig) -> bool:
    return float(cfg.sam_rho) > 0.0


def _torch_gc_mode_min_ndim(*, cfg: TorchTrainConfig) -> int | None:
    mode = str(cfg.gc_mode).lower().strip()
    if mode == "off":
        return None
    if mode == "all":
        return 2
    if mode == "conv_only":
        return 3
    raise ValueError(_GC_MODE_OPTIONS_MSG)


def _apply_torch_gradient_centralization(torch: Any, model: Any, *, cfg: TorchTrainConfig) -> None:
    min_ndim = _torch_gc_mode_min_ndim(cfg=cfg)
    if min_ndim is None:
        return

    with torch.no_grad():
        for param in model.parameters():
            grad = getattr(param, "grad", None)
            if grad is None or int(grad.ndim) < int(min_ndim):
                continue
            dims = tuple(range(1, int(grad.ndim)))
            grad.sub_(grad.mean(dim=dims, keepdim=True))


def _torch_agc_active(*, cfg: TorchTrainConfig) -> bool:
    return float(cfg.agc_clip_factor) > 0.0


def _torch_agc_unitwise_norm(torch: Any, tensor: Any) -> Any:
    if int(tensor.ndim) <= 1:
        return torch.norm(tensor, p=2)
    dims = tuple(range(1, int(tensor.ndim)))
    return torch.norm(tensor, p=2, dim=dims, keepdim=True)


def _apply_torch_agc(torch: Any, model: Any, *, cfg: TorchTrainConfig) -> None:
    clip_factor = float(cfg.agc_clip_factor)
    if clip_factor <= 0.0:
        return

    eps = float(cfg.agc_eps)
    with torch.no_grad():
        for param in model.parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            grad_norm = _torch_agc_unitwise_norm(torch, grad.detach())
            param_norm = _torch_agc_unitwise_norm(torch, param.detach())
            max_norm = torch.clamp(param_norm, min=eps) * clip_factor
            clip_coef = max_norm / torch.clamp(grad_norm, min=1e-6)
            grad.mul_(torch.clamp(clip_coef, max=1.0))


def _torch_sam_gradient_norm(torch: Any, *, model: Any, cfg: TorchTrainConfig) -> Any | None:
    adaptive = bool(cfg.sam_adaptive)
    norms: list[Any] = []
    for param in model.parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        grad_detached = grad.detach()
        if adaptive:
            grad_detached = torch.abs(param.detach()) * grad_detached
        norms.append(grad_detached.norm(p=2))
    if not norms:
        return None
    return torch.norm(torch.stack(norms), p=2)


def _apply_torch_sam_perturbation(
    torch: Any,
    *,
    model: Any,
    cfg: TorchTrainConfig,
) -> list[tuple[Any, Any]]:
    grad_norm = _torch_sam_gradient_norm(torch, model=model, cfg=cfg)
    if grad_norm is None:
        return []
    grad_norm_value = float(grad_norm.detach().cpu().item())
    if grad_norm_value <= 0.0:
        return []

    scale = float(cfg.sam_rho) / (grad_norm + 1e-12)
    perturbations: list[tuple[Any, Any]] = []
    adaptive = bool(cfg.sam_adaptive)
    with torch.no_grad():
        for param in model.parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            direction = grad.detach()
            if adaptive:
                direction = torch.pow(param.detach(), 2) * direction
            e_w = direction * scale.to(device=param.device, dtype=param.dtype)
            param.add_(e_w)
            perturbations.append((param, e_w))
    return perturbations


def _restore_torch_sam_perturbation(
    torch: Any,
    *,
    perturbations: list[tuple[Any, Any]],
) -> None:
    if not perturbations:
        return
    with torch.no_grad():
        for param, e_w in perturbations:
            param.sub_(e_w)


def _update_torch_lookahead_model(
    torch: Any,
    *,
    lookahead_model: Any,
    model: Any,
    cfg: TorchTrainConfig,
    lookahead_step: int,
) -> int:
    next_step = int(lookahead_step) + 1
    sync_steps = int(cfg.lookahead_steps)
    if sync_steps <= 0 or next_step % sync_steps != 0:
        return next_step

    alpha = float(cfg.lookahead_alpha)
    slow_params = dict(lookahead_model.named_parameters())
    slow_buffers = dict(lookahead_model.named_buffers())
    model_params = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name, slow_param in slow_params.items():
            fast_param = model_params[name]
            slow_param.lerp_(fast_param.detach(), weight=alpha)
            fast_param.copy_(slow_param)
        for name, slow_buf in slow_buffers.items():
            fast_buf = model_buffers[name]
            slow_buf.copy_(fast_buf.detach())
            fast_buf.copy_(slow_buf)
    return next_step


def _select_torch_deploy_model(
    *,
    model: Any,
    cfg: TorchTrainConfig,
    ema_model: Any | None = None,
    ema_active: bool = False,
    swa_model: Any | None = None,
    swa_n_averaged: int = 0,
    lookahead_model: Any | None = None,
    lookahead_step: int = 0,
) -> Any:
    if swa_model is not None and int(swa_n_averaged) > 0:
        return swa_model
    if ema_model is not None and bool(ema_active):
        return ema_model
    if lookahead_model is not None and _torch_lookahead_active(
        cfg=cfg,
        lookahead_step=int(lookahead_step),
    ):
        return lookahead_model
    return model


def _maybe_torch_model_state_for_checkpoint(
    *,
    model: Any,
    cfg: TorchTrainConfig,
    ema_model: Any | None = None,
    ema_active: bool = False,
    swa_model: Any | None = None,
    swa_n_averaged: int = 0,
    lookahead_model: Any | None = None,
    lookahead_step: int = 0,
) -> dict[str, Any] | None:
    if lookahead_model is not None:
        return model.state_dict()
    deploy_model = _select_torch_deploy_model(
        model=model,
        cfg=cfg,
        ema_model=ema_model,
        ema_active=bool(ema_active),
        swa_model=swa_model,
        swa_n_averaged=int(swa_n_averaged),
        lookahead_model=lookahead_model,
        lookahead_step=int(lookahead_step),
    )
    if deploy_model is model:
        return None
    return model.state_dict()


def _save_torch_checkpoint(
    torch: Any,
    *,
    checkpoint_dir: str,
    filename: str,
    state_dict: dict[str, Any],
    monitor: float,
    epoch: int,
    extra_payload: dict[str, Any] | None = None,
) -> str:
    checkpoint_path = Path(str(checkpoint_dir).strip())
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    target_path = checkpoint_path / str(filename)
    payload = {
        "state_dict": state_dict,
        "monitor": float(monitor),
        "epoch": int(epoch),
    }
    if extra_payload is not None:
        payload.update(extra_payload)
    torch.save(
        payload,
        target_path,
    )
    emit_cli_event(
        "CHECKPOINT saved",
        event="train_checkpoint_saved",
        payload=compact_log_payload(
            path=target_path.as_posix(),
            epoch=int(epoch),
            monitor=float(monitor),
        ),
    )
    return target_path.as_posix()


def _snapshot_torch_training_state(
    *,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    best_state: dict[str, Any] | None,
    best_monitor: float,
    bad_epochs: int,
    best_epoch: int,
    base_lrs: tuple[float, ...],
    ema_state: dict[str, Any] | None = None,
    swa_state: dict[str, Any] | None = None,
    swa_n_averaged: int = 0,
    lookahead_state: dict[str, Any] | None = None,
    lookahead_step: int = 0,
    model_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "best_monitor": float(best_monitor),
        "bad_epochs": int(bad_epochs),
        "best_epoch": int(best_epoch),
        "base_lrs": tuple(float(lr) for lr in base_lrs),
    }
    if scheduler is not None:
        payload["scheduler_state"] = copy.deepcopy(scheduler.state_dict())
    if scaler is not None:
        payload["scaler_state"] = copy.deepcopy(scaler.state_dict())
    if best_state is not None:
        payload["best_state"] = _clone_torch_state_dict_to_cpu(best_state)
    if ema_state is not None:
        payload["ema_state"] = _clone_torch_state_dict_to_cpu(ema_state)
    if swa_state is not None:
        payload["swa_state"] = _clone_torch_state_dict_to_cpu(swa_state)
        payload["swa_n_averaged"] = int(swa_n_averaged)
    if lookahead_state is not None:
        payload["lookahead_state"] = _clone_torch_state_dict_to_cpu(lookahead_state)
        payload["lookahead_step"] = int(lookahead_step)
    if model_state is not None:
        payload["model_state"] = _clone_torch_state_dict_to_cpu(model_state)
    return payload


def _maybe_save_torch_checkpoints(
    torch: Any,
    *,
    cfg: TorchTrainConfig,
    best_state: dict[str, Any] | None,
    best_monitor: float,
    best_epoch: int,
    last_state: dict[str, Any] | None,
    last_monitor: float | None,
    last_epoch: int,
    best_extra_payload: dict[str, Any] | None = None,
    last_extra_payload: dict[str, Any] | None = None,
) -> dict[str, str]:
    saved_paths: dict[str, str] = {}
    if bool(cfg.save_best_checkpoint) and best_state is not None:
        saved_paths["best_checkpoint_path"] = _save_torch_checkpoint(
            torch,
            checkpoint_dir=str(cfg.checkpoint_dir),
            filename="best.pt",
            state_dict=best_state,
            monitor=float(best_monitor),
            epoch=int(best_epoch),
            extra_payload=best_extra_payload,
        )
    if bool(cfg.save_last_checkpoint) and last_state is not None and last_monitor is not None:
        saved_paths["last_checkpoint_path"] = _save_torch_checkpoint(
            torch,
            checkpoint_dir=str(cfg.checkpoint_dir),
            filename="last.pt",
            state_dict=last_state,
            monitor=float(last_monitor),
            epoch=int(last_epoch),
            extra_payload=last_extra_payload,
        )
    return saved_paths


def _extract_torch_checkpoint_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint payload must contain a state_dict mapping")
    return dict(state_dict)


def _load_torch_checkpoint_into_model(
    torch: Any,
    model: Any,
    *,
    cfg: TorchTrainConfig | None = None,
    checkpoint_path: str = "",
    strict: bool = True,
) -> None:
    resume_checkpoint_path = str(checkpoint_path).strip()
    resume_strict = bool(strict)
    if cfg is not None:
        resume_checkpoint_path = str(cfg.resume_checkpoint_path).strip()
        resume_strict = bool(cfg.resume_checkpoint_strict)
    if not resume_checkpoint_path:
        return
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        payload = torch.load(resume_checkpoint_path, weights_only=True, **load_kwargs)
    except TypeError:
        payload = torch.load(resume_checkpoint_path, **load_kwargs)
    state_dict = _extract_torch_checkpoint_state_dict(payload)
    model.load_state_dict(state_dict, strict=resume_strict)


def _load_torch_training_state(
    torch: Any,
    model: Any,
    *,
    cfg: TorchTrainConfig | None = None,
    checkpoint_path: str = "",
    strict: bool = True,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> TorchCheckpointResumeState:
    resume_checkpoint_path = str(checkpoint_path).strip()
    resume_strict = bool(strict)
    if cfg is not None:
        resume_checkpoint_path = str(cfg.resume_checkpoint_path).strip()
        resume_strict = bool(cfg.resume_checkpoint_strict)
    if not resume_checkpoint_path:
        return TorchCheckpointResumeState()

    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        payload = torch.load(resume_checkpoint_path, weights_only=True, **load_kwargs)
    except TypeError:
        payload = torch.load(resume_checkpoint_path, **load_kwargs)
    emit_cli_event(
        "CHECKPOINT resume",
        event="train_checkpoint_resumed",
        payload=compact_log_payload(path=resume_checkpoint_path),
    )

    state_dict = _extract_torch_checkpoint_state_dict(payload)
    resume_model_state = state_dict
    if isinstance(payload, dict) and "model_state" in payload and isinstance(payload["model_state"], dict):
        resume_model_state = dict(payload["model_state"])
    model.load_state_dict(resume_model_state, strict=resume_strict)

    if not isinstance(payload, dict):
        return TorchCheckpointResumeState()

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and "scaler_state" in payload:
        scaler.load_state_dict(payload["scaler_state"])

    best_state = None
    if "best_state" in payload and isinstance(payload["best_state"], dict):
        best_state = _clone_torch_state_dict_to_cpu(payload["best_state"])

    base_lrs = None
    if "base_lrs" in payload:
        raw_base_lrs = payload["base_lrs"]
        if isinstance(raw_base_lrs, list | tuple):
            base_lrs = tuple(float(lr) for lr in raw_base_lrs)

    ema_state = None
    if "ema_state" in payload and isinstance(payload["ema_state"], dict):
        ema_state = _clone_torch_state_dict_to_cpu(payload["ema_state"])

    swa_state = None
    if "swa_state" in payload and isinstance(payload["swa_state"], dict):
        swa_state = _clone_torch_state_dict_to_cpu(payload["swa_state"])
    swa_n_averaged = int(payload.get("swa_n_averaged", 0))
    lookahead_state = None
    if "lookahead_state" in payload and isinstance(payload["lookahead_state"], dict):
        lookahead_state = _clone_torch_state_dict_to_cpu(payload["lookahead_state"])
    lookahead_step = int(payload.get("lookahead_step", 0))

    last_monitor = None
    if "monitor" in payload:
        last_monitor = float(payload["monitor"])

    best_monitor = None
    if "best_monitor" in payload:
        best_monitor = float(payload["best_monitor"])
    elif last_monitor is not None:
        best_monitor = float(last_monitor)

    best_epoch = -1
    if "best_epoch" in payload:
        best_epoch = int(payload["best_epoch"])
    elif best_monitor is not None:
        best_epoch = int(payload.get("epoch", -1))

    return TorchCheckpointResumeState(
        start_epoch=max(0, int(payload.get("epoch", 0))),
        last_monitor=last_monitor,
        best_monitor=best_monitor,
        bad_epochs=int(payload.get("bad_epochs", 0)),
        best_epoch=best_epoch,
        best_state=best_state,
        base_lrs=base_lrs,
        ema_state=ema_state,
        swa_state=swa_state,
        swa_n_averaged=swa_n_averaged,
        lookahead_state=lookahead_state,
        lookahead_step=lookahead_step,
    )


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


def _make_torch_optimizer(torch: Any, model: Any, *, cfg: TorchTrainConfig) -> Any:
    opt_name = str(cfg.optimizer).lower().strip()
    if opt_name in {"adam", ""}:
        return torch.optim.Adam(
            model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
        )
    if opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
        )
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.lr),
            momentum=float(cfg.momentum),
            weight_decay=float(cfg.weight_decay),
        )
    raise ValueError("optimizer must be one of: adam, adamw, sgd")


def _split_torch_batch(batch: Any) -> tuple[tuple[Any, ...], Any]:
    if not isinstance(batch, tuple | list):
        raise TypeError("torch training batches must be tuples/lists of tensors")
    items = tuple(batch)
    if len(items) < 2:
        raise ValueError("torch training batches must include model inputs and target")
    return items[:-1], items[-1]


def _move_torch_batch_to_device(
    batch: Any,
    *,
    dev: Any,
    non_blocking: bool,
) -> tuple[tuple[Any, ...], Any]:
    model_inputs, target = _split_torch_batch(batch)
    moved_inputs = tuple(item.to(dev, non_blocking=non_blocking) for item in model_inputs)
    moved_target = target.to(dev, non_blocking=non_blocking)
    return moved_inputs, moved_target


def _torch_loader_sample_count(loader: Any | None) -> int | None:
    if loader is None:
        return None
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    try:
        return int(len(dataset))
    except TypeError:
        return None


def _torch_parameter_counts(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if bool(parameter.requires_grad):
            trainable += count
    return trainable, total


def _torch_primary_lr(optimizer: Any) -> float | None:
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list | tuple) or not param_groups:
        return None
    first_group = param_groups[0]
    if not isinstance(first_group, dict) or "lr" not in first_group:
        return None
    return float(first_group["lr"])


def _torch_global_gradient_norm_value(torch: Any, *, model: Any) -> float | None:
    norms: list[Any] = []
    for param in model.parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        norms.append(grad.detach().norm(p=2))
    if not norms:
        return None
    grad_norm = torch.norm(torch.stack(norms), p=2)
    return float(grad_norm.detach().cpu().item())


def _torch_device_log_payload(torch: Any, *, dev: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "device_type": str(dev.type),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if str(dev.type) == "cuda" and bool(torch.cuda.is_available()):
        device_index = dev.index
        if device_index is None:
            device_index = int(torch.cuda.current_device())
        payload["cuda_device_index"] = int(device_index)
        payload["cuda_device_name"] = str(torch.cuda.get_device_name(device_index))
    return payload


def _torch_cuda_memory_payload(torch: Any, *, dev: Any) -> dict[str, Any]:
    if str(dev.type) != "cuda" or not bool(torch.cuda.is_available()):
        return {}
    device_index = dev.index
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    mib = float(1024**2)
    return {
        "memory_allocated_mb": float(torch.cuda.memory_allocated(device_index)) / mib,
        "memory_reserved_mb": float(torch.cuda.memory_reserved(device_index)) / mib,
        "peak_memory_allocated_mb": float(torch.cuda.max_memory_allocated(device_index)) / mib,
        "peak_memory_reserved_mb": float(torch.cuda.max_memory_reserved(device_index)) / mib,
    }


def _resolve_torch_tensorboard_run_dir(*, cfg: TorchTrainConfig) -> str:
    root_dir = str(cfg.tensorboard_log_dir).strip()
    if not root_dir:
        return ""
    run_name = str(cfg.tensorboard_run_name).strip() or time.strftime("run-%Y%m%d-%H%M%S")
    return (Path(root_dir).expanduser().resolve(strict=False) / run_name).as_posix()


def _open_torch_tensorboard_writer(*, cfg: TorchTrainConfig) -> tuple[Any | None, str]:
    run_dir = _resolve_torch_tensorboard_run_dir(cfg=cfg)
    if not run_dir:
        return None, ""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "tensorboard_log_dir requires torch.utils.tensorboard.SummaryWriter; install `tensorboard` to enable tracking"
        ) from exc
    return SummaryWriter(log_dir=run_dir, flush_secs=int(cfg.tensorboard_flush_secs)), run_dir


@contextmanager
def _torch_tensorboard_tracking_session(*, cfg: TorchTrainConfig) -> Any:
    writer, run_dir = _open_torch_tensorboard_writer(cfg=cfg)
    try:
        yield writer, run_dir
    finally:
        if writer is None:
            return
        flush = getattr(writer, "flush", None)
        if callable(flush):
            flush()
        close = getattr(writer, "close", None)
        if callable(close):
            close()


def _torch_tracking_add_text(
    writer: Any | None,
    *,
    tag: str,
    payload: Mapping[str, Any] | str,
    global_step: int | None = None,
) -> None:
    if writer is None:
        return
    add_text = getattr(writer, "add_text", None)
    if not callable(add_text):
        return
    text = (
        str(payload)
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    )
    if global_step is None:
        add_text(str(tag), text)
        return
    add_text(str(tag), text, int(global_step))


def _torch_tracking_add_scalar(
    writer: Any | None,
    *,
    tag: str,
    value: Any,
    global_step: int,
) -> None:
    if writer is None or value is None:
        return
    add_scalar = getattr(writer, "add_scalar", None)
    if not callable(add_scalar):
        return
    add_scalar(str(tag), float(value), int(global_step))


def _normalize_torch_tracking_payload_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_torch_tracking_payload_value(item) for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_normalize_torch_tracking_payload_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_normalize_torch_tracking_payload_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _normalize_torch_tracking_payload_value(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _coerce_torch_tracking_hparam_value(value: Any) -> bool | int | float | str:
    normalized = _normalize_torch_tracking_payload_value(value)
    if isinstance(normalized, bool):
        return normalized
    if isinstance(normalized, int):
        return int(normalized)
    if isinstance(normalized, float):
        return float(normalized) if math.isfinite(normalized) else str(normalized)
    if isinstance(normalized, str):
        return normalized
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _torch_tracking_numeric_metrics(
    payload: Mapping[str, Any] | None,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for key, value in compact_log_payload(payload or {}).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            metrics[str(key)] = int(value)
            continue
        if isinstance(value, float) and math.isfinite(value):
            metrics[str(key)] = float(value)
    return metrics


def _build_torch_tracking_hparams_payload(*, cfg: TorchTrainConfig) -> dict[str, bool | int | float | str]:
    return {
        str(key): _coerce_torch_tracking_hparam_value(value)
        for key, value in compact_log_payload(vars(cfg)).items()
    }


def _build_torch_tracking_artifact_payload(
    *,
    cfg: TorchTrainConfig,
    tracking_session: TorchTrackingSession | None = None,
    checkpoint_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    checkpoint_dir = str(cfg.checkpoint_dir).strip()
    saved_paths = {str(key): str(value) for key, value in dict(checkpoint_paths or {}).items()}
    payload = compact_log_payload(
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint_path=str(cfg.resume_checkpoint_path).strip(),
        best_checkpoint_path=saved_paths.get("best_checkpoint_path"),
        last_checkpoint_path=saved_paths.get("last_checkpoint_path"),
    )
    if tracking_session is None:
        return payload
    session_payload = dict(tracking_session.metadata or {})
    backend = str(tracking_session.backend)
    if backend == "tensorboard":
        payload["tensorboard_run_dir"] = str(tracking_session.run_dir)
    elif backend == "mlflow":
        payload.update(
            compact_log_payload(
                mlflow_tracking_uri=session_payload.get("tracking_uri"),
                mlflow_experiment_name=session_payload.get("experiment_name"),
                mlflow_run_name=session_payload.get("run_name"),
                mlflow_run_id=session_payload.get("run_id"),
            )
        )
    elif backend == "wandb":
        payload.update(
            compact_log_payload(
                wandb_project=session_payload.get("project"),
                wandb_entity=session_payload.get("entity"),
                wandb_run_name=session_payload.get("run_name"),
                wandb_run_id=session_payload.get("run_id"),
                wandb_run_path=session_payload.get("run_path"),
                wandb_dir=session_payload.get("directory"),
                wandb_mode=session_payload.get("mode"),
            )
        )
    return payload


def _torch_tracking_add_hparams(
    writer: Any | None,
    *,
    hparam_payload: Mapping[str, Any],
    metric_payload: Mapping[str, Any],
    run_name: str,
) -> None:
    if writer is None:
        return
    add_hparams = getattr(writer, "add_hparams", None)
    if not callable(add_hparams):
        return
    try:
        add_hparams(dict(hparam_payload), dict(metric_payload), run_name=str(run_name))
    except TypeError:
        add_hparams(dict(hparam_payload), dict(metric_payload))


def _build_torch_tracking_metric_payload(
    *,
    train_completed_payload: Mapping[str, Any],
    epoch_payload: Mapping[str, Any] | None,
) -> dict[str, float | int]:
    return _torch_tracking_numeric_metrics(
        compact_log_payload(
            {
                "train/epochs_ran": train_completed_payload.get("epochs_ran"),
                "monitor/final_best": train_completed_payload.get("best"),
                "train/final_lr": train_completed_payload.get("final_lr"),
                "time/total_seconds": train_completed_payload.get("total_seconds"),
                "train/final_loss": (None if epoch_payload is None else epoch_payload.get("train_loss")),
                "validation/final_loss": (None if epoch_payload is None else epoch_payload.get("val_loss")),
                "monitor/final_value": (None if epoch_payload is None else epoch_payload.get("monitor")),
            }
        )
    )


def _torch_tracking_artifact_file(*, tag: str, suffix: str) -> str:
    parts = [part.strip().replace(" ", "_") for part in str(tag).split("/") if part.strip()]
    if not parts:
        parts = ["payload"]
    return f"tracking/{'/'.join(parts)}.{suffix}"


def _torch_tracking_session_payload(session: TorchTrackingSession) -> dict[str, Any]:
    return compact_log_payload(dict(session.metadata or {}), backend=str(session.backend))


def _torch_tracking_summary_assign(summary: Any, *, key: str, value: Any) -> None:
    if summary is None:
        return
    try:
        summary[str(key)] = value
    except Exception:  # noqa: BLE001
        update = getattr(summary, "update", None)
        if callable(update):
            update({str(key): value})


@contextmanager
def _torch_mlflow_tracking_session(*, cfg: TorchTrainConfig) -> Any:
    tracking_uri = str(cfg.mlflow_tracking_uri).strip()
    experiment_name = str(cfg.mlflow_experiment_name).strip()
    run_name = str(cfg.mlflow_run_name).strip() or time.strftime("run-%Y%m%d-%H%M%S")
    if not tracking_uri and not experiment_name and not str(cfg.mlflow_run_name).strip():
        yield None
        return
    mlflow = _require_mlflow()
    if tracking_uri:
        set_tracking_uri = getattr(mlflow, "set_tracking_uri", None)
        if callable(set_tracking_uri):
            set_tracking_uri(str(tracking_uri))
    resolved_tracking_uri = tracking_uri
    get_tracking_uri = getattr(mlflow, "get_tracking_uri", None)
    if callable(get_tracking_uri):
        resolved_tracking_uri = str(get_tracking_uri())
    resolved_experiment_name = experiment_name or _MLFLOW_DEFAULT_EXPERIMENT_NAME
    set_experiment = getattr(mlflow, "set_experiment", None)
    if callable(set_experiment):
        set_experiment(str(resolved_experiment_name))
    start_run = getattr(mlflow, "start_run", None)
    if not callable(start_run):
        raise AttributeError("mlflow.start_run is required for MLflow tracking")
    active_run = start_run(run_name=str(run_name))
    run_info = getattr(active_run, "info", None)
    session = TorchTrackingSession(
        backend="mlflow",
        handle=mlflow,
        run_name=str(run_name),
        metadata=compact_log_payload(
            tracking_uri=resolved_tracking_uri,
            experiment_name=resolved_experiment_name,
            run_name=str(run_name),
            run_id=(None if run_info is None else getattr(run_info, "run_id", None)),
        ),
    )
    try:
        yield session
    finally:
        end_run = getattr(mlflow, "end_run", None)
        if callable(end_run):
            end_run()


@contextmanager
def _torch_wandb_tracking_session(*, cfg: TorchTrainConfig) -> Any:
    project = str(cfg.wandb_project).strip()
    if not project:
        yield None
        return
    wandb = _require_wandb()
    run_name = str(cfg.wandb_run_name).strip() or None
    entity = str(cfg.wandb_entity).strip() or None
    directory = str(cfg.wandb_dir).strip() or None
    mode = str(cfg.wandb_mode).strip() or None
    init = getattr(wandb, "init", None)
    if not callable(init):
        raise AttributeError("wandb.init is required for Weights & Biases tracking")
    run = init(
        project=str(project),
        entity=entity,
        name=run_name,
        dir=directory,
        mode=mode,
        config=_build_torch_tracking_hparams_payload(cfg=cfg),
    )
    resolved_run_name = run_name or getattr(run, "name", None) or ""
    session = TorchTrackingSession(
        backend="wandb",
        handle=run,
        run_name=str(resolved_run_name),
        metadata=compact_log_payload(
            project=str(project),
            entity=entity,
            run_name=resolved_run_name,
            run_id=getattr(run, "id", None),
            run_path=getattr(run, "path", None),
            directory=directory,
            mode=mode,
        ),
    )
    try:
        yield session
    finally:
        finish = getattr(run, "finish", None)
        if callable(finish):
            finish()


@contextmanager
def _torch_tracking_sessions(*, cfg: TorchTrainConfig) -> Any:
    with ExitStack() as stack:
        sessions: list[TorchTrackingSession] = []
        writer, run_dir = stack.enter_context(_torch_tensorboard_tracking_session(cfg=cfg))
        if writer is not None:
            sessions.append(
                TorchTrackingSession(
                    backend="tensorboard",
                    handle=writer,
                    run_dir=str(run_dir),
                    run_name=(str(cfg.tensorboard_run_name).strip() or Path(str(run_dir)).name),
                    metadata=compact_log_payload(log_dir=str(run_dir)),
                )
            )
        mlflow_session = stack.enter_context(_torch_mlflow_tracking_session(cfg=cfg))
        if mlflow_session is not None:
            sessions.append(mlflow_session)
        wandb_session = stack.enter_context(_torch_wandb_tracking_session(cfg=cfg))
        if wandb_session is not None:
            sessions.append(wandb_session)
        yield tuple(sessions)


def _torch_tracking_run_name(
    session: TorchTrackingSession,
    *,
    cfg: TorchTrainConfig,
) -> str:
    if session.run_name:
        return str(session.run_name)
    if session.backend == "tensorboard" and session.run_dir:
        return Path(str(session.run_dir)).name
    return str(cfg.mlflow_run_name or cfg.wandb_run_name or cfg.tensorboard_run_name).strip() or str(
        session.backend
    )


def _torch_tracking_session_add_text(
    session: TorchTrackingSession,
    *,
    tag: str,
    payload: Mapping[str, Any] | str,
    global_step: int | None = None,
) -> None:
    if session.backend == "tensorboard":
        _torch_tracking_add_text(
            session.handle,
            tag=tag,
            payload=payload,
            global_step=global_step,
        )
        return
    if session.backend == "mlflow":
        log_dict = getattr(session.handle, "log_dict", None)
        if not callable(log_dict):
            return
        normalized = (
            {"text": str(payload)}
            if isinstance(payload, str)
            else _normalize_torch_tracking_payload_value(dict(payload))
        )
        if global_step is not None and isinstance(normalized, dict):
            normalized["global_step"] = int(global_step)
        log_dict(normalized, _torch_tracking_artifact_file(tag=tag, suffix="json"))
        return
    if session.backend == "wandb":
        summary_payload = (
            str(payload)
            if isinstance(payload, str)
            else _normalize_torch_tracking_payload_value(dict(payload))
        )
        if global_step is not None:
            summary_payload = compact_log_payload(value=summary_payload, global_step=int(global_step))
        _torch_tracking_summary_assign(getattr(session.handle, "summary", None), key=tag, value=summary_payload)


def _torch_tracking_session_add_scalar(
    session: TorchTrackingSession,
    *,
    tag: str,
    value: Any,
    global_step: int,
) -> None:
    if value is None:
        return
    if session.backend == "tensorboard":
        _torch_tracking_add_scalar(
            session.handle,
            tag=tag,
            value=value,
            global_step=global_step,
        )
        return
    if session.backend == "mlflow":
        log_metric = getattr(session.handle, "log_metric", None)
        if callable(log_metric):
            log_metric(str(tag), float(value), step=int(global_step))
        return
    if session.backend == "wandb":
        log = getattr(session.handle, "log", None)
        if callable(log):
            log({str(tag): float(value)}, step=int(global_step))


def _torch_tracking_session_add_hparams(
    session: TorchTrackingSession,
    *,
    hparam_payload: Mapping[str, Any],
    metric_payload: Mapping[str, Any],
    run_name: str,
) -> None:
    if session.backend == "tensorboard":
        _torch_tracking_add_hparams(
            session.handle,
            hparam_payload=hparam_payload,
            metric_payload=metric_payload,
            run_name=run_name,
        )
        return
    if session.backend == "mlflow":
        log_params = getattr(session.handle, "log_params", None)
        if callable(log_params):
            log_params(dict(hparam_payload))
        log_metrics = getattr(session.handle, "log_metrics", None)
        if callable(log_metrics) and metric_payload:
            log_metrics(dict(metric_payload))
        return
    if session.backend == "wandb":
        config = getattr(session.handle, "config", None)
        if config is not None:
            update = getattr(config, "update", None)
            if callable(update):
                try:
                    update(dict(hparam_payload), allow_val_change=True)
                except TypeError:
                    update(dict(hparam_payload))
            elif isinstance(config, dict):
                config.update(dict(hparam_payload))
        log = getattr(session.handle, "log", None)
        if callable(log) and metric_payload:
            log(dict(metric_payload))
        _torch_tracking_summary_assign(
            getattr(session.handle, "summary", None),
            key="foresight/hparams",
            value=_normalize_torch_tracking_payload_value(dict(hparam_payload)),
        )


def _log_torch_tensorboard_run_metadata(
    writer: Any | None,
    *,
    cfg: TorchTrainConfig,
    run_dir: str,
    device_payload: Mapping[str, Any],
) -> None:
    if writer is None:
        return
    _torch_tracking_add_text(
        writer,
        tag="foresight/config",
        payload=compact_log_payload(
            vars(cfg),
            tensorboard_run_dir=str(run_dir),
        ),
    )
    _torch_tracking_add_text(
        writer,
        tag="foresight/device",
        payload=dict(device_payload),
    )
    _torch_tracking_add_text(
        writer,
        tag="foresight/hparams",
        payload=_build_torch_tracking_hparams_payload(cfg=cfg),
    )


def _log_torch_tensorboard_run_artifacts(
    writer: Any | None,
    *,
    cfg: TorchTrainConfig,
    run_dir: str,
    checkpoint_paths: Mapping[str, str] | None = None,
) -> None:
    _torch_tracking_add_text(
        writer,
        tag="foresight/artifacts",
        payload=_build_torch_tracking_artifact_payload(
            cfg=cfg,
            tracking_session=TorchTrackingSession(
                backend="tensorboard",
                handle=writer,
                run_dir=str(run_dir),
            ),
            checkpoint_paths=checkpoint_paths,
        ),
    )


def _log_torch_tensorboard_run_summary(
    writer: Any | None,
    *,
    cfg: TorchTrainConfig,
    run_dir: str,
    train_completed_payload: Mapping[str, Any],
    epoch_payload: Mapping[str, Any] | None,
    checkpoint_paths: Mapping[str, str] | None = None,
) -> None:
    if writer is None:
        return
    metric_payload = _build_torch_tracking_metric_payload(
        train_completed_payload=train_completed_payload,
        epoch_payload=epoch_payload,
    )
    _torch_tracking_add_hparams(
        writer,
        hparam_payload=_build_torch_tracking_hparams_payload(cfg=cfg),
        metric_payload=metric_payload,
        run_name=(str(cfg.tensorboard_run_name).strip() or Path(str(run_dir)).name),
    )
    _log_torch_tensorboard_run_artifacts(
        writer,
        cfg=cfg,
        run_dir=run_dir,
        checkpoint_paths=checkpoint_paths,
    )


def _log_torch_tracking_run_metadata(
    sessions: tuple[TorchTrackingSession, ...],
    *,
    cfg: TorchTrainConfig,
    device_payload: Mapping[str, Any],
) -> None:
    for session in sessions:
        if session.backend == "tensorboard":
            _log_torch_tensorboard_run_metadata(
                session.handle,
                cfg=cfg,
                run_dir=str(session.run_dir),
                device_payload=device_payload,
            )
            continue
        _torch_tracking_session_add_text(
            session,
            tag="foresight/config",
            payload=compact_log_payload(vars(cfg)),
        )
        _torch_tracking_session_add_text(
            session,
            tag="foresight/device",
            payload=dict(device_payload),
        )
        if session.backend == "wandb":
            _torch_tracking_summary_assign(
                getattr(session.handle, "summary", None),
                key="foresight/device",
                value=_normalize_torch_tracking_payload_value(dict(device_payload)),
            )


def _log_torch_tracking_epoch_metrics(
    sessions: tuple[TorchTrackingSession, ...],
    *,
    epoch_payload: Mapping[str, Any],
) -> None:
    if not sessions:
        return
    scalar_map = {
        "train/loss": epoch_payload.get("train_loss"),
        "validation/loss": epoch_payload.get("val_loss"),
        "monitor/value": epoch_payload.get("monitor"),
        "monitor/best": epoch_payload.get("best"),
        "train/lr": epoch_payload.get("lr"),
        "system/avg_grad_norm": epoch_payload.get("avg_grad_norm"),
        "time/epoch_seconds": epoch_payload.get("epoch_seconds"),
        "time/step_seconds": epoch_payload.get("step_seconds"),
        "throughput/samples_per_second": epoch_payload.get("samples_per_second"),
        "throughput/batches_per_second": epoch_payload.get("batches_per_second"),
        "train/optimizer_steps": epoch_payload.get("optimizer_steps"),
        "cuda/memory_allocated_mb": epoch_payload.get("memory_allocated_mb"),
        "cuda/memory_reserved_mb": epoch_payload.get("memory_reserved_mb"),
        "cuda/peak_memory_allocated_mb": epoch_payload.get("peak_memory_allocated_mb"),
        "cuda/peak_memory_reserved_mb": epoch_payload.get("peak_memory_reserved_mb"),
    }
    epoch = int(epoch_payload.get("epoch", 0))
    for session in sessions:
        if session.backend == "wandb":
            payload = {
                str(tag): float(value)
                for tag, value in scalar_map.items()
                if value is not None
            }
            log = getattr(session.handle, "log", None)
            if callable(log) and payload:
                log(payload, step=epoch)
            continue
        for tag, value in scalar_map.items():
            _torch_tracking_session_add_scalar(
                session,
                tag=tag,
                value=value,
                global_step=epoch,
            )


def _log_torch_tracking_checkpoint_artifacts(
    session: TorchTrackingSession,
    *,
    checkpoint_paths: Mapping[str, str],
) -> None:
    if not checkpoint_paths:
        return
    if session.backend == "mlflow":
        log_artifact = getattr(session.handle, "log_artifact", None)
        if not callable(log_artifact):
            return
        for path in dict(checkpoint_paths).values():
            log_artifact(str(path), artifact_path="checkpoints")
        return
    if session.backend == "wandb":
        run = session.handle
        artifact_cls = getattr(_require_wandb(), "Artifact", None)
        if artifact_cls is None:
            return
        artifact_name = f"{str(session.run_name or session.backend)}-checkpoints"
        artifact = artifact_cls(str(artifact_name), type="model")
        add_file = getattr(artifact, "add_file", None)
        if not callable(add_file):
            return
        for path in dict(checkpoint_paths).values():
            add_file(str(path), name=Path(str(path)).name)
        log_artifact = getattr(run, "log_artifact", None)
        if callable(log_artifact):
            log_artifact(artifact)


def _log_torch_tracking_run_summary(
    sessions: tuple[TorchTrackingSession, ...],
    *,
    cfg: TorchTrainConfig,
    train_completed_payload: Mapping[str, Any],
    epoch_payload: Mapping[str, Any] | None,
    checkpoint_paths: Mapping[str, str] | None = None,
) -> None:
    hparam_payload = _build_torch_tracking_hparams_payload(cfg=cfg)
    metric_payload = _build_torch_tracking_metric_payload(
        train_completed_payload=train_completed_payload,
        epoch_payload=epoch_payload,
    )
    saved_paths = dict(checkpoint_paths or {})
    for session in sessions:
        if session.backend == "tensorboard":
            _log_torch_tensorboard_run_summary(
                session.handle,
                cfg=cfg,
                run_dir=str(session.run_dir),
                train_completed_payload=train_completed_payload,
                epoch_payload=epoch_payload,
                checkpoint_paths=saved_paths,
            )
        else:
            _torch_tracking_session_add_hparams(
                session,
                hparam_payload=hparam_payload,
                metric_payload=metric_payload,
                run_name=_torch_tracking_run_name(session, cfg=cfg),
            )
            _torch_tracking_session_add_text(
                session,
                tag="foresight/artifacts",
                payload=_build_torch_tracking_artifact_payload(
                    cfg=cfg,
                    tracking_session=session,
                    checkpoint_paths=saved_paths,
                ),
            )
            if session.backend == "wandb":
                summary = getattr(session.handle, "summary", None)
                for key, value in metric_payload.items():
                    _torch_tracking_summary_assign(summary, key=str(key), value=value)
            _log_torch_tracking_checkpoint_artifacts(
                session,
                checkpoint_paths=saved_paths,
            )


def _log_torch_tensorboard_epoch_metrics(
    writer: Any | None,
    *,
    epoch_payload: Mapping[str, Any],
) -> None:
    if writer is None:
        return
    epoch = int(epoch_payload.get("epoch", 0))
    scalar_map = {
        "train/loss": epoch_payload.get("train_loss"),
        "validation/loss": epoch_payload.get("val_loss"),
        "monitor/value": epoch_payload.get("monitor"),
        "monitor/best": epoch_payload.get("best"),
        "train/lr": epoch_payload.get("lr"),
        "system/avg_grad_norm": epoch_payload.get("avg_grad_norm"),
        "time/epoch_seconds": epoch_payload.get("epoch_seconds"),
        "time/step_seconds": epoch_payload.get("step_seconds"),
        "throughput/samples_per_second": epoch_payload.get("samples_per_second"),
        "throughput/batches_per_second": epoch_payload.get("batches_per_second"),
        "train/optimizer_steps": epoch_payload.get("optimizer_steps"),
        "cuda/memory_allocated_mb": epoch_payload.get("memory_allocated_mb"),
        "cuda/memory_reserved_mb": epoch_payload.get("memory_reserved_mb"),
        "cuda/peak_memory_allocated_mb": epoch_payload.get("peak_memory_allocated_mb"),
        "cuda/peak_memory_reserved_mb": epoch_payload.get("peak_memory_reserved_mb"),
    }
    for tag, value in scalar_map.items():
        _torch_tracking_add_scalar(
            writer,
            tag=tag,
            value=value,
            global_step=epoch,
        )


def _apply_torch_train_input_transforms(
    torch: Any,
    model_inputs: tuple[Any, ...],
    *,
    cfg: TorchTrainConfig,
) -> tuple[Any, ...]:
    if not model_inputs:
        return model_inputs
    first = _apply_torch_train_input_dropout(torch, model_inputs[0], cfg=cfg)
    first = _apply_torch_train_temporal_dropout(torch, first, cfg=cfg)
    return (first,) + tuple(model_inputs[1:])


def _reset_torch_module_parameters(model: Any) -> None:
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if not callable(reset):
            continue
        has_uninitialized_params = getattr(module, "has_uninitialized_params", None)
        if callable(has_uninitialized_params) and bool(has_uninitialized_params()):
            continue
        reset()


def _predict_torch_batch(
    model: Any,
    model_inputs: tuple[Any, ...],
    target: Any,
    *,
    epoch_idx: int,
    training: bool,
    batch_predict_fn: TorchBatchPredictFn | None,
) -> Any:
    if batch_predict_fn is None:
        return model(*model_inputs)
    return batch_predict_fn(
        model,
        model_inputs,
        target,
        epoch_idx=int(epoch_idx),
        training=bool(training),
    )


def _snapshot_torch_runtime_payload(
    *,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    best_state: dict[str, Any] | None,
    best_monitor: float,
    bad_epochs: int,
    best_epoch: int,
    base_lrs: tuple[float, ...],
    model: Any,
    cfg: TorchTrainConfig,
    ema_model: Any | None,
    ema_active: bool,
    swa_model: Any | None,
    swa_n_averaged: int,
    lookahead_model: Any | None,
    lookahead_step: int,
) -> dict[str, Any]:
    return _snapshot_torch_training_state(
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_state=best_state,
        best_monitor=float(best_monitor),
        bad_epochs=int(bad_epochs),
        best_epoch=int(best_epoch),
        base_lrs=base_lrs,
        ema_state=(None if ema_model is None or not ema_active else ema_model.state_dict()),
        swa_state=(
            None if swa_model is None or int(swa_n_averaged) <= 0 else swa_model.state_dict()
        ),
        swa_n_averaged=int(swa_n_averaged),
        lookahead_state=(None if lookahead_model is None else lookahead_model.state_dict()),
        lookahead_step=int(lookahead_step),
        model_state=_maybe_torch_model_state_for_checkpoint(
            model=model,
            cfg=cfg,
            ema_model=ema_model,
            ema_active=ema_active,
            swa_model=swa_model,
            swa_n_averaged=int(swa_n_averaged),
            lookahead_model=lookahead_model,
            lookahead_step=int(lookahead_step),
        ),
    )


def _train_torch_model_with_loaders(
    model: Any,
    train_loader: Any,
    val_loader: Any | None,
    *,
    cfg: TorchTrainConfig,
    device: str,
    loss_fn_override: Any | None = None,
    batch_predict_fn: TorchBatchPredictFn | None = None,
    optimizer_factory: TorchOptimizerFactory | None = None,
    scheduler_factory: TorchSchedulerFactory | None = None,
) -> Any:
    torch = _require_torch()
    nn = torch.nn

    cfg = _merge_torch_runtime_override(cfg)
    _validate_torch_train_config(cfg)

    torch.manual_seed(int(cfg.seed))

    dev = torch.device(str(device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available")
    amp_enabled, amp_dtype, scaler = _make_torch_amp_state(torch, cfg=cfg, dev=dev)

    model = model.to(dev)
    _reset_torch_module_parameters(model)

    optimizer_factory_resolved = (
        _make_torch_optimizer if optimizer_factory is None else optimizer_factory
    )
    scheduler_factory_resolved = (
        _make_torch_scheduler if scheduler_factory is None else scheduler_factory
    )

    opt = optimizer_factory_resolved(torch, model, cfg=cfg)
    base_lrs = tuple(float(group["lr"]) for group in opt.param_groups)

    loss_fn = _make_torch_loss_fn(
        torch,
        nn,
        cfg=cfg,
        loss_fn_override=loss_fn_override,
    )

    accum_steps = int(cfg.grad_accum_steps)
    sched, sched_name = scheduler_factory_resolved(
        torch,
        opt,
        cfg=cfg,
        steps_per_epoch=max(1, (len(train_loader) + accum_steps - 1) // accum_steps),
    )
    resume_state = _load_torch_training_state(
        torch,
        model,
        cfg=cfg,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
    )
    start_epoch = max(0, int(resume_state.start_epoch))
    base_lrs = resume_state.base_lrs or base_lrs

    best_monitor_default = (
        float("-inf")
        if str(cfg.monitor_mode).lower().strip() == "max"
        else float("inf")
    )
    best_monitor = (
        best_monitor_default
        if resume_state.best_monitor is None
        else float(resume_state.best_monitor)
    )
    best_state: dict[str, Any] | None = (
        None
        if resume_state.best_state is None
        else _clone_torch_state_dict_to_cpu(resume_state.best_state)
    )
    ema_model = _make_torch_ema_model(model, cfg=cfg)
    ema_active = False
    if ema_model is not None:
        if resume_state.ema_state is not None:
            ema_model.load_state_dict(resume_state.ema_state)
            ema_active = True
        elif int(start_epoch) > int(cfg.ema_warmup_epochs):
            ema_model.load_state_dict(model.state_dict())
            ema_active = True
    swa_model = _make_torch_swa_model(model, cfg=cfg)
    swa_n_averaged = int(resume_state.swa_n_averaged)
    if swa_model is not None:
        if resume_state.swa_state is not None:
            swa_model.load_state_dict(resume_state.swa_state)
            swa_n_averaged = max(1, int(resume_state.swa_n_averaged))
        elif int(start_epoch) > int(cfg.swa_start_epoch):
            swa_model.load_state_dict(model.state_dict())
            swa_n_averaged = 1
    lookahead_model = _make_torch_lookahead_model(model, cfg=cfg)
    lookahead_step = int(resume_state.lookahead_step)
    if lookahead_model is not None and resume_state.lookahead_state is not None:
        lookahead_model.load_state_dict(resume_state.lookahead_state)
    best_epoch = int(resume_state.best_epoch)
    bad_epochs = int(resume_state.bad_epochs)
    last_monitor = resume_state.last_monitor
    last_epoch = int(start_epoch) if int(start_epoch) > 0 else -1
    best_extra_payload = (
        None
        if best_state is None
        else _snapshot_torch_runtime_payload(
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            best_state=best_state,
            best_monitor=float(best_monitor),
            bad_epochs=int(bad_epochs),
            best_epoch=int(best_epoch),
            base_lrs=base_lrs,
            model=model,
            cfg=cfg,
            ema_model=ema_model,
            ema_active=ema_active,
            swa_model=swa_model,
            swa_n_averaged=int(swa_n_averaged),
            lookahead_model=lookahead_model,
            lookahead_step=int(lookahead_step),
        )
    )
    last_extra_payload = (
        None
        if last_monitor is None
        else _snapshot_torch_runtime_payload(
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            best_state=best_state,
            best_monitor=float(best_monitor),
            bad_epochs=int(bad_epochs),
            best_epoch=int(best_epoch),
            base_lrs=base_lrs,
            model=model,
            cfg=cfg,
            ema_model=ema_model,
            ema_active=ema_active,
            swa_model=swa_model,
            swa_n_averaged=int(swa_n_averaged),
            lookahead_model=lookahead_model,
            lookahead_step=int(lookahead_step),
        )
    )
    non_blocking = bool(cfg.pin_memory) and dev.type == "cuda"
    sam_active = _torch_sam_active(cfg=cfg)
    device_payload = compact_log_payload(
        device=str(device),
        **_torch_device_log_payload(torch, dev=dev),
    )
    train_samples = _torch_loader_sample_count(train_loader)
    val_samples = _torch_loader_sample_count(val_loader)
    trainable_parameters, total_parameters = _torch_parameter_counts(model)
    training_started_at = time.perf_counter()
    stop_reason = "completed"
    last_epoch_payload: dict[str, Any] | None = None

    with _torch_tracking_sessions(cfg=cfg) as tracking_sessions:
        tensorboard_session = next(
            (session for session in tracking_sessions if session.backend == "tensorboard"),
            None,
        )
        mlflow_session = next(
            (session for session in tracking_sessions if session.backend == "mlflow"),
            None,
        )
        wandb_session = next(
            (session for session in tracking_sessions if session.backend == "wandb"),
            None,
        )
        for session in tracking_sessions:
            emit_cli_event(
                f"TRACKING {str(session.backend)}",
                event="train_tracking_enabled",
                payload=_torch_tracking_session_payload(session),
            )
        _log_torch_tracking_run_metadata(
            tracking_sessions,
            cfg=cfg,
            device_payload=device_payload,
        )

        emit_cli_event(
            "TRAIN start",
            event="train_started",
            payload=compact_log_payload(
                epochs=int(cfg.epochs),
                start_epoch=int(start_epoch) + 1,
                train_batches=int(len(train_loader)),
                val_batches=(0 if val_loader is None else int(len(val_loader))),
                patience=int(cfg.patience),
                monitor=str(cfg.monitor),
                monitor_mode=str(cfg.monitor_mode),
                optimizer=str(cfg.optimizer),
                scheduler=str(cfg.scheduler),
                grad_accum_steps=int(cfg.grad_accum_steps),
                effective_batch_size=int(cfg.batch_size) * int(cfg.grad_accum_steps),
                train_samples=train_samples,
                val_samples=val_samples,
                trainable_parameters=int(trainable_parameters),
                total_parameters=int(total_parameters),
                amp=bool(amp_enabled),
                amp_dtype=(None if not amp_enabled else str(cfg.amp_dtype)),
                resumed=bool(int(start_epoch) > 0),
                pin_memory=bool(cfg.pin_memory),
                non_blocking=bool(non_blocking),
                tracking_backends=[str(session.backend) for session in tracking_sessions],
                tensorboard_log_dir=(
                    None if tensorboard_session is None else str(tensorboard_session.run_dir)
                ),
                mlflow_run_id=(
                    None if mlflow_session is None else (mlflow_session.metadata or {}).get("run_id")
                ),
                wandb_run_path=(
                    None if wandb_session is None else (wandb_session.metadata or {}).get("run_path")
                ),
                **device_payload,
            ),
        )

        for epoch_idx in range(start_epoch, int(cfg.epochs)):
            epoch_started_at = time.perf_counter()
            if str(dev.type) == "cuda" and bool(torch.cuda.is_available()):
                torch.cuda.reset_peak_memory_stats(dev)
            _apply_torch_warmup(opt, cfg=cfg, epoch_idx=int(epoch_idx), base_lrs=base_lrs)
            model.train()
            total = 0.0
            count = 0
            optimizer_steps = 0
            grad_norm_total = 0.0
            grad_norm_count = 0
            opt.zero_grad(set_to_none=True)
            num_batches = len(train_loader)
            for batch_idx, batch in enumerate(train_loader, start=1):
                model_inputs, yb = _move_torch_batch_to_device(
                    batch,
                    dev=dev,
                    non_blocking=non_blocking,
                )
                train_inputs = _apply_torch_train_input_transforms(torch, model_inputs, cfg=cfg)
                with _make_torch_autocast_context(
                    torch,
                    enabled=bool(amp_enabled),
                    dev=dev,
                    dtype=amp_dtype,
                ):
                    pred = _predict_torch_batch(
                        model,
                        train_inputs,
                        yb,
                        epoch_idx=int(epoch_idx),
                        training=True,
                        batch_predict_fn=batch_predict_fn,
                    )
                    loss = loss_fn(pred, yb)
                loss_to_backprop = loss / float(accum_steps)
                if scaler is not None and bool(scaler.is_enabled()):
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()
                should_step = batch_idx % accum_steps == 0 or batch_idx == num_batches
                if should_step:
                    scaler_enabled = scaler is not None and bool(scaler.is_enabled())
                    if sam_active:
                        perturbations = _apply_torch_sam_perturbation(
                            torch,
                            model=model,
                            cfg=cfg,
                        )
                        if perturbations:
                            opt.zero_grad(set_to_none=True)
                            with _make_torch_autocast_context(
                                torch,
                                enabled=bool(amp_enabled),
                                dev=dev,
                                dtype=amp_dtype,
                            ):
                                pred = _predict_torch_batch(
                                    model,
                                    train_inputs,
                                    yb,
                                    epoch_idx=int(epoch_idx),
                                    training=True,
                                    batch_predict_fn=batch_predict_fn,
                                )
                                loss_second = loss_fn(pred, yb)
                            loss_second.backward()
                            _restore_torch_sam_perturbation(
                                torch,
                                perturbations=perturbations,
                            )
                        _apply_torch_gradient_clipping(torch, model, cfg=cfg)
                        opt.step()
                    else:
                        if scaler_enabled:
                            scaler.unscale_(opt)
                        grad_norm_value = _torch_global_gradient_norm_value(torch, model=model)
                        if grad_norm_value is not None:
                            grad_norm_total += float(grad_norm_value)
                            grad_norm_count += 1
                        _apply_torch_gradient_clipping(torch, model, cfg=cfg)
                        if scaler_enabled:
                            scaler.step(opt)
                            scaler.update()
                        else:
                            opt.step()
                    if lookahead_model is not None:
                        lookahead_step = _update_torch_lookahead_model(
                            torch,
                            lookahead_model=lookahead_model,
                            model=model,
                            cfg=cfg,
                            lookahead_step=int(lookahead_step),
                        )
                    if ema_model is not None and _torch_ema_active_for_epoch(cfg=cfg, epoch_idx=int(epoch_idx)):
                        if not ema_active:
                            ema_model.load_state_dict(model.state_dict())
                            ema_active = True
                        else:
                            _update_torch_ema_model(torch, ema_model=ema_model, model=model, cfg=cfg)
                    if swa_model is not None and _torch_swa_active_for_epoch(cfg=cfg, epoch_idx=int(epoch_idx)):
                        swa_n_averaged = _update_torch_swa_model(
                            torch,
                            swa_model=swa_model,
                            model=model,
                            n_averaged=int(swa_n_averaged),
                        )
                    if sched is not None and _torch_scheduler_steps_per_batch(sched_name):
                        sched.step()
                    opt.zero_grad(set_to_none=True)
                    optimizer_steps += 1

                total += float(loss.detach().cpu().item()) * int(yb.shape[0])
                count += int(yb.shape[0])

            train_loss = total / max(1, count)

            val_loss: float | None = None
            eval_model = _select_torch_deploy_model(
                model=model,
                cfg=cfg,
                ema_model=ema_model,
                ema_active=ema_active,
                swa_model=swa_model,
                swa_n_averaged=int(swa_n_averaged),
                lookahead_model=lookahead_model,
                lookahead_step=int(lookahead_step),
            )
            if val_loader is not None:
                eval_model.eval()
                v_total = 0.0
                v_count = 0
                with torch.no_grad():
                    for batch in val_loader:
                        model_inputs, yb = _move_torch_batch_to_device(
                            batch,
                            dev=dev,
                            non_blocking=non_blocking,
                        )
                        with _make_torch_autocast_context(
                            torch,
                            enabled=bool(amp_enabled),
                            dev=dev,
                            dtype=amp_dtype,
                        ):
                            pred = _predict_torch_batch(
                                eval_model,
                                model_inputs,
                                yb,
                                epoch_idx=int(epoch_idx),
                                training=False,
                                batch_predict_fn=batch_predict_fn,
                            )
                            v_loss = loss_fn(pred, yb)
                        v_total += float(v_loss.detach().cpu().item()) * int(yb.shape[0])
                        v_count += int(yb.shape[0])
                val_loss = v_total / max(1, v_count)

            monitor = _select_torch_monitor_value(cfg, train_loss=float(train_loss), val_loss=val_loss)
            last_monitor = float(monitor)
            last_epoch = int(epoch_idx) + 1

            stop_training = False
            best_improved = False
            if _torch_monitor_improved(value=float(monitor), best=float(best_monitor), cfg=cfg):
                best_monitor = float(monitor)
                bad_epochs = 0
                best_epoch = int(epoch_idx) + 1
                best_improved = True
                if bool(cfg.restore_best) or bool(cfg.save_best_checkpoint):
                    best_state = _clone_torch_state_dict_to_cpu(eval_model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= int(cfg.patience) and int(epoch_idx) + 1 >= int(cfg.min_epochs):
                    stop_training = True
            epoch_seconds = time.perf_counter() - epoch_started_at
            current_lr = _torch_primary_lr(opt)
            epoch_payload = compact_log_payload(
                epoch=int(epoch_idx) + 1,
                total_epochs=int(cfg.epochs),
                train_loss=float(train_loss),
                val_loss=val_loss,
                monitor=float(monitor),
                best=float(best_monitor),
                best_epoch=int(best_epoch),
                best_improved=bool(best_improved),
                bad_epochs=int(bad_epochs),
                lr=current_lr,
                avg_grad_norm=(
                    None
                    if int(grad_norm_count) <= 0
                    else float(grad_norm_total) / float(grad_norm_count)
                ),
                optimizer_steps=int(optimizer_steps),
                epoch_seconds=float(epoch_seconds),
                step_seconds=(
                    None if int(optimizer_steps) <= 0 or float(epoch_seconds) <= 0.0
                    else float(epoch_seconds) / float(optimizer_steps)
                ),
                samples_per_second=(
                    None if int(count) <= 0 or float(epoch_seconds) <= 0.0
                    else float(count) / float(epoch_seconds)
                ),
                batches_per_second=(
                    None if int(num_batches) <= 0 or float(epoch_seconds) <= 0.0
                    else float(num_batches) / float(epoch_seconds)
                ),
                **_torch_cuda_memory_payload(torch, dev=dev),
            )
            emit_cli_event(
                f"EPOCH {int(epoch_idx) + 1}/{int(cfg.epochs)}",
                event="train_epoch_completed",
                payload=epoch_payload,
                progress=True,
            )
            last_epoch_payload = dict(epoch_payload)
            _log_torch_tracking_epoch_metrics(
                tracking_sessions,
                epoch_payload=epoch_payload,
            )

            if not stop_training and sched is not None and not _torch_scheduler_steps_per_batch(sched_name):
                if int(epoch_idx) + 1 > int(cfg.warmup_epochs):
                    if sched_name == "plateau":
                        sched.step(float(monitor))
                    else:
                        sched.step()
                    _clamp_torch_optimizer_min_lr(opt, cfg=cfg)
            if best_state is not None:
                best_extra_payload = _snapshot_torch_runtime_payload(
                    optimizer=opt,
                    scheduler=sched,
                    scaler=scaler,
                    best_state=best_state,
                    best_monitor=float(best_monitor),
                    bad_epochs=int(bad_epochs),
                    best_epoch=int(best_epoch),
                    base_lrs=base_lrs,
                    model=model,
                    cfg=cfg,
                    ema_model=ema_model,
                    ema_active=ema_active,
                    swa_model=swa_model,
                    swa_n_averaged=int(swa_n_averaged),
                    lookahead_model=lookahead_model,
                    lookahead_step=int(lookahead_step),
                )
            last_extra_payload = _snapshot_torch_runtime_payload(
                optimizer=opt,
                scheduler=sched,
                scaler=scaler,
                best_state=best_state,
                best_monitor=float(best_monitor),
                bad_epochs=int(bad_epochs),
                best_epoch=int(best_epoch),
                base_lrs=base_lrs,
                model=model,
                cfg=cfg,
                ema_model=ema_model,
                ema_active=ema_active,
                swa_model=swa_model,
                swa_n_averaged=int(swa_n_averaged),
                lookahead_model=lookahead_model,
                lookahead_step=int(lookahead_step),
            )
            if stop_training:
                stop_reason = "early_stop"
                emit_cli_event(
                    "EARLY stop",
                    event="train_early_stopped",
                    payload=compact_log_payload(
                        epoch=int(epoch_idx) + 1,
                        best_epoch=int(best_epoch),
                        best=float(best_monitor),
                        bad_epochs=int(bad_epochs),
                    ),
                )
                break

        last_state = None
        if bool(cfg.save_last_checkpoint):
            deploy_model = _select_torch_deploy_model(
                model=model,
                cfg=cfg,
                ema_model=ema_model,
                ema_active=ema_active,
                swa_model=swa_model,
                swa_n_averaged=int(swa_n_averaged),
                lookahead_model=lookahead_model,
                lookahead_step=int(lookahead_step),
            )
            last_state = _clone_torch_state_dict_to_cpu(deploy_model.state_dict())
        saved_checkpoint_paths = _maybe_save_torch_checkpoints(
            torch,
            cfg=cfg,
            best_state=best_state,
            best_monitor=float(best_monitor),
            best_epoch=int(best_epoch),
            last_state=last_state,
            last_monitor=last_monitor,
            last_epoch=int(last_epoch),
            best_extra_payload=best_extra_payload,
            last_extra_payload=last_extra_payload,
        )

        if bool(cfg.restore_best) and best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        total_seconds = time.perf_counter() - training_started_at
        train_completed_payload = compact_log_payload(
            epochs_ran=int(last_epoch),
            best_epoch=int(best_epoch),
            best=float(best_monitor),
            restored_best=bool(cfg.restore_best) and best_state is not None,
            stop_reason=str(stop_reason),
            total_seconds=float(total_seconds),
            final_lr=_torch_primary_lr(opt),
        )
        emit_cli_event(
            "TRAIN done",
            event="train_completed",
            payload=train_completed_payload,
        )
        _log_torch_tracking_run_summary(
            tracking_sessions,
            cfg=cfg,
            train_completed_payload=train_completed_payload,
            epoch_payload=last_epoch_payload,
            checkpoint_paths=saved_checkpoint_paths,
        )
        for session in tracking_sessions:
            if session.backend != "tensorboard":
                continue
            _torch_tracking_session_add_scalar(
                session,
                tag="train/epochs_ran",
                value=int(last_epoch),
                global_step=max(1, int(last_epoch)),
            )
            _torch_tracking_session_add_scalar(
                session,
                tag="monitor/final_best",
                value=float(best_monitor),
                global_step=max(1, int(last_epoch)),
            )
            _torch_tracking_session_add_scalar(
                session,
                tag="train/final_lr",
                value=train_completed_payload.get("final_lr"),
                global_step=max(1, int(last_epoch)),
            )
    return model


def _train_loop(
    model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cfg: TorchTrainConfig,
    device: str,
    loss_fn_override: Any | None = None,
    batch_predict_fn: TorchBatchPredictFn | None = None,
) -> Any:
    torch = _require_torch()
    _validate_torch_train_config(cfg)

    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(Y, dtype=torch.float32)

    n = int(x_tensor.shape[0])
    val_n = 0
    if float(cfg.val_split) > 0.0 and n >= 5:
        val_n = max(1, int(round(float(cfg.val_split) * n)))
        val_n = min(val_n, n - 1)

    if val_n > 0:
        train_end = n - val_n
        x_train, y_train = x_tensor[:train_end], y_tensor[:train_end]
        x_val, y_val = x_tensor[train_end:], y_tensor[train_end:]
    else:
        x_train, y_train = x_tensor, y_tensor
        x_val, y_val = None, None

    train_loader = _make_torch_dataloader(
        torch,
        torch.utils.data.TensorDataset(x_train, y_train),
        cfg=cfg,
        shuffle=True,
    )
    val_loader = (
        None
        if x_val is None
        else _make_torch_dataloader(
            torch,
            torch.utils.data.TensorDataset(x_val, y_val),
            cfg=cfg,
            shuffle=False,
        )
    )
    return _train_torch_model_with_loaders(
        model,
        train_loader,
        val_loader,
        cfg=cfg,
        device=device,
        loss_fn_override=loss_fn_override,
        batch_predict_fn=batch_predict_fn,
    )


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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_model(lag_count, h)
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
            self.enc = _make_transformer_encoder(
                nn=nn,
                layer=layer,
                num_layers=int(num_layers),
            )
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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
                self.enc = _make_transformer_encoder(
                    nn=nn,
                    layer=layer,
                    num_layers=layers,
                )
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
                self.enc = _make_transformer_encoder(
                    nn=nn,
                    layer=layer,
                    num_layers=layers,
                )
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
                self.enc = _make_transformer_encoder(
                    nn=nn,
                    layer=layer,
                    num_layers=layers,
                )
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


def torch_tinytimemixer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    patch_len: int = 8,
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
    Torch TinyTimeMixer-style direct multi-horizon forecast on patch tokens.

    Patchifies the lag window, mixes information across patch tokens with token
    MLPs, then applies channel mixing before decoding the forecast horizon.
    """
    _validate_torch_train_config_kwargs(locals())
    h = int(horizon)
    lag_count = int(lags)
    patch = int(patch_len)
    d = int(d_model)
    blocks = int(num_blocks)
    token_hidden = int(token_mixing_hidden)
    channel_hidden = int(channel_mixing_hidden)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if patch <= 0:
        raise ValueError(_PATCH_LEN_MIN_MSG)
    if patch > lag_count:
        raise ValueError("patch_len must be <= lags")
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

    torch = _require_torch()
    nn = torch.nn

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

    def _build_model(window_len: int, forecast_horizon: int) -> Any:
        n_patches = int(math.ceil(window_len / patch))
        total_len = int(n_patches * patch)
        pad_left = int(total_len - window_len)

        class _PatchMixerBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm_t = nn.LayerNorm(d)
                self.token_mlp = nn.Sequential(
                    nn.Linear(n_patches, token_hidden),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(token_hidden, n_patches),
                )
                self.norm_c = nn.LayerNorm(d)
                self.channel_mlp = nn.Sequential(
                    nn.Linear(d, channel_hidden),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(channel_hidden, d),
                )
                self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

            def forward(self, xb: Any) -> Any:  # (B, P, d)
                z = self.norm_t(xb).transpose(1, 2)
                z = self.token_mlp(z).transpose(1, 2)
                xb = xb + self.drop(z)
                z = self.channel_mlp(self.norm_c(xb))
                return xb + self.drop(z)

        class _TinyTimeMixerDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_proj = nn.Linear(patch, d)
                self.patch_norm = nn.LayerNorm(d)
                self.pos = nn.Parameter(torch.zeros((1, n_patches, d), dtype=torch.float32))
                self.blocks = nn.ModuleList([_PatchMixerBlock() for _ in range(blocks)])
                self.out_norm = nn.LayerNorm(2 * d)
                self.head = nn.Sequential(
                    nn.Linear(2 * d, d),
                    nn.GELU(),
                    nn.Dropout(p=drop),
                    nn.Linear(d, forecast_horizon),
                )

            def _patchify(self, xb: Any) -> Any:
                x0 = xb.squeeze(-1)
                if pad_left > 0:
                    left_pad = x0[:, :1].expand(-1, pad_left)
                    x0 = torch.cat([left_pad, x0], dim=1)
                return x0.unfold(dimension=1, size=patch, step=patch)

            def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
                patches = self._patchify(xb)
                z = self.patch_norm(self.patch_proj(patches)) + self.pos
                for blk in self.blocks:
                    z = blk(z)
                feat = torch.cat([z[:, -1, :], z.mean(dim=1)], dim=1)
                feat = self.out_norm(feat)
                return self.head(feat)

        return _TinyTimeMixerDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=lag_count,
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
    model = _train_loop(_FreTSDirect(), X, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, -1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)


def torch_fits_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    low_freq_bins: int = 12,
    hidden_size: int = 64,
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
    Torch FITS-style low-frequency interpolation direct forecast on lag windows.

    Keeps a small band of low FFT bins from the lag window, learns an MLP to
    interpolate them onto an extended context+horizon spectrum, then reconstructs
    the forecast tail with an inverse real FFT.
    """
    h = int(horizon)
    lag_count = int(lags)
    low_bins = int(low_freq_bins)
    hidden = int(hidden_size)
    layers = int(num_layers)
    drop = float(dropout)
    if h <= 0:
        raise ValueError(_HORIZON_MIN_MSG)
    if lag_count <= 0:
        raise ValueError(_LAGS_MIN_MSG)
    if low_bins <= 0:
        raise ValueError(_LOW_FREQ_BINS_MIN_MSG)
    if hidden <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    torch = _require_torch()
    nn = torch.nn

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

    def _build_model(window_len: int, forecast_horizon: int) -> Any:
        in_bins = int(window_len // 2 + 1)
        total_len = int(window_len + forecast_horizon)
        out_bins = int(total_len // 2 + 1)
        k_in = min(low_bins, in_bins)
        scaled_out_bins = max(1, int(math.ceil(float(k_in) * float(total_len) / float(window_len))))
        k_out = min(out_bins, scaled_out_bins)

        class _FITSDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                in_dim = 2 * k_in
                out_dim = 2 * k_out
                blocks: list[Any] = []
                cur_dim = in_dim
                for _ in range(layers):
                    blocks.extend(
                        [
                            nn.Linear(cur_dim, hidden),
                            nn.GELU(),
                            nn.Dropout(p=drop),
                        ]
                    )
                    cur_dim = hidden
                self.input_norm = nn.LayerNorm(in_dim)
                self.backbone = nn.Sequential(*blocks)
                self.output_norm = nn.LayerNorm(cur_dim)
                self.output = nn.Linear(cur_dim, out_dim)

            def forward(self, xb: Any) -> Any:  # xb: (B, T, 1)
                x0 = xb.squeeze(-1)
                xf = torch.fft.rfft(x0, n=window_len, dim=1)
                xf_low = xf[:, :k_in]
                feat = torch.cat([xf_low.real, xf_low.imag], dim=1)
                z = self.input_norm(feat)
                z = self.backbone(z)
                z = self.output_norm(z)
                freq_vec = self.output(z)
                real = freq_vec[:, :k_out]
                imag = freq_vec[:, k_out:]
                imag = imag.clone()
                imag[:, 0] = 0.0
                if total_len % 2 == 0 and k_out == out_bins:
                    imag[:, -1] = 0.0
                spec_low = torch.complex(real, imag)
                spec = torch.zeros(
                    (x0.shape[0], out_bins),
                    dtype=xf.dtype,
                    device=x0.device,
                )
                spec[:, :k_out] = spec_low
                seq_ext = torch.fft.irfft(spec, n=total_len, dim=1)
                return seq_ext[:, -forecast_horizon:]

        return _FITSDirect()

    return _fit_encoder_direct_model(
        train,
        horizon,
        lags=lag_count,
        build_model=_build_model,
        normalize=bool(normalize),
        device=str(device),
        cfg=cfg,
    )


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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)
    y_next = Y.reshape(Y.shape[0], 1)

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
    model = _train_loop(model, x_seq, y_next, cfg=cfg, device=str(device))

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
            self.enc = _make_transformer_encoder(
                nn=nn,
                layer=enc_layer,
                num_layers=layers,
            )
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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
            self.enc = _make_transformer_encoder(
                nn=nn,
                layer=layer,
                num_layers=int(num_layers),
            )
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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
            self.enc = _make_transformer_encoder(
                nn=nn,
                layer=layer,
                num_layers=int(num_layers),
            )
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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
            self.enc = _make_transformer_encoder(
                nn=nn,
                layer=layer,
                num_layers=int(num_layers),
            )
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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
            self.enc = _make_transformer_encoder(
                nn=nn,
                layer=layer,
                num_layers=int(num_layers),
            )
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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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

    x_segments = _segmentize_windows(X)

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
    model = _train_loop(model, x_segments, Y, cfg=cfg, device=str(device))

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

    x_patches = _patchify_windows(X)
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
    model = _train_loop(model, x_patches, Y, cfg=cfg, device=str(device))

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

    x_patches = _patchify_windows(X)

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
            self.encoder = _make_transformer_encoder(
                nn=nn,
                layer=enc_layer,
                num_layers=layers,
            )
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
    model = _train_loop(model, x_patches, Y, cfg=cfg, device=str(device))

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

    x_grid = _gridify_windows(X)

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
    model = _train_loop(model, x_grid, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)
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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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

    x_patches = _patchify_windows(X)

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
    model = _train_loop(model, x_patches, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)
    y_next = Y.reshape(Y.shape[0], 1)

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
        return 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) + 0.5 * (z**2)

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
    model = _train_loop(
        model, x_seq, y_next, cfg=cfg, device=str(device), loss_fn_override=_gaussian_nll
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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)
    y_next = Y.reshape(Y.shape[0], 1)

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
        return torch.maximum(q_f * u, (q_f - 1.0) * u)

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
    model = _train_loop(
        model, x_seq, y_next, cfg=cfg, device=str(device), loss_fn_override=_pinball
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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
            self.enc = _make_transformer_encoder(
                nn=nn,
                layer=layer,
                num_layers=int(num_layers),
            )
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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

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
    x_seq = X.reshape(X.shape[0], X.shape[1], 1)

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
    model = _train_loop(model, x_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)
