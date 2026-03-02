from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..features.time import build_time_features


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'Torch global models require PyTorch. Install with: pip install -e ".[torch]"'
        ) from e
    return torch


def _as_float_2d(a: Any, *, n: int) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(n, 1)
    if arr.ndim != 2 or arr.shape[0] != int(n):
        raise ValueError(f"Expected array shape ({n}, d), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Non-finite values in covariates/features")
    return arr


def _normalize_x_cols(x_cols: Any) -> tuple[str, ...]:
    if x_cols is None:
        return ()
    if isinstance(x_cols, str):
        s = x_cols.strip()
        if not s:
            return ()
        return tuple([p.strip() for p in s.split(",") if p.strip()])
    if isinstance(x_cols, list | tuple):
        out = [str(c).strip() for c in x_cols if str(c).strip()]
        return tuple(out)
    s = str(x_cols).strip()
    return (s,) if s else ()


def _find_cutoff_index(ds: np.ndarray, cutoff: Any) -> int | None:
    idx = pd.Index(ds).get_indexer([cutoff])[0]
    if int(idx) < 0:
        return None
    return int(idx)


def _normalize_series(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(y))
    std = float(np.std(y))
    if std < 1e-8:
        std = 1.0
    return (y - mean) / std, mean, std


@dataclass(frozen=True)
class TorchGlobalTrainConfig:
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


def _train_loop_global(
    model: Any,
    X: np.ndarray,
    ids: np.ndarray,
    Y: np.ndarray,
    *,
    cfg: TorchGlobalTrainConfig,
    device: str,
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
    ids_t = torch.tensor(ids, dtype=torch.long, device=dev)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=dev)

    n = int(X_t.shape[0])
    val_n = 0
    if float(cfg.val_split) > 0.0 and n >= 5:
        val_n = max(1, int(round(float(cfg.val_split) * n)))
        val_n = min(val_n, n - 1)

    if val_n > 0:
        train_end = n - val_n
        X_train, ids_train, Y_train = X_t[:train_end], ids_t[:train_end], Y_t[:train_end]
        X_val, ids_val, Y_val = X_t[train_end:], ids_t[train_end:], Y_t[train_end:]
    else:
        X_train, ids_train, Y_train = X_t, ids_t, Y_t
        X_val, ids_val, Y_val = None, None, None

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, ids_train, Y_train),
        batch_size=int(cfg.batch_size),
        shuffle=True,
    )
    val_loader = (
        None
        if X_val is None
        else torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, ids_val, Y_val),
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
        for xb, idb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb, idb)
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
                for xb, idb, yb in val_loader:
                    pred = model(xb, idb)
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


def _make_grn(d_in: int, d_hidden: int, d_out: int | None = None, dropout: float = 0.0) -> Any:
    """
    Build a minimal TFT-style Gated Residual Network (GRN) as an nn.Module.

    Defined as a factory to avoid importing torch at module import time.
    """
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

        def forward(self, x: Any) -> Any:  # noqa: D401
            h = F.elu(self.fc1(x))
            h = self.fc2(h)
            h = self.dropout(h)
            g = torch.sigmoid(self.gate(h))
            out = self.skip(x) + g * h
            return self.norm(out)

    return _GRN()


def _make_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    # Standard sine/cos positional encoding.
    pe = np.zeros((int(seq_len), int(d_model)), dtype=float)
    position = np.arange(int(seq_len), dtype=float).reshape(-1, 1)
    div_term = np.exp(np.arange(0, int(d_model), 2, dtype=float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def _build_panel_dataset(
    df: pd.DataFrame,
    *,
    cutoff: Any,
    horizon: int,
    context_length: int,
    x_cols: tuple[str, ...],
    normalize: bool,
    max_train_size: int | None,
    sample_step: int,
    add_time_features: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    int,
]:
    """
    Returns:
      X_train, ids_train, Y_train,
      X_pred, ids_pred,
      pred_uids, pred_ds_list, pred_mean, pred_std,
      n_total_series
    """
    if df.empty:
        raise ValueError("long_df is empty")
    required = {"unique_id", "ds", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"long_df missing required columns: {sorted(missing)}")

    x_cols = _normalize_x_cols(x_cols)
    for c in x_cols:
        if c not in df.columns:
            raise KeyError(f"x_cols column not found: {c!r}")

    h = int(horizon)
    ctx = int(context_length)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if ctx <= 0:
        raise ValueError("context_length must be >= 1")
    if int(sample_step) <= 0:
        raise ValueError("sample_step must be >= 1")

    groups = list(df.groupby("unique_id", sort=False))
    if not groups:
        raise ValueError("No series found in long_df")

    uid_to_idx = {str(uid): i for i, (uid, _g) in enumerate(groups)}
    n_total_series = int(len(uid_to_idx))

    x_dim = int(len(x_cols))

    X_chunks: list[np.ndarray] = []
    ids_chunks: list[np.ndarray] = []
    Y_chunks: list[np.ndarray] = []

    pred_X: list[np.ndarray] = []
    pred_ids: list[int] = []
    pred_uids: list[str] = []
    pred_ds_list: list[np.ndarray] = []
    pred_mean: list[float] = []
    pred_std: list[float] = []

    for uid, g in groups:
        uid_s = str(uid)
        g = g.sort_values("ds", kind="mergesort")
        ds_arr = g["ds"].to_numpy(copy=False)
        y_arr = g["y"].to_numpy(dtype=float, copy=False)
        if y_arr.size <= 0:
            continue

        cut_idx = _find_cutoff_index(ds_arr, cutoff)
        if cut_idx is None:
            continue
        train_end = int(cut_idx + 1)

        slice_start = 0
        if max_train_size is not None:
            slice_start = max(0, train_end - int(max_train_size))

        y_train = y_arr[slice_start:train_end]
        if y_train.size < (ctx + h):
            continue

        if bool(normalize):
            y_scaled_train, mean, std = _normalize_series(y_train)
        else:
            y_scaled_train = np.asarray(y_train, dtype=float)
            mean = 0.0
            std = 1.0

        if x_dim > 0:
            x_full = _as_float_2d(g.loc[:, list(x_cols)].to_numpy(copy=False), n=int(y_arr.size))
        else:
            x_full = np.empty((int(y_arr.size), 0), dtype=float)

        if add_time_features:
            time_full, _names = build_time_features(ds_arr)
        else:
            time_full = np.empty((int(y_arr.size), 0), dtype=float)

        time_dim = int(time_full.shape[1])
        input_dim = 1 + x_dim + time_dim
        seq_len = ctx + h

        # Training windows within the sliced train segment.
        # Use y_scaled_train aligned to the sliced segment.
        x_train_seg = x_full[slice_start:train_end]
        time_train_seg = time_full[slice_start:train_end]
        n_train = int(y_scaled_train.size)

        n_windows = n_train - ctx - h + 1
        if n_windows <= 0:
            continue

        step = int(sample_step)
        win_indices = list(range(0, n_windows, step))
        n_samples = int(len(win_indices))

        X_series = np.empty((n_samples, seq_len, input_dim), dtype=float)
        Y_series = np.empty((n_samples, h), dtype=float)
        ids_series = np.empty((n_samples,), dtype=int)

        for j, w0 in enumerate(win_indices):
            t = int(w0 + ctx)
            past = slice(t - ctx, t)
            fut = slice(t, t + h)

            # y feature: past y, future zeros
            y_past = y_scaled_train[past]
            y_future = np.zeros((h,), dtype=float)
            y_feat = np.concatenate([y_past, y_future], axis=0).reshape(seq_len, 1)

            x_feat = np.concatenate([x_train_seg[past], x_train_seg[fut]], axis=0)
            time_feat = np.concatenate([time_train_seg[past], time_train_seg[fut]], axis=0)

            X_series[j] = np.concatenate([y_feat, x_feat, time_feat], axis=1)
            Y_series[j] = y_scaled_train[fut]
            ids_series[j] = int(uid_to_idx[uid_s])

        X_chunks.append(X_series)
        ids_chunks.append(ids_series)
        Y_chunks.append(Y_series)

        # Prediction sample for this series.
        if train_end + h > int(y_arr.size):
            continue
        if train_end < ctx:
            continue

        # Scale prediction context using the same scaler as the current cutoff slice.
        y_ctx = y_arr[train_end - ctx : train_end]
        y_ctx_scaled = (y_ctx - mean) / std
        y_future_zeros = np.zeros((h,), dtype=float)
        y_feat_pred = np.concatenate([y_ctx_scaled, y_future_zeros], axis=0).reshape(seq_len, 1)

        x_ctx = x_full[train_end - ctx : train_end]
        x_fut = x_full[train_end : train_end + h]
        x_feat_pred = np.concatenate([x_ctx, x_fut], axis=0)

        time_ctx = time_full[train_end - ctx : train_end]
        time_fut = time_full[train_end : train_end + h]
        time_feat_pred = np.concatenate([time_ctx, time_fut], axis=0)

        X_pred = np.concatenate([y_feat_pred, x_feat_pred, time_feat_pred], axis=1)
        pred_X.append(X_pred.astype(float, copy=False))
        pred_ids.append(int(uid_to_idx[uid_s]))
        pred_uids.append(uid_s)
        pred_ds_list.append(ds_arr[train_end : train_end + h])
        pred_mean.append(mean)
        pred_std.append(std)

    if not X_chunks:
        raise ValueError("No training windows could be constructed for the given cutoff.")
    if not pred_X:
        raise ValueError("No prediction windows could be constructed for the given cutoff.")

    X_train = np.concatenate(X_chunks, axis=0)
    ids_train = np.concatenate(ids_chunks, axis=0)
    Y_train = np.concatenate(Y_chunks, axis=0)

    X_pred_arr = np.stack(pred_X, axis=0)
    ids_pred_arr = np.asarray(pred_ids, dtype=int)
    mean_arr = np.asarray(pred_mean, dtype=float)
    std_arr = np.asarray(pred_std, dtype=float)

    return (
        X_train,
        ids_train,
        Y_train,
        X_pred_arr,
        ids_pred_arr,
        pred_uids,
        pred_ds_list,
        mean_arr,
        std_arr,
        n_total_series,
    )


def _predict_torch_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    model_name: str,
    context_length: int,
    x_cols: Any,
    add_time_features: bool,
    normalize: bool,
    max_train_size: int | None,
    sample_step: int,
    # training params
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
    device: str,
    # model params
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    lstm_layers: int,
    id_emb_dim: int,
    ma_window: int,
) -> pd.DataFrame:
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)

    (
        X_train,
        ids_train,
        Y_train,
        X_pred,
        ids_pred,
        pred_uids,
        pred_ds_list,
        pred_mean,
        pred_std,
        n_total_series,
    ) = _build_panel_dataset(
        df,
        cutoff=cutoff,
        horizon=int(horizon),
        context_length=int(context_length),
        x_cols=x_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(X_train.shape[2])

    class _BaseGlobalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))

        def _concat_id(self, x: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)  # (B, E)
            emb_t = emb.unsqueeze(1).expand(-1, x.shape[1], -1)
            return torch.cat([x, emb_t], dim=-1)

    class _TFTLite(_BaseGlobalModel):
        def __init__(self) -> None:
            super().__init__()
            d = int(d_model)
            self.in_proj = nn.Linear(input_dim + int(id_emb_dim), d)
            self.pre_grn = _make_grn(d, max(8, d), d, dropout=float(dropout))
            self.lstm = nn.LSTM(
                input_size=d,
                hidden_size=d,
                num_layers=int(lstm_layers),
                dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
                batch_first=True,
            )
            self.attn = nn.MultiheadAttention(
                d, int(nhead), dropout=float(dropout), batch_first=True
            )
            self.post_grn = _make_grn(d, max(8, d), d, dropout=float(dropout))
            self.out = nn.Linear(d, 1)

        def forward(self, x: Any, ids: Any) -> Any:  # noqa: D401
            x = self._concat_id(x, ids)
            h0 = self.in_proj(x)
            h0 = self.pre_grn(h0)
            h1, _ = self.lstm(h0)
            enc = h1[:, :ctx, :]
            dec = h1[:, ctx:, :]
            attn, _w = self.attn(dec, enc, enc, need_weights=False)
            h2 = self.post_grn(dec + attn)
            yhat = self.out(h2).squeeze(-1)
            return yhat

    class _InformerLite(_BaseGlobalModel):
        def __init__(self) -> None:
            super().__init__()
            d = int(d_model)
            self.in_proj = nn.Linear(input_dim + int(id_emb_dim), d)
            pe = _make_positional_encoding(seq_len, d)
            self.register_buffer("pe", torch.tensor(pe, dtype=torch.float32), persistent=False)
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=int(nhead),
                dim_feedforward=int(dim_feedforward),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=int(num_layers))
            self.out = nn.Linear(d, 1)

        def forward(self, x: Any, ids: Any) -> Any:
            x = self._concat_id(x, ids)
            h0 = self.in_proj(x) + self.pe.unsqueeze(0)
            h1 = self.enc(h0)
            yhat = self.out(h1[:, -h:, :]).squeeze(-1)
            return yhat

    class _AutoformerLite(_BaseGlobalModel):
        def __init__(self) -> None:
            super().__init__()
            d = int(d_model)
            self.in_proj = nn.Linear(input_dim + int(id_emb_dim), d)
            pe = _make_positional_encoding(seq_len, d)
            self.register_buffer("pe", torch.tensor(pe, dtype=torch.float32), persistent=False)
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=int(nhead),
                dim_feedforward=int(dim_feedforward),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=int(num_layers))
            self.seasonal_out = nn.Linear(d, 1)
            self.trend_proj = nn.Linear(ctx, h)

        def forward(self, x: Any, ids: Any) -> Any:
            # Decompose y channel on context only.
            y_ctx = x[:, :ctx, 0]  # (B, ctx)
            # Moving average trend.
            w = int(max(1, ma_window))
            pad = w // 2
            y_in = y_ctx.unsqueeze(1)  # (B,1,ctx)
            y_pad = torch.nn.functional.pad(y_in, (pad, pad), mode="replicate")
            trend = torch.nn.functional.avg_pool1d(y_pad, kernel_size=w, stride=1).squeeze(1)
            seasonal = y_ctx - trend

            x2 = x.clone()
            x2[:, :ctx, 0] = seasonal

            x2 = self._concat_id(x2, ids)
            h0 = self.in_proj(x2) + self.pe.unsqueeze(0)
            h1 = self.enc(h0)
            seasonal_hat = self.seasonal_out(h1[:, -h:, :]).squeeze(-1)
            trend_hat = self.trend_proj(trend)
            return seasonal_hat + trend_hat

    name = str(model_name).lower().strip()
    if name == "tft":
        model = _TFTLite()
    elif name == "informer":
        model = _InformerLite()
    elif name == "autoformer":
        model = _AutoformerLite()
    else:
        raise ValueError(f"Unknown global torch model: {model_name!r}")

    cfg = TorchGlobalTrainConfig(
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

    model = _train_loop_global(model, X_train, ids_train, Y_train, cfg=cfg, device=str(device))

    dev = torch.device(str(device))
    Xp = torch.tensor(X_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(Xp, idp).detach().cpu().numpy()

    if yhat_scaled.shape != (int(len(pred_uids)), h):
        raise ValueError(
            f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
        )

    yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)

    rows: list[dict[str, Any]] = []
    for i, uid in enumerate(pred_uids):
        ds_f = pred_ds_list[i]
        for j in range(h):
            rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def torch_tft_global_forecaster(
    *,
    context_length: int = 48,
    x_cols: Any = (),
    add_time_features: bool = True,
    normalize: bool = True,
    max_train_size: int | None = None,
    sample_step: int = 1,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    seed: int = 0,
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.1,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    restore_best: bool = True,
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    lstm_layers: int = 1,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
) -> Any:
    """
    Temporal Fusion Transformer (lite) global/panel forecaster.

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_global(
            long_df,
            cutoff,
            int(horizon),
            model_name="tft",
            context_length=int(context_length),
            x_cols=x_cols,
            add_time_features=bool(add_time_features),
            normalize=bool(normalize),
            max_train_size=max_train_size,
            sample_step=int(sample_step),
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=1,
            dim_feedforward=256,
            dropout=float(dropout),
            lstm_layers=int(lstm_layers),
            id_emb_dim=int(id_emb_dim),
            ma_window=7,
        )

    return _f


def torch_informer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    add_time_features: bool = True,
    normalize: bool = True,
    max_train_size: int | None = None,
    sample_step: int = 1,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    seed: int = 0,
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.1,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    restore_best: bool = True,
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
) -> Any:
    """
    Informer (lite) global/panel forecaster.

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_global(
            long_df,
            cutoff,
            int(horizon),
            model_name="informer",
            context_length=int(context_length),
            x_cols=x_cols,
            add_time_features=bool(add_time_features),
            normalize=bool(normalize),
            max_train_size=max_train_size,
            sample_step=int(sample_step),
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            lstm_layers=1,
            id_emb_dim=int(id_emb_dim),
            ma_window=7,
        )

    return _f


def torch_autoformer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    add_time_features: bool = True,
    normalize: bool = True,
    max_train_size: int | None = None,
    sample_step: int = 1,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    seed: int = 0,
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.1,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    restore_best: bool = True,
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    ma_window: int = 7,
) -> Any:
    """
    Autoformer (lite) global/panel forecaster (trend/seasonal decomposition + transformer encoder).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_global(
            long_df,
            cutoff,
            int(horizon),
            model_name="autoformer",
            context_length=int(context_length),
            x_cols=x_cols,
            add_time_features=bool(add_time_features),
            normalize=bool(normalize),
            max_train_size=max_train_size,
            sample_step=int(sample_step),
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            lstm_layers=1,
            id_emb_dim=int(id_emb_dim),
            ma_window=int(ma_window),
        )

    return _f


def _positional_encoding_sincos(seq_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((int(seq_len), int(d_model)), dtype=float)
    position = np.arange(int(seq_len), dtype=float).reshape(-1, 1)
    div_term = np.exp(
        np.arange(0, int(d_model), 2, dtype=float) * (-math.log(10000.0) / float(d_model))
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def _predict_torch_xformer_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    add_time_features: bool,
    normalize: bool,
    max_train_size: int | None,
    sample_step: int,
    # training params
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
    device: str,
    # model params
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    id_emb_dim: int,
    attn: str,
    pos_emb: str,
    norm: str,
    ffn: str,
    local_window: int,
    performer_features: int,
    linformer_k: int,
    nystrom_landmarks: int,
    residual_gating: bool,
    drop_path: float,
) -> pd.DataFrame:
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)

    (
        X_train,
        ids_train,
        Y_train,
        X_pred,
        ids_pred,
        pred_uids,
        pred_ds_list,
        pred_mean,
        pred_std,
        n_total_series,
    ) = _build_panel_dataset(
        df,
        cutoff=cutoff,
        horizon=int(horizon),
        context_length=int(context_length),
        x_cols=x_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(X_train.shape[2])

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

    attn_s = str(attn).lower().strip()
    pos_s = str(pos_emb).lower().strip()
    norm_s = str(norm).lower().strip()
    ffn_s = str(ffn).lower().strip()

    if attn_s not in {"full", "local", "performer", "linformer", "nystrom"}:
        raise ValueError("attn must be one of: full, local, performer, linformer, nystrom")
    if pos_s not in {"learned", "sincos", "rope", "time2vec", "none"}:
        raise ValueError("pos_emb must be one of: learned, sincos, rope, time2vec, none")
    if norm_s not in {"layer", "rms"}:
        raise ValueError("norm must be one of: layer, rms")
    if ffn_s not in {"gelu", "swiglu"}:
        raise ValueError("ffn must be one of: gelu, swiglu")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0,1)")
    drop_path_f = float(drop_path)
    if not (0.0 <= drop_path_f < 1.0):
        raise ValueError("drop_path must be in [0,1)")

    head_dim = d // heads

    def _make_rmsnorm(dim: int) -> Any:
        class _RMSNorm(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.ones((dim,), dtype=torch.float32))

            def forward(self, xb: Any) -> Any:
                denom = torch.sqrt(torch.mean(xb * xb, dim=-1, keepdim=True) + 1e-8)
                return (xb / denom) * self.scale

        return _RMSNorm()

    def _rope_apply(q: Any, k: Any) -> tuple[Any, Any]:
        L = int(q.shape[2])
        D = int(q.shape[3])
        if D % 2 != 0:
            return q, k
        half = D // 2
        pos = torch.arange(L, device=q.device, dtype=torch.float32).reshape(1, 1, L, 1)
        freq = torch.exp(
            torch.arange(half, device=q.device, dtype=torch.float32)
            * (-math.log(10000.0) / float(half))
        ).reshape(1, 1, 1, half)
        ang = pos * freq
        sin = torch.sin(ang)
        cos = torch.cos(ang)

        def _rot(xb: Any) -> Any:
            x1 = xb[..., :half]
            x2 = xb[..., half:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _rot(q), _rot(k)

    class _Time2Vec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w0 = nn.Parameter(torch.randn(1, 1, 1))
            self.b0 = nn.Parameter(torch.zeros(1, 1, 1))
            self.W = nn.Parameter(torch.randn(1, 1, d - 1) * 0.1)
            self.b = nn.Parameter(torch.zeros(1, 1, d - 1))

        def forward(self, t: Any) -> Any:
            v0 = self.w0 * t + self.b0
            v1 = torch.sin(t * self.W + self.b)
            return torch.cat([v0, v1], dim=-1)

    class _SwiGLUFFN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(dim_feedforward)
            self.fc = nn.Linear(d, 2 * hidden)
            self.proj = nn.Linear(hidden, d)
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any) -> Any:
            x2 = self.fc(xb)
            a, b = x2.chunk(2, dim=-1)
            z = F.silu(a) * b
            z = self.drop(z)
            return self.proj(z)

    class _GELUFFN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(dim_feedforward)
            self.fc1 = nn.Linear(d, hidden)
            self.fc2 = nn.Linear(hidden, d)
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any) -> Any:
            z = F.gelu(self.fc1(xb))
            z = self.drop(z)
            return self.fc2(z)

    class _MultiheadSelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(d, 3 * d)
            self.out = nn.Linear(d, d)

            if attn_s == "linformer":
                self.E = nn.Parameter(torch.randn(heads, int(linformer_k), seq_len) * 0.02)
                self.F = nn.Parameter(torch.randn(heads, int(linformer_k), seq_len) * 0.02)
            else:
                self.register_parameter("E", None)
                self.register_parameter("F", None)

            if attn_s == "performer":
                W = torch.randn(heads, head_dim, int(performer_features)) / math.sqrt(
                    float(head_dim)
                )
                self.register_buffer("W", W, persistent=False)
            else:
                self.register_buffer("W", torch.empty(0), persistent=False)

        def _split_heads(self, xb: Any) -> Any:
            B, L, _D = xb.shape
            return xb.view(B, L, heads, head_dim).transpose(1, 2)

        def _merge_heads(self, xb: Any) -> Any:
            B, Hh, L, dh = xb.shape
            return xb.transpose(1, 2).contiguous().view(B, L, Hh * dh)

        def forward(self, xb: Any) -> Any:
            B, L, _ = xb.shape
            qkv = self.qkv(xb)
            q, k, v = qkv.chunk(3, dim=-1)
            q = self._split_heads(q)
            k = self._split_heads(k)
            v = self._split_heads(v)

            if pos_s == "rope":
                q, k = _rope_apply(q, k)

            scale = 1.0 / math.sqrt(float(head_dim))

            if attn_s == "performer":
                W = self.W
                qf = torch.einsum("bhld,hdm->bhlm", q * scale, W)
                kf = torch.einsum("bhld,hdm->bhlm", k, W)
                qf = F.elu(qf) + 1.0
                kf = F.elu(kf) + 1.0

                kv = torch.einsum("bhlm,bhld->bhmd", kf, v)
                z = torch.einsum("bhlm,bhm->bhl", qf, torch.sum(kf, dim=2) + 1e-8)
                out = torch.einsum("bhlm,bhmd->bhld", qf, kv) / (z.unsqueeze(-1) + 1e-8)
                return self.out(self._merge_heads(out))

            if attn_s == "linformer":
                E = self.E
                Fp = self.F
                k_proj = torch.einsum("hml,bhld->bhmd", E, k)
                v_proj = torch.einsum("hml,bhld->bhmd", Fp, v)
                scores = torch.einsum("bhld,bhmd->bhlm", q * scale, k_proj)
                w = torch.softmax(scores, dim=-1)
                out = torch.einsum("bhlm,bhmd->bhld", w, v_proj)
                return self.out(self._merge_heads(out))

            if attn_s == "nystrom":
                m = int(min(int(nystrom_landmarks), L))
                chunk = int(math.ceil(L / m))
                pad = int(m * chunk - L)
                if pad > 0:
                    q_pad = torch.cat([q, q[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
                    k_pad = torch.cat([k, k[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
                else:
                    q_pad, k_pad = q, k
                q_lm = q_pad.reshape(B, heads, m, chunk, head_dim).mean(dim=3)
                k_lm = k_pad.reshape(B, heads, m, chunk, head_dim).mean(dim=3)

                A = torch.softmax(torch.einsum("bhld,bhmd->bhlm", q * scale, k_lm), dim=-1)
                Bmat = torch.softmax(torch.einsum("bhmd,bhnd->bhmn", q_lm * scale, k_lm), dim=-1)
                C = torch.softmax(torch.einsum("bhmd,bhld->bhml", q_lm * scale, k), dim=-1)
                B_inv = torch.linalg.pinv(Bmat)
                CV = torch.einsum("bhml,bhld->bhmd", C, v)
                out = torch.einsum("bhlm,bhmn,bhnd->bhld", A, B_inv, CV)
                return self.out(self._merge_heads(out))

            scores = torch.einsum("bhld,bhmd->bhlm", q * scale, k)
            if attn_s == "local":
                w = int(local_window)
                idx = torch.arange(L, device=xb.device)
                dist = (idx.reshape(-1, 1) - idx.reshape(1, -1)).abs()
                mask = dist > int(w)
                scores = scores.masked_fill(mask.reshape(1, 1, L, L), float("-inf"))
            w = torch.softmax(scores, dim=-1)
            out = torch.einsum("bhlm,bhmd->bhld", w, v)
            return self.out(self._merge_heads(out))

    class _XFormerBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = _MultiheadSelfAttention()
            if norm_s == "rms":
                self.norm1 = _make_rmsnorm(d)
                self.norm2 = _make_rmsnorm(d)
            else:
                self.norm1 = nn.LayerNorm(d)
                self.norm2 = nn.LayerNorm(d)
            self.drop1 = nn.Dropout(p=drop)
            self.drop2 = nn.Dropout(p=drop)
            self.ffn = _SwiGLUFFN() if ffn_s == "swiglu" else _GELUFFN()
            self.gate = None if not bool(residual_gating) else nn.Linear(d, d)

        def _residual(self, xb: Any, update: Any) -> Any:
            if self.gate is None:
                return xb + update
            g = torch.sigmoid(self.gate(xb))
            return xb + g * update

        def forward(self, xb: Any) -> Any:
            z = self.attn(self.norm1(xb))
            z = self.drop1(z)
            if drop_path_f > 0.0 and self.training:
                keep = 1.0 - drop_path_f
                shape = (z.shape[0],) + (1,) * (z.ndim - 1)
                mask = (torch.rand(shape, device=z.device) < keep).to(z.dtype)
                z = z * mask / keep
            xb = self._residual(xb, z)

            z2 = self.ffn(self.norm2(xb))
            z2 = self.drop2(z2)
            if drop_path_f > 0.0 and self.training:
                keep = 1.0 - drop_path_f
                shape = (z2.shape[0],) + (1,) * (z2.ndim - 1)
                mask = (torch.rand(shape, device=z2.device) < keep).to(z2.dtype)
                z2 = z2 * mask / keep
            xb = self._residual(xb, z2)
            return xb

    class _XFormerGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)

            if pos_s == "learned":
                self.pos = nn.Parameter(torch.zeros((1, seq_len, d), dtype=torch.float32))
                self.register_buffer("pos_buf", torch.empty(0), persistent=False)
                self.time2vec = None
            elif pos_s == "sincos":
                pe = _positional_encoding_sincos(seq_len, d)
                self.register_buffer(
                    "pos_buf", torch.tensor(pe, dtype=torch.float32).unsqueeze(0), persistent=False
                )
                self.pos = None
                self.time2vec = None
            elif pos_s == "time2vec":
                self.pos = None
                self.register_buffer("pos_buf", torch.empty(0), persistent=False)
                self.time2vec = _Time2Vec()
            else:
                self.pos = None
                self.register_buffer("pos_buf", torch.empty(0), persistent=False)
                self.time2vec = None

            self.blocks = nn.ModuleList([_XFormerBlock() for _ in range(int(num_layers))])
            self.head = nn.Linear(d, 1)

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2)

            if self.pos is not None:
                z = z + self.pos
            elif self.pos_buf.numel() > 0:
                z = z + self.pos_buf
            elif self.time2vec is not None:
                t = torch.linspace(0.0, 1.0, steps=seq_len, device=z.device).reshape(1, seq_len, 1)
                z = z + self.time2vec(t)

            for blk in self.blocks:
                z = blk(z)

            yhat = self.head(z[:, -h:, :]).squeeze(-1)
            return yhat

    model = _XFormerGlobal()

    cfg = TorchGlobalTrainConfig(
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
    model = _train_loop_global(model, X_train, ids_train, Y_train, cfg=cfg, device=str(device))

    dev = torch.device(str(device))
    Xp = torch.tensor(X_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(Xp, idp).detach().cpu().numpy()

    if yhat_scaled.shape != (int(len(pred_uids)), h):
        raise ValueError(
            f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
        )

    yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)

    rows: list[dict[str, Any]] = []
    for i, uid in enumerate(pred_uids):
        ds_f = pred_ds_list[i]
        for j in range(h):
            rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def torch_xformer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    add_time_features: bool = True,
    normalize: bool = True,
    max_train_size: int | None = None,
    sample_step: int = 1,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    seed: int = 0,
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.1,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    restore_best: bool = True,
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    attn: str = "full",
    pos_emb: str = "learned",
    norm: str = "layer",
    ffn: str = "gelu",
    local_window: int = 16,
    performer_features: int = 64,
    linformer_k: int = 32,
    nystrom_landmarks: int = 16,
    residual_gating: bool = False,
    drop_path: float = 0.0,
) -> Any:
    """
    Configurable global/panel Transformer-family forecaster.

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_xformer_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            add_time_features=bool(add_time_features),
            normalize=bool(normalize),
            max_train_size=max_train_size,
            sample_step=int(sample_step),
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            attn=str(attn),
            pos_emb=str(pos_emb),
            norm=str(norm),
            ffn=str(ffn),
            local_window=int(local_window),
            performer_features=int(performer_features),
            linformer_k=int(linformer_k),
            nystrom_landmarks=int(nystrom_landmarks),
            residual_gating=bool(residual_gating),
            drop_path=float(drop_path),
        )

    return _f


def _predict_torch_rnn_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    add_time_features: bool,
    normalize: bool,
    max_train_size: int | None,
    sample_step: int,
    # training params
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
    device: str,
    # model params
    cell: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    id_emb_dim: int,
) -> pd.DataFrame:
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)

    (
        X_train,
        ids_train,
        Y_train,
        X_pred,
        ids_pred,
        pred_uids,
        pred_ds_list,
        pred_mean,
        pred_std,
        n_total_series,
    ) = _build_panel_dataset(
        df,
        cutoff=cutoff,
        horizon=int(horizon),
        context_length=int(context_length),
        x_cols=x_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    input_dim = int(X_train.shape[2])

    cell_s = str(cell).lower().strip()
    if cell_s not in {"lstm", "gru"}:
        raise ValueError("cell must be one of: lstm, gru")
    hidden = int(hidden_size)
    layers = int(num_layers)
    if hidden <= 0:
        raise ValueError("hidden_size must be >= 1")
    if layers <= 0:
        raise ValueError("num_layers must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0,1)")
    rnn_drop = drop if layers > 1 else 0.0

    class _RNNGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            in_dim = int(input_dim + id_emb_dim)
            if cell_s == "lstm":
                self.rnn = nn.LSTM(
                    input_size=in_dim,
                    hidden_size=hidden,
                    num_layers=layers,
                    dropout=rnn_drop,
                    batch_first=True,
                )
            else:
                self.rnn = nn.GRU(
                    input_size=in_dim,
                    hidden_size=hidden,
                    num_layers=layers,
                    dropout=rnn_drop,
                    batch_first=True,
                )
            self.head = nn.Linear(hidden, 1)

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z, _st = self.rnn(x2)
            yhat = self.head(z[:, -h:, :]).squeeze(-1)
            return yhat

    model = _RNNGlobal()

    cfg = TorchGlobalTrainConfig(
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
    model = _train_loop_global(model, X_train, ids_train, Y_train, cfg=cfg, device=str(device))

    dev = torch.device(str(device))
    Xp = torch.tensor(X_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(Xp, idp).detach().cpu().numpy()

    if yhat_scaled.shape != (int(len(pred_uids)), h):
        raise ValueError(
            f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
        )

    yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)

    rows: list[dict[str, Any]] = []
    for i, uid in enumerate(pred_uids):
        ds_f = pred_ds_list[i]
        for j in range(h):
            rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def torch_rnn_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    add_time_features: bool = True,
    normalize: bool = True,
    max_train_size: int | None = None,
    sample_step: int = 1,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    seed: int = 0,
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.1,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    restore_best: bool = True,
    device: str = "cpu",
    cell: str = "lstm",
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    id_emb_dim: int = 8,
) -> Any:
    """
    Simple global/panel RNN backbone (LSTM/GRU) with a token-wise horizon head.

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_rnn_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            add_time_features=bool(add_time_features),
            normalize=bool(normalize),
            max_train_size=max_train_size,
            sample_step=int(sample_step),
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
            device=str(device),
            cell=str(cell),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
        )

    return _f
