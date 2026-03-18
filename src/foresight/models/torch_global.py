from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from ..features.time import build_time_features
from .torch_nn import (
    _AGC_CLIP_FACTOR_MIN_MSG,
    _AGC_EPS_POSITIVE_MSG,
    _EMA_SWA_CONFLICT_MSG,
    _GC_MODE_OPTIONS_MSG,
    _GRAD_NOISE_STD_MIN_MSG,
    _HORIZON_LOSS_DECAY_POSITIVE_MSG,
    _INPUT_DROPOUT_RANGE_MSG,
    _LOOKAHEAD_ALPHA_RANGE_MSG,
    _LOOKAHEAD_STEPS_MIN_MSG,
    _SAM_REQUIRES_AMP_DISABLED_MSG,
    _SAM_REQUIRES_SINGLE_ACCUM_MSG,
    _SAM_RHO_MIN_MSG,
    _SWA_START_EPOCH_MAX_EPOCHS_MSG,
    _SWA_START_EPOCH_MIN_MSG,
    _TEMPORAL_DROPOUT_RANGE_MSG,
    TorchTrainConfig,
    _apply_torch_gradient_clipping,
    _apply_torch_sam_perturbation,
    _apply_torch_train_input_dropout,
    _apply_torch_train_temporal_dropout,
    _apply_torch_warmup,
    _clamp_torch_optimizer_min_lr,
    _clone_torch_state_dict_to_cpu,
    _load_torch_training_state,
    _make_manual_gru,
    _make_manual_gru_cell,
    _make_manual_lstm,
    _make_manual_lstm_cell,
    _make_torch_amp_state,
    _make_torch_autocast_context,
    _make_torch_dataloader,
    _make_torch_ema_model,
    _make_torch_lookahead_model,
    _make_torch_loss_fn,
    _make_torch_scheduler,
    _make_torch_swa_model,
    _maybe_save_torch_checkpoints,
    _maybe_torch_model_state_for_checkpoint,
    _restore_torch_sam_perturbation,
    _save_torch_checkpoint,
    _select_torch_deploy_model,
    _select_torch_monitor_value,
    _snapshot_torch_training_state,
    _torch_ema_active_for_epoch,
    _torch_monitor_improved,
    _torch_sam_active,
    _torch_scheduler_steps_per_batch,
    _torch_swa_active_for_epoch,
    _update_torch_ema_model,
    _update_torch_lookahead_model,
    _update_torch_swa_model,
    _validate_torch_train_config,
)

_D_MODEL_MIN_MSG = "d_model must be >= 1"
_NHEAD_MIN_MSG = "nhead must be >= 1"
_D_MODEL_DIVISIBLE_BY_NHEAD_MSG = "d_model must be divisible by nhead"
_NUM_LAYERS_MIN_MSG = "num_layers must be >= 1"
_DROPOUT_RANGE_MSG = "dropout must be in [0,1)"
_FFN_DIM_MIN_MSG = "ffn_dim must be >= 1"
_DIM_FEEDFORWARD_MIN_MSG = "dim_feedforward must be >= 1"
_STRIDE_MIN_MSG = "stride must be >= 1"
_HIDDEN_SIZE_MIN_MSG = "hidden_size must be >= 1"
_KERNEL_SIZE_MIN_MSG = "kernel_size must be >= 1"
_NUM_BLOCKS_MIN_MSG = "num_blocks must be >= 1"
_CHANNELS_MIN_MSG = "channels must be >= 1"
_EINSUM_QK_PROJ = "bhld,hdm->bhlm"
_EINSUM_QK_SCORES = "bhld,bhmd->bhlm"
_EINSUM_ATTN_OUT = "bhlm,bhmd->bhld"


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
        return tuple(p.strip() for p in s.split(",") if p.strip())
    if isinstance(x_cols, list | tuple):
        out = [str(c).strip() for c in x_cols if str(c).strip()]
        return tuple(out)
    s = str(x_cols).strip()
    return (s,) if s else ()


def _normalize_static_cols(static_cols: Any) -> tuple[str, ...]:
    return _normalize_x_cols(static_cols)


def _normalize_quantiles(quantiles: Any) -> tuple[float, ...]:
    """
    Parse quantiles like "0.1,0.5,0.9" / (0.1, 0.5, 0.9) into a sorted tuple.

    For simplicity and stable column naming, only supports quantiles that are
    exactly representable as integer percentiles (e.g. 0.1 -> 10, 0.05 -> 5).
    """
    if quantiles is None:
        return ()

    items: list[Any]
    if isinstance(quantiles, list | tuple):
        items = list(quantiles)
    elif isinstance(quantiles, str):
        s = quantiles.strip()
        items = [] if not s else [p.strip() for p in s.split(",") if p.strip()]
    else:
        items = [quantiles]

    pct_set: set[int] = set()
    for it in items:
        q = float(it)
        if not (0.0 < q < 1.0):
            raise ValueError("quantiles must be in (0,1)")
        pct_f = q * 100.0
        pct = int(round(pct_f))
        if abs(pct_f - float(pct)) > 1e-6:
            raise ValueError(
                f"quantiles must align to integer percentiles (e.g. 0.1,0.5,0.9). Got: {q!r}"
            )
        if pct <= 0 or pct >= 100:
            raise ValueError("quantiles must be strictly between 0% and 100%")
        pct_set.add(pct)

    return tuple(sorted({p / 100.0 for p in pct_set}))


def _quantile_col(q: float) -> str:
    pct = int(round(float(q) * 100.0))
    return f"yhat_p{pct}"


def _pick_point_quantile(quantiles: tuple[float, ...]) -> float:
    if not quantiles:
        return 0.5
    if 0.5 in quantiles:
        return 0.5
    return min(quantiles, key=lambda x: abs(x - 0.5))


def _make_pinball_loss(quantiles: tuple[float, ...]) -> Any:
    torch = _require_torch()
    q_cpu = torch.tensor(list(quantiles), dtype=torch.float32)

    def _pinball(pred: Any, yb: Any) -> Any:
        # pred: (B, H, K), yb: (B, H)
        q = q_cpu.to(device=pred.device).reshape(1, 1, -1)
        u = yb.unsqueeze(-1) - pred
        return torch.maximum(q * u, (q - 1.0) * u)

    return _pinball


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


def _as_local_torch_train_config(cfg: TorchGlobalTrainConfig) -> TorchTrainConfig:
    return TorchTrainConfig(
        epochs=int(cfg.epochs),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        batch_size=int(cfg.batch_size),
        seed=int(cfg.seed),
        patience=int(cfg.patience),
        loss=str(cfg.loss),
        val_split=float(cfg.val_split),
        grad_clip_norm=float(cfg.grad_clip_norm),
        optimizer=str(cfg.optimizer),
        momentum=float(cfg.momentum),
        scheduler=str(cfg.scheduler),
        scheduler_step_size=int(cfg.scheduler_step_size),
        scheduler_gamma=float(cfg.scheduler_gamma),
        scheduler_restart_period=int(cfg.scheduler_restart_period),
        scheduler_restart_mult=int(cfg.scheduler_restart_mult),
        scheduler_pct_start=float(cfg.scheduler_pct_start),
        restore_best=bool(cfg.restore_best),
        min_epochs=int(cfg.min_epochs),
        amp=bool(cfg.amp),
        amp_dtype=str(cfg.amp_dtype),
        warmup_epochs=int(cfg.warmup_epochs),
        min_lr=float(cfg.min_lr),
        grad_accum_steps=int(cfg.grad_accum_steps),
        monitor=str(cfg.monitor),
        monitor_mode=str(cfg.monitor_mode),
        min_delta=float(cfg.min_delta),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        persistent_workers=bool(cfg.persistent_workers),
        scheduler_patience=int(cfg.scheduler_patience),
        grad_clip_mode=str(cfg.grad_clip_mode),
        grad_clip_value=float(cfg.grad_clip_value),
        scheduler_plateau_factor=float(cfg.scheduler_plateau_factor),
        scheduler_plateau_threshold=float(cfg.scheduler_plateau_threshold),
        ema_decay=float(cfg.ema_decay),
        ema_warmup_epochs=int(cfg.ema_warmup_epochs),
        swa_start_epoch=int(cfg.swa_start_epoch),
        lookahead_steps=int(cfg.lookahead_steps),
        lookahead_alpha=float(cfg.lookahead_alpha),
        sam_rho=float(cfg.sam_rho),
        sam_adaptive=bool(cfg.sam_adaptive),
        horizon_loss_decay=float(cfg.horizon_loss_decay),
        input_dropout=float(cfg.input_dropout),
        temporal_dropout=float(cfg.temporal_dropout),
        grad_noise_std=float(cfg.grad_noise_std),
        gc_mode=str(cfg.gc_mode),
        agc_clip_factor=float(cfg.agc_clip_factor),
        agc_eps=float(cfg.agc_eps),
        checkpoint_dir=str(cfg.checkpoint_dir),
        save_best_checkpoint=bool(cfg.save_best_checkpoint),
        save_last_checkpoint=bool(cfg.save_last_checkpoint),
        resume_checkpoint_path=str(cfg.resume_checkpoint_path),
        resume_checkpoint_strict=bool(cfg.resume_checkpoint_strict),
    )


def _train_loop_global(
    model: Any,
    X: np.ndarray,
    ids: np.ndarray,
    Y: np.ndarray,
    *,
    cfg: TorchGlobalTrainConfig,
    device: str,
    loss_fn_override: Any | None = None,
) -> Any:
    torch = _require_torch()
    nn = torch.nn

    cfg_local = _as_local_torch_train_config(cfg)
    _validate_torch_train_config(cfg_local)

    torch.manual_seed(int(cfg.seed))

    dev = torch.device(str(device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available")
    amp_enabled, amp_dtype, scaler = _make_torch_amp_state(torch, cfg=cfg_local, dev=dev)

    model = model.to(dev)

    x_tensor = torch.tensor(X, dtype=torch.float32)
    ids_t = torch.tensor(ids, dtype=torch.long)
    y_tensor = torch.tensor(Y, dtype=torch.float32)

    n = int(x_tensor.shape[0])
    val_n = 0
    if float(cfg.val_split) > 0.0 and n >= 5:
        val_n = max(1, int(round(float(cfg.val_split) * n)))
        val_n = min(val_n, n - 1)

    if val_n > 0:
        train_end = n - val_n
        x_train, ids_train, y_train = x_tensor[:train_end], ids_t[:train_end], y_tensor[:train_end]
        x_val, ids_val, y_val = x_tensor[train_end:], ids_t[train_end:], y_tensor[train_end:]
    else:
        x_train, ids_train, y_train = x_tensor, ids_t, y_tensor
        x_val, ids_val, y_val = None, None, None

    train_loader = _make_torch_dataloader(
        torch,
        torch.utils.data.TensorDataset(x_train, ids_train, y_train),
        cfg=cfg_local,
        shuffle=True,
    )
    val_loader = (
        None
        if x_val is None
        else _make_torch_dataloader(
            torch,
            torch.utils.data.TensorDataset(x_val, ids_val, y_val),
            cfg=cfg_local,
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
    base_lrs = tuple(float(group["lr"]) for group in opt.param_groups)

    loss_fn = _make_torch_loss_fn(
        torch,
        nn,
        cfg=cfg_local,
        loss_fn_override=loss_fn_override,
    )

    accum_steps = int(cfg.grad_accum_steps)
    sched, sched_name = _make_torch_scheduler(
        torch,
        opt,
        cfg=cfg_local,
        steps_per_epoch=max(1, (len(train_loader) + accum_steps - 1) // accum_steps),
    )
    resume_state = _load_torch_training_state(
        torch,
        model,
        cfg=cfg_local,
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
    ema_model = _make_torch_ema_model(model, cfg=cfg_local)
    ema_active = False
    if ema_model is not None:
        if resume_state.ema_state is not None:
            ema_model.load_state_dict(resume_state.ema_state)
            ema_active = True
        elif int(start_epoch) > int(cfg_local.ema_warmup_epochs):
            ema_model.load_state_dict(model.state_dict())
            ema_active = True
    swa_model = _make_torch_swa_model(model, cfg=cfg_local)
    swa_n_averaged = int(resume_state.swa_n_averaged)
    if swa_model is not None:
        if resume_state.swa_state is not None:
            swa_model.load_state_dict(resume_state.swa_state)
            swa_n_averaged = max(1, int(resume_state.swa_n_averaged))
        elif int(start_epoch) > int(cfg_local.swa_start_epoch):
            swa_model.load_state_dict(model.state_dict())
            swa_n_averaged = 1
    lookahead_model = _make_torch_lookahead_model(model, cfg=cfg_local)
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
        else _snapshot_torch_training_state(
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            best_state=best_state,
            best_monitor=float(best_monitor),
            bad_epochs=int(bad_epochs),
            best_epoch=int(best_epoch),
            base_lrs=base_lrs,
            ema_state=(
                None if ema_model is None or not ema_active else ema_model.state_dict()
            ),
            swa_state=(
                None
                if swa_model is None or int(swa_n_averaged) <= 0
                else swa_model.state_dict()
            ),
            swa_n_averaged=int(swa_n_averaged),
            lookahead_state=(
                None if lookahead_model is None else lookahead_model.state_dict()
            ),
            lookahead_step=int(lookahead_step),
            model_state=_maybe_torch_model_state_for_checkpoint(
                model=model,
                cfg=cfg_local,
                ema_model=ema_model,
                ema_active=ema_active,
                swa_model=swa_model,
                swa_n_averaged=int(swa_n_averaged),
                lookahead_model=lookahead_model,
                lookahead_step=int(lookahead_step),
            ),
        )
    )
    last_extra_payload = (
        None
        if last_monitor is None
        else _snapshot_torch_training_state(
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            best_state=best_state,
            best_monitor=float(best_monitor),
            bad_epochs=int(bad_epochs),
            best_epoch=int(best_epoch),
            base_lrs=base_lrs,
            ema_state=(
                None if ema_model is None or not ema_active else ema_model.state_dict()
            ),
            swa_state=(
                None
                if swa_model is None or int(swa_n_averaged) <= 0
                else swa_model.state_dict()
            ),
            swa_n_averaged=int(swa_n_averaged),
            lookahead_state=(
                None if lookahead_model is None else lookahead_model.state_dict()
            ),
            lookahead_step=int(lookahead_step),
            model_state=_maybe_torch_model_state_for_checkpoint(
                model=model,
                cfg=cfg_local,
                ema_model=ema_model,
                ema_active=ema_active,
                swa_model=swa_model,
                swa_n_averaged=int(swa_n_averaged),
                lookahead_model=lookahead_model,
                lookahead_step=int(lookahead_step),
            ),
        )
    )
    non_blocking = bool(cfg.pin_memory) and dev.type == "cuda"
    sam_active = _torch_sam_active(cfg=cfg_local)

    for epoch_idx in range(start_epoch, int(cfg.epochs)):
        _apply_torch_warmup(opt, cfg=cfg_local, epoch_idx=int(epoch_idx), base_lrs=base_lrs)
        model.train()
        total = 0.0
        count = 0
        opt.zero_grad(set_to_none=True)
        num_batches = len(train_loader)
        for batch_idx, (xb, idb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(dev, non_blocking=non_blocking)
            idb = idb.to(dev, non_blocking=non_blocking)
            yb = yb.to(dev, non_blocking=non_blocking)
            xb_train = _apply_torch_train_input_dropout(torch, xb, cfg=cfg_local)
            xb_train = _apply_torch_train_temporal_dropout(torch, xb_train, cfg=cfg_local)
            with _make_torch_autocast_context(
                torch,
                enabled=bool(amp_enabled),
                dev=dev,
                dtype=amp_dtype,
            ):
                pred = model(xb_train, idb)
                loss = loss_fn(pred, yb)
            loss_to_backprop = loss / float(accum_steps)
            if scaler is not None and bool(scaler.is_enabled()):
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()
            should_step = batch_idx % accum_steps == 0 or batch_idx == num_batches
            if should_step:
                needs_unscale = (
                    scaler is not None
                    and bool(scaler.is_enabled())
                    and (
                        float(cfg.grad_clip_norm) > 0.0
                        or (
                            str(cfg.grad_clip_mode).lower().strip() == "value"
                            and float(cfg.grad_clip_value) > 0.0
                        )
                    )
                )
                if sam_active:
                    perturbations = _apply_torch_sam_perturbation(
                        torch,
                        model=model,
                        cfg=cfg_local,
                    )
                    if perturbations:
                        opt.zero_grad(set_to_none=True)
                        with _make_torch_autocast_context(
                            torch,
                            enabled=bool(amp_enabled),
                            dev=dev,
                            dtype=amp_dtype,
                        ):
                            pred = model(xb_train, idb)
                            loss_second = loss_fn(pred, yb)
                        loss_second.backward()
                        _restore_torch_sam_perturbation(
                            torch,
                            perturbations=perturbations,
                        )
                    _apply_torch_gradient_clipping(torch, model, cfg=cfg_local)
                    opt.step()
                else:
                    if needs_unscale:
                        scaler.unscale_(opt)
                    _apply_torch_gradient_clipping(torch, model, cfg=cfg_local)
                    if scaler is not None and bool(scaler.is_enabled()):
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                if lookahead_model is not None:
                    lookahead_step = _update_torch_lookahead_model(
                        torch,
                        lookahead_model=lookahead_model,
                        model=model,
                        cfg=cfg_local,
                        lookahead_step=int(lookahead_step),
                    )
                if ema_model is not None and _torch_ema_active_for_epoch(
                    cfg=cfg_local,
                    epoch_idx=int(epoch_idx),
                ):
                    if not ema_active:
                        ema_model.load_state_dict(model.state_dict())
                        ema_active = True
                    else:
                        _update_torch_ema_model(
                            torch,
                            ema_model=ema_model,
                            model=model,
                            cfg=cfg_local,
                        )
                if swa_model is not None and _torch_swa_active_for_epoch(
                    cfg=cfg_local,
                    epoch_idx=int(epoch_idx),
                ):
                    swa_n_averaged = _update_torch_swa_model(
                        torch,
                        swa_model=swa_model,
                        model=model,
                        n_averaged=int(swa_n_averaged),
                    )
                if sched is not None and _torch_scheduler_steps_per_batch(sched_name):
                    sched.step()
                opt.zero_grad(set_to_none=True)

            total += float(loss.detach().cpu().item()) * int(xb.shape[0])
            count += int(xb.shape[0])

        train_loss = total / max(1, count)

        val_loss: float | None = None
        eval_model = _select_torch_deploy_model(
            model=model,
            cfg=cfg_local,
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
                for xb, idb, yb in val_loader:
                    xb = xb.to(dev, non_blocking=non_blocking)
                    idb = idb.to(dev, non_blocking=non_blocking)
                    yb = yb.to(dev, non_blocking=non_blocking)
                    with _make_torch_autocast_context(
                        torch,
                        enabled=bool(amp_enabled),
                        dev=dev,
                        dtype=amp_dtype,
                    ):
                        pred = eval_model(xb, idb)
                        v_loss = loss_fn(pred, yb)
                    v_total += float(v_loss.detach().cpu().item()) * int(xb.shape[0])
                    v_count += int(xb.shape[0])
            val_loss = v_total / max(1, v_count)

        monitor = _select_torch_monitor_value(
            cfg_local,
            train_loss=float(train_loss),
            val_loss=val_loss,
        )
        last_monitor = float(monitor)
        last_epoch = int(epoch_idx) + 1

        stop_training = False
        if _torch_monitor_improved(value=float(monitor), best=float(best_monitor), cfg=cfg_local):
            best_monitor = float(monitor)
            bad_epochs = 0
            best_epoch = int(epoch_idx) + 1
            if bool(cfg.restore_best) or bool(cfg.save_best_checkpoint):
                best_state = _clone_torch_state_dict_to_cpu(eval_model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg.patience) and int(epoch_idx) + 1 >= int(cfg.min_epochs):
                stop_training = True

        if not stop_training and sched is not None and not _torch_scheduler_steps_per_batch(sched_name):
            if int(epoch_idx) + 1 > int(cfg.warmup_epochs):
                if sched_name == "plateau":
                    sched.step(float(monitor))
                else:
                    sched.step()
                _clamp_torch_optimizer_min_lr(opt, cfg=cfg_local)
        if best_state is not None:
            best_extra_payload = _snapshot_torch_training_state(
                optimizer=opt,
                scheduler=sched,
                scaler=scaler,
                best_state=best_state,
                best_monitor=float(best_monitor),
                bad_epochs=int(bad_epochs),
                best_epoch=int(best_epoch),
                base_lrs=base_lrs,
                ema_state=(
                    None if ema_model is None or not ema_active else ema_model.state_dict()
                ),
                swa_state=(
                    None
                    if swa_model is None or int(swa_n_averaged) <= 0
                    else swa_model.state_dict()
                ),
                swa_n_averaged=int(swa_n_averaged),
                lookahead_state=(
                    None if lookahead_model is None else lookahead_model.state_dict()
                ),
                lookahead_step=int(lookahead_step),
                model_state=_maybe_torch_model_state_for_checkpoint(
                    model=model,
                    cfg=cfg_local,
                    ema_model=ema_model,
                    ema_active=ema_active,
                    swa_model=swa_model,
                    swa_n_averaged=int(swa_n_averaged),
                    lookahead_model=lookahead_model,
                    lookahead_step=int(lookahead_step),
                ),
            )
        last_extra_payload = _snapshot_torch_training_state(
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            best_state=best_state,
            best_monitor=float(best_monitor),
            bad_epochs=int(bad_epochs),
            best_epoch=int(best_epoch),
            base_lrs=base_lrs,
            ema_state=(
                None if ema_model is None or not ema_active else ema_model.state_dict()
            ),
            swa_state=(
                None
                if swa_model is None or int(swa_n_averaged) <= 0
                else swa_model.state_dict()
            ),
            swa_n_averaged=int(swa_n_averaged),
            lookahead_state=(
                None if lookahead_model is None else lookahead_model.state_dict()
            ),
            lookahead_step=int(lookahead_step),
            model_state=_maybe_torch_model_state_for_checkpoint(
                model=model,
                cfg=cfg_local,
                ema_model=ema_model,
                ema_active=ema_active,
                swa_model=swa_model,
                swa_n_averaged=int(swa_n_averaged),
                lookahead_model=lookahead_model,
                lookahead_step=int(lookahead_step),
            ),
        )
        if stop_training:
            break

    last_state = None
    if bool(cfg.save_last_checkpoint):
        deploy_model = _select_torch_deploy_model(
            model=model,
            cfg=cfg_local,
            ema_model=ema_model,
            ema_active=ema_active,
            swa_model=swa_model,
            swa_n_averaged=int(swa_n_averaged),
            lookahead_model=lookahead_model,
            lookahead_step=int(lookahead_step),
        )
        last_state = _clone_torch_state_dict_to_cpu(deploy_model.state_dict())
    _maybe_save_torch_checkpoints(
        torch,
        cfg=cfg_local,
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
    static_cols: tuple[str, ...] = (),
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
      x_train, ids_train, y_train,
      x_pred, ids_pred,
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
    static_cols = _normalize_static_cols(static_cols)
    for c in x_cols:
        if c not in df.columns:
            raise KeyError(f"x_cols column not found: {c!r}")
    for c in static_cols:
        if c not in df.columns:
            raise KeyError(f"static_cols column not found: {c!r}")

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
    static_dim = int(len(static_cols))

    x_chunks: list[np.ndarray] = []
    ids_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []

    pred_x: list[np.ndarray] = []
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

        if static_dim > 0:
            static_values: list[float] = []
            for c in static_cols:
                observed = pd.Series(g[c]).dropna()
                if observed.empty:
                    raise ValueError(
                        f"static_cols column {c!r} has no observed value for unique_id={uid_s!r}"
                    )
                unique_values = pd.unique(observed.to_numpy(copy=False))
                if len(unique_values) != 1:
                    raise ValueError(
                        f"static_cols column {c!r} must be constant within unique_id={uid_s!r}"
                    )
                try:
                    value = float(unique_values[0])
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"static_cols column {c!r} for unique_id={uid_s!r} must be numeric"
                    ) from e
                if not math.isfinite(value):
                    raise ValueError(
                        f"static_cols column {c!r} for unique_id={uid_s!r} must be finite"
                    )
                static_values.append(value)
            static_vec = np.asarray(static_values, dtype=float)
        else:
            static_vec = np.empty((0,), dtype=float)

        if add_time_features:
            time_full, _names = build_time_features(ds_arr)
        else:
            time_full = np.empty((int(y_arr.size), 0), dtype=float)

        time_dim = int(time_full.shape[1])
        input_dim = 1 + x_dim + static_dim + time_dim
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

        x_series = np.empty((n_samples, seq_len, input_dim), dtype=float)
        y_series = np.empty((n_samples, h), dtype=float)
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
            if static_dim > 0:
                static_feat = np.broadcast_to(static_vec.reshape(1, static_dim), (seq_len, static_dim))
            else:
                static_feat = np.empty((seq_len, 0), dtype=float)
            time_feat = np.concatenate([time_train_seg[past], time_train_seg[fut]], axis=0)

            x_series[j] = np.concatenate([y_feat, x_feat, static_feat, time_feat], axis=1)
            y_series[j] = y_scaled_train[fut]
            ids_series[j] = int(uid_to_idx[uid_s])

        x_chunks.append(x_series)
        ids_chunks.append(ids_series)
        y_chunks.append(y_series)

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
        if static_dim > 0:
            static_feat_pred = np.broadcast_to(
                static_vec.reshape(1, static_dim),
                (seq_len, static_dim),
            )
        else:
            static_feat_pred = np.empty((seq_len, 0), dtype=float)

        time_ctx = time_full[train_end - ctx : train_end]
        time_fut = time_full[train_end : train_end + h]
        time_feat_pred = np.concatenate([time_ctx, time_fut], axis=0)

        x_pred = np.concatenate([y_feat_pred, x_feat_pred, static_feat_pred, time_feat_pred], axis=1)
        pred_x.append(x_pred.astype(float, copy=False))
        pred_ids.append(int(uid_to_idx[uid_s]))
        pred_uids.append(uid_s)
        pred_ds_list.append(ds_arr[train_end : train_end + h])
        pred_mean.append(mean)
        pred_std.append(std)

    if not x_chunks:
        raise ValueError("No training windows could be constructed for the given cutoff.")
    if not pred_x:
        raise ValueError("No prediction windows could be constructed for the given cutoff.")

    x_train = np.concatenate(x_chunks, axis=0)
    ids_train = np.concatenate(ids_chunks, axis=0)
    y_train = np.concatenate(y_chunks, axis=0)

    x_pred_arr = np.stack(pred_x, axis=0)
    ids_pred_arr = np.asarray(pred_ids, dtype=int)
    mean_arr = np.asarray(pred_mean, dtype=float)
    std_arr = np.asarray(pred_std, dtype=float)

    return (
        x_train,
        ids_train,
        y_train,
        x_pred_arr,
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
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
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
    quantiles: Any,
) -> pd.DataFrame:
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = int(ctx + h)
    input_dim = int(x_train.shape[2])

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
            rnn_drop = float(dropout) if int(lstm_layers) > 1 else 0.0
            self.lstm = _make_manual_lstm(
                input_size=d,
                hidden_size=d,
                num_layers=int(lstm_layers),
                dropout=float(rnn_drop),
                bidirectional=False,
            )
            self.attn = nn.MultiheadAttention(
                d, int(nhead), dropout=float(dropout), batch_first=True
            )
            self.post_grn = _make_grn(d, max(8, d), d, dropout=float(dropout))
            self.out = nn.Linear(d, out_dim)

        def forward(self, x: Any, ids: Any) -> Any:  # noqa: D401
            x = self._concat_id(x, ids)
            h0 = self.in_proj(x)
            h0 = self.pre_grn(h0)
            h1, _ = self.lstm(h0)
            enc = h1[:, :ctx, :]
            dec = h1[:, ctx:, :]
            attn, _w = self.attn(dec, enc, enc, need_weights=False)
            h2 = self.post_grn(dec + attn)
            yhat = self.out(h2)  # (B, h, out_dim)
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    class _InformerLite(_BaseGlobalModel):
        def __init__(self, seq_len_i: int) -> None:
            super().__init__()
            d = int(d_model)
            self.in_proj = nn.Linear(input_dim + int(id_emb_dim), d)
            pe = _make_positional_encoding(int(seq_len_i), d)
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
            self.out = nn.Linear(d, out_dim)

        def forward(self, x: Any, ids: Any) -> Any:
            x = self._concat_id(x, ids)
            h0 = self.in_proj(x) + self.pe.unsqueeze(0)
            h1 = self.enc(h0)
            yhat = self.out(h1[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    class _AutoformerLite(_BaseGlobalModel):
        def __init__(self, seq_len_i: int) -> None:
            super().__init__()
            d = int(d_model)
            self.in_proj = nn.Linear(input_dim + int(id_emb_dim), d)
            pe = _make_positional_encoding(int(seq_len_i), d)
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
            self.seasonal_out = nn.Linear(d, out_dim)
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
            seasonal_hat = self.seasonal_out(h1[:, -h:, :])
            trend_hat = self.trend_proj(trend)
            if out_dim == 1:
                return seasonal_hat.squeeze(-1) + trend_hat
            return seasonal_hat + trend_hat.unsqueeze(-1)

    name = str(model_name).lower().strip()
    if name == "tft":
        model = _TFTLite()
    elif name == "informer":
        model = _InformerLite(seq_len)
    elif name == "autoformer":
        model = _AutoformerLite(seq_len)
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

    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    if qs:
        if yhat_scaled.shape != (int(len(pred_uids)), h, out_dim):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}, {out_dim}), got {yhat_scaled.shape}"
            )

        yhat_all = yhat_scaled * pred_std.reshape(-1, 1, 1) + pred_mean.reshape(-1, 1, 1)
        q_point = _pick_point_quantile(qs)
        point_idx = int(qs.index(q_point))
        yhat_point = yhat_all[:, :, point_idx]

        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                row = {"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat_point[i, j])}
                for k, qv in enumerate(qs):
                    row[_quantile_col(float(qv))] = float(yhat_all[i, j, k])
                rows.append(row)
    else:
        if yhat_scaled.shape != (int(len(pred_uids)), h):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
            )

        yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)
        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def torch_tft_global_forecaster(
    *,
    context_length: int = 48,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    lstm_layers: int = 1,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
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
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=1,
            dim_feedforward=256,
            dropout=float(dropout),
            lstm_layers=int(lstm_layers),
            id_emb_dim=int(id_emb_dim),
            ma_window=7,
            quantiles=quantiles,
        )

    return _f


def torch_informer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
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
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            lstm_layers=1,
            id_emb_dim=int(id_emb_dim),
            ma_window=7,
            quantiles=quantiles,
        )

    return _f


def _predict_torch_timexer_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
    add_time_features: bool,
    normalize: bool,
    max_train_size: int | None,
    sample_step: int,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    d_model: int,
    nhead: int,
    num_layers: int,
    id_emb_dim: int,
    dropout: float,
) -> pd.DataFrame:
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    if not x_cols_tup:
        raise ValueError("torch-timexer-global requires non-empty x_cols")

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    input_dim = int(x_train.shape[2])
    query_dim = int(input_dim - 1)
    d = int(d_model)
    heads = int(nhead)
    layers = int(num_layers)
    drop = float(dropout)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_DIVISIBLE_BY_NHEAD_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

    class _TimeXerGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.past_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.future_proj = nn.Linear(int(query_dim + id_emb_dim), d)
            past_pe = _make_positional_encoding(int(ctx), d)
            future_pe = _make_positional_encoding(int(h), d)
            self.register_buffer(
                "past_pe", torch.tensor(past_pe, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                "future_pe", torch.tensor(future_pe, dtype=torch.float32), persistent=False
            )
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

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_p = emb.unsqueeze(1).expand(-1, ctx, -1)
            emb_f = emb.unsqueeze(1).expand(-1, h, -1)
            past = torch.cat([xb[:, :ctx, :], emb_p], dim=-1)
            future_cov = xb[:, ctx:, 1:]
            future = torch.cat([future_cov, emb_f], dim=-1)
            mem = self.enc(self.past_proj(past) + self.past_pe.unsqueeze(0))
            q = self.future_proj(future) + self.future_pe.unsqueeze(0)
            for blk in self.blocks:
                q = blk(q, mem)
            return self.out(q).squeeze(-1)

    model = _TimeXerGlobal()
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
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=np.asarray(yhat_scaled, dtype=float),
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=(),
    )


def torch_timexer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
) -> Any:
    """
    TimeXer-style exogenous-aware global/panel forecaster (lite).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_timexer_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            id_emb_dim=int(id_emb_dim),
            dropout=float(dropout),
        )

    return _f


def torch_autoformer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    ma_window: int = 7,
    quantiles: Any = (),
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
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            lstm_layers=1,
            id_emb_dim=int(id_emb_dim),
            ma_window=int(ma_window),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_fedformer_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    num_layers: int,
    ffn_dim: int,
    dropout: float,
    modes: int,
    ma_window: int,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    FEDformer-style global/panel model (lite).

    This variant uses:
      - trend/seasonal decomposition (moving average) on the target channel
      - frequency-domain token mixing (FFT) on the seasonal component (FNet-like)
      - a linear trend head (ctx -> h)

    The goal is a "runs-fast, backtesting-friendly" baseline rather than a
    faithful reimplementation of every FEDformer component.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = int(ctx + h)
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(ffn_dim) <= 0:
        raise ValueError(_FFN_DIM_MIN_MSG)
    if int(modes) <= 0:
        raise ValueError("modes must be >= 1")
    w = int(ma_window)
    if w <= 0:
        raise ValueError("ma_window must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    class _FourierMix(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # complex weights for low-frequency modes (DC ..)
            self.w_re = nn.Parameter(torch.randn(int(modes), d, dtype=torch.float32) * 0.02)
            self.w_im = nn.Parameter(torch.randn(int(modes), d, dtype=torch.float32) * 0.02)

        def forward(self, xb: Any) -> Any:
            # xb: (B, L, D) real
            xf = torch.fft.rfft(xb, dim=1)  # (B, Lf, D) complex
            lf = int(xf.shape[1])
            m = min(int(modes), lf)
            w_c = torch.complex(self.w_re[:m, :], self.w_im[:m, :])  # (m, D)

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
                nn.Linear(d, int(ffn_dim)),
                nn.GELU(),
                nn.Dropout(p=float(drop)),
                nn.Linear(int(ffn_dim), d),
            )
            self.drop = nn.Dropout(p=float(drop))

        def forward(self, xb: Any) -> Any:
            xb = xb + self.drop(self.mix(self.norm1(xb)))
            xb = xb + self.drop(self.ffn(self.norm2(xb)))
            return xb

    class _FEDformerGlobal(nn.Module):
        def __init__(self, seq_len_i: int) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            pe = _make_positional_encoding(int(seq_len_i), int(d))
            self.register_buffer("pe", torch.tensor(pe, dtype=torch.float32), persistent=False)
            self.blocks = nn.ModuleList([_FEDformerBlock() for _ in range(int(num_layers))])
            self.seasonal_out = nn.Linear(d, out_dim)
            self.trend_proj = nn.Linear(int(ctx), int(h))

        def forward(self, xb: Any, ids: Any) -> Any:
            # y trend/seasonal decomposition on context (scaled y)
            y_ctx = xb[:, :ctx, 0]  # (B, ctx)
            pad = int(w // 2)
            y_in = y_ctx.unsqueeze(1)
            y_pad = F.pad(y_in, (pad, pad), mode="replicate")
            trend = F.avg_pool1d(y_pad, kernel_size=int(w), stride=1).squeeze(1)  # (B, ctx)
            seasonal = y_ctx - trend

            x2 = xb.clone()
            x2[:, :ctx, 0] = seasonal
            x2[:, ctx:, 0] = 0.0

            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, x2.shape[1], -1)
            z = self.in_proj(torch.cat([x2, emb_t], dim=-1)) + self.pe.unsqueeze(0)

            for blk in self.blocks:
                z = blk(z)

            seasonal_hat = self.seasonal_out(z[:, -h:, :])  # (B,h,out_dim)
            trend_hat = self.trend_proj(trend)  # (B,h)
            if out_dim == 1:
                return seasonal_hat.squeeze(-1) + trend_hat
            return seasonal_hat + trend_hat.unsqueeze(-1)

    model = _FEDformerGlobal(seq_len)

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_fedformer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    num_layers: int = 2,
    ffn_dim: int = 256,
    dropout: float = 0.1,
    modes: int = 16,
    ma_window: int = 7,
    id_emb_dim: int = 8,
    quantiles: Any = (),
) -> Any:
    """
    FEDformer-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_fedformer_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            num_layers=int(num_layers),
            ffn_dim=int(ffn_dim),
            dropout=float(dropout),
            modes=int(modes),
            ma_window=int(ma_window),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_nonstationary_transformer_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    Non-stationary Transformer-style global/panel model (lite).

    Implements a minimal version of the key ideas:
      - per-window RevIN-like normalization on the target channel
      - learned de-stationary factors (tau, delta) that modulate attention logits

    This is intentionally compact and designed for smoke testing / benchmarking.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = int(ctx + h)
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_DIVISIBLE_BY_NHEAD_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    head_dim = d // heads

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
            feats = torch.stack([mu, sigma], dim=-1)  # (B,2)
            out = self.mlp(feats)  # (B,2H)
            tau_raw, delta = out.chunk(2, dim=-1)
            tau = F.softplus(tau_raw) + 1e-3
            return tau, delta

    class _NSAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(d, 3 * d)
            self.out = nn.Linear(d, d)
            self.drop = nn.Dropout(p=float(drop))

        def forward(self, xb: Any, tau: Any, delta: Any) -> Any:
            # xb: (B,L,d), tau/delta: (B,H)
            qkv = self.qkv(xb)  # (B,L,3d)
            q, k, v = qkv.chunk(3, dim=-1)

            def _reshape(z: Any) -> Any:
                return z.reshape(z.shape[0], z.shape[1], heads, head_dim).permute(0, 2, 1, 3)

            qh = _reshape(q)
            kh = _reshape(k)
            vh = _reshape(v)

            scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(float(head_dim))  # (B,H,L,L)
            t = tau.reshape(tau.shape[0], heads, 1, 1)
            dlt = delta.reshape(delta.shape[0], heads, 1, 1)
            scores = scores * t + dlt
            attn = torch.softmax(scores, dim=-1)
            attn = self.drop(attn)
            out = attn @ vh  # (B,H,L,Dh)
            out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], d)
            return self.out(out)

    class _NSTBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.attn = _NSAttention()
            self.norm2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, int(dim_feedforward)),
                nn.GELU(),
                nn.Dropout(p=float(drop)),
                nn.Linear(int(dim_feedforward), d),
            )
            self.drop = nn.Dropout(p=float(drop))

        def forward(self, xb: Any, tau: Any, delta: Any) -> Any:
            xb = xb + self.drop(self.attn(self.norm1(xb), tau, delta))
            xb = xb + self.drop(self.ffn(self.norm2(xb)))
            return xb

    class _NSTGlobal(nn.Module):
        def __init__(self, seq_len_i: int) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            pe = _make_positional_encoding(int(seq_len_i), int(d))
            self.register_buffer("pe", torch.tensor(pe, dtype=torch.float32), persistent=False)
            self.ds = _DSFactors()
            self.blocks = nn.ModuleList([_NSTBlock() for _ in range(int(num_layers))])
            self.head = nn.Linear(d, out_dim)

        def forward(self, xb: Any, ids: Any) -> Any:
            # RevIN on target channel (scaled y)
            y_ctx = xb[:, :ctx, 0]
            mu = torch.mean(y_ctx, dim=1)
            sigma = torch.sqrt(torch.mean((y_ctx - mu.unsqueeze(1)) ** 2, dim=1) + 1e-6)

            x2 = xb.clone()
            x2[:, :ctx, 0] = (y_ctx - mu.unsqueeze(1)) / sigma.unsqueeze(1)
            x2[:, ctx:, 0] = 0.0

            tau, delta = self.ds(mu, sigma)

            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, x2.shape[1], -1)
            z = self.in_proj(torch.cat([x2, emb_t], dim=-1)) + self.pe.unsqueeze(0)

            for blk in self.blocks:
                z = blk(z, tau, delta)

            yhat_norm = self.head(z[:, -h:, :])
            if out_dim == 1:
                yhat_norm = yhat_norm.squeeze(-1)
                return yhat_norm * sigma.unsqueeze(1) + mu.unsqueeze(1)

            return yhat_norm * sigma.unsqueeze(1).unsqueeze(2) + mu.unsqueeze(1).unsqueeze(2)

    model = _NSTGlobal(seq_len)

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_nonstationary_transformer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Non-stationary Transformer-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_nonstationary_transformer_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
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
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
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
    bigbird_random_k: int,
    performer_features: int,
    linformer_k: int,
    nystrom_landmarks: int,
    reformer_bucket_size: int,
    reformer_n_hashes: int,
    probsparse_top_u: int,
    autocorr_top_k: int,
    residual_gating: bool,
    drop_path: float,
    quantiles: Any,
) -> pd.DataFrame:
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_DIVISIBLE_BY_NHEAD_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)

    attn_s = str(attn).lower().strip()
    pos_s = str(pos_emb).lower().strip()
    norm_s = str(norm).lower().strip()
    ffn_s = str(ffn).lower().strip()

    if attn_s not in {
        "full",
        "local",
        "logsparse",
        "longformer",
        "bigbird",
        "performer",
        "linformer",
        "nystrom",
        "probsparse",
        "autocorr",
        "reformer",
    }:
        raise ValueError(
            "attn must be one of: full, local, logsparse, longformer, bigbird, performer, linformer, nystrom, probsparse, autocorr, reformer"
        )
    if pos_s not in {"learned", "sincos", "rope", "time2vec", "none"}:
        raise ValueError("pos_emb must be one of: learned, sincos, rope, time2vec, none")
    if norm_s not in {"layer", "rms"}:
        raise ValueError("norm must be one of: layer, rms")
    if ffn_s not in {"gelu", "swiglu"}:
        raise ValueError("ffn must be one of: gelu, swiglu")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)
    drop_path_f = float(drop_path)
    if not (0.0 <= drop_path_f < 1.0):
        raise ValueError("drop_path must be in [0,1)")

    if int(local_window) <= 0:
        raise ValueError("local_window must be >= 1")
    bigbird_k = int(bigbird_random_k)
    if bigbird_k < 0:
        raise ValueError("bigbird_random_k must be >= 0")
    if int(performer_features) <= 0:
        raise ValueError("performer_features must be >= 1")
    if int(linformer_k) <= 0:
        raise ValueError("linformer_k must be >= 1")
    if int(nystrom_landmarks) <= 0:
        raise ValueError("nystrom_landmarks must be >= 1")

    probs_u = int(probsparse_top_u)
    if probs_u <= 0:
        raise ValueError("probsparse_top_u must be >= 1")
    auto_k = int(autocorr_top_k)
    if auto_k <= 0:
        raise ValueError("autocorr_top_k must be >= 1")

    reformer_bs = int(reformer_bucket_size)
    if reformer_bs <= 0:
        raise ValueError("reformer_bucket_size must be >= 1")
    reformer_hashes = int(reformer_n_hashes)
    if reformer_hashes <= 0:
        raise ValueError("reformer_n_hashes must be >= 1")

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

            if attn_s == "reformer":
                n_buckets = int(max(1, int(math.ceil(float(seq_len) / float(reformer_bs)))))
                R = torch.randn(int(reformer_hashes), heads, head_dim, n_buckets) / math.sqrt(
                    float(head_dim)
                )
                self.register_buffer("R", R, persistent=False)
            else:
                self.register_buffer("R", torch.empty(0), persistent=False)

            # BigBird random connections (precomputed for this fixed sequence length).
            self.register_buffer("bigbird_rand", torch.empty(0, dtype=torch.bool), persistent=False)
            if attn_s == "bigbird":
                L = int(seq_len)
                rng = np.random.default_rng(int(seed) + 1337)

                idx_np = np.arange(L, dtype=int)
                dist = np.abs(idx_np.reshape(-1, 1) - idx_np.reshape(1, -1))
                local_allowed_np = dist <= int(local_window)

                global_mask_np = np.zeros((L,), dtype=bool)
                if int(h) > 0:
                    global_mask_np[L - int(h) :] = True
                last_ctx = L - int(h) - 1
                if last_ctx >= 0:
                    global_mask_np[int(last_ctx)] = True

                base_allowed = (
                    local_allowed_np | global_mask_np.reshape(L, 1) | global_mask_np.reshape(1, L)
                )

                rand_allowed = np.zeros((L, L), dtype=bool)
                if int(bigbird_k) > 0:
                    for i in range(L):
                        cand = np.flatnonzero(~base_allowed[i])
                        if cand.size == 0:
                            continue
                        pick = int(min(int(bigbird_k), int(cand.size)))
                        chosen = rng.choice(cand, size=pick, replace=False)
                        rand_allowed[i, chosen] = True
                    rand_allowed |= rand_allowed.T

                self.bigbird_rand = torch.tensor(rand_allowed, dtype=torch.bool)

        def _split_heads(self, xb: Any) -> Any:
            B, L, _D = xb.shape
            return xb.view(B, L, heads, head_dim).transpose(1, 2)

        def _merge_heads(self, xb: Any) -> Any:
            batch_size, head_count, seq_len, head_width = xb.shape
            return (
                xb.transpose(1, 2).contiguous().view(batch_size, seq_len, head_count * head_width)
            )

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
                qf = torch.einsum(_EINSUM_QK_PROJ, q * scale, W)
                kf = torch.einsum(_EINSUM_QK_PROJ, k, W)
                qf = F.elu(qf) + 1.0
                kf = F.elu(kf) + 1.0

                kv = torch.einsum("bhlm,bhld->bhmd", kf, v)
                z = torch.einsum("bhlm,bhm->bhl", qf, torch.sum(kf, dim=2) + 1e-8)
                out = torch.einsum(_EINSUM_ATTN_OUT, qf, kv) / (z.unsqueeze(-1) + 1e-8)
                return self.out(self._merge_heads(out))

            if attn_s == "linformer":
                E = self.E
                f_proj = self.F
                k_proj = torch.einsum("hml,bhld->bhmd", E, k)
                v_proj = torch.einsum("hml,bhld->bhmd", f_proj, v)
                scores = torch.einsum(_EINSUM_QK_SCORES, q * scale, k_proj)
                w = torch.softmax(scores, dim=-1)
                out = torch.einsum(_EINSUM_ATTN_OUT, w, v_proj)
                return self.out(self._merge_heads(out))

            if attn_s == "reformer":
                # Reformer LSH attention (lite): hash tokens into buckets with random projections,
                # then do full attention within sorted chunks (and previous chunk) to approximate.
                if self.R.numel() == 0:
                    raise RuntimeError(
                        "Reformer attention misconfigured (missing projection buffer)"
                    )

                bs = int(reformer_bs)
                out_acc = torch.zeros_like(q)
                n_hash = int(self.R.shape[0])

                for r in range(n_hash):
                    R = self.R[r]  # (H,dh,n_buckets)
                    proj = torch.einsum(_EINSUM_QK_PROJ, q, R)  # (B,H,L,n_buckets)
                    buckets = proj.argmax(dim=-1)  # (B,H,L)

                    sort_idx = buckets.argsort(dim=-1)  # (B,H,L)
                    gather_idx = sort_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    q_s = q.gather(dim=2, index=gather_idx)
                    k_s = k.gather(dim=2, index=gather_idx)
                    v_s = v.gather(dim=2, index=gather_idx)

                    padded_length = int(int(math.ceil(float(L) / float(bs))) * bs)
                    pad = int(padded_length - int(L))
                    if pad > 0:
                        q_s = torch.cat([q_s, q_s[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
                        k_s = torch.cat([k_s, k_s[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
                        v_s = torch.cat([v_s, v_s[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)

                    n_chunks = int(padded_length // bs)
                    q_c = q_s.reshape(B, heads, n_chunks, bs, head_dim)
                    k_c = k_s.reshape(B, heads, n_chunks, bs, head_dim)
                    v_c = v_s.reshape(B, heads, n_chunks, bs, head_dim)

                    k_prev = torch.cat([k_c[:, :, :1, :, :], k_c[:, :, :-1, :, :]], dim=2)
                    v_prev = torch.cat([v_c[:, :, :1, :, :], v_c[:, :, :-1, :, :]], dim=2)
                    k_cat = torch.cat([k_prev, k_c], dim=3)  # (B,H,nc,2*bs,dh)
                    v_cat = torch.cat([v_prev, v_c], dim=3)

                    scores = torch.einsum("bhnqd,bhnkd->bhnqk", q_c * scale, k_cat)
                    w = torch.softmax(scores, dim=-1)
                    out_c = torch.einsum("bhnqk,bhnkd->bhnqd", w, v_cat)

                    out_s = out_c.reshape(B, heads, padded_length, head_dim)[:, :, :L, :]
                    inv = sort_idx.argsort(dim=-1)
                    out = out_s.gather(dim=2, index=inv.unsqueeze(-1).expand(-1, -1, -1, head_dim))
                    out_acc = out_acc + out

                out_acc = out_acc / float(max(1, n_hash))
                return self.out(self._merge_heads(out_acc))

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

                A = torch.softmax(torch.einsum(_EINSUM_QK_SCORES, q * scale, k_lm), dim=-1)
                bridge_matrix = torch.softmax(
                    torch.einsum("bhmd,bhnd->bhmn", q_lm * scale, k_lm),
                    dim=-1,
                )
                C = torch.softmax(torch.einsum("bhmd,bhld->bhml", q_lm * scale, k), dim=-1)
                bridge_matrix_inv = torch.linalg.pinv(bridge_matrix)
                CV = torch.einsum("bhml,bhld->bhmd", C, v)
                out = torch.einsum("bhlm,bhmn,bhnd->bhld", A, bridge_matrix_inv, CV)
                return self.out(self._merge_heads(out))

            if attn_s == "autocorr":
                # AutoCorrelation-style attention (lite): pick top-k delays and aggregate shifted V.
                top_k = int(min(L, auto_k))
                qf = torch.fft.rfft(q, dim=2)
                kf = torch.fft.rfft(k, dim=2)
                corr_f = (qf * torch.conj(kf)).sum(dim=-1)  # (B,H,Lf)
                corr = torch.fft.irfft(corr_f, n=int(L), dim=2)  # (B,H,L)

                delays = corr.topk(k=top_k, dim=-1).indices  # (B,H,K)
                weights = torch.softmax(corr.gather(dim=-1, index=delays), dim=-1)  # (B,H,K)

                pos = torch.arange(L, device=xb.device).reshape(1, 1, L)
                out = torch.zeros_like(v)
                for i in range(top_k):
                    dly = delays[:, :, i]  # (B,H)
                    idx_t = (pos + dly.unsqueeze(-1)) % int(L)
                    v_shift = v.gather(
                        dim=2, index=idx_t.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1])
                    )
                    out = out + weights[:, :, i].unsqueeze(-1).unsqueeze(-1) * v_shift
                return self.out(self._merge_heads(out))

            if attn_s == "probsparse":
                # Informer ProbSparse-style attention (lite): compute attention for top-u queries,
                # use mean(V) for the rest.
                scores = torch.einsum(_EINSUM_QK_SCORES, q * scale, k)  # (B,H,L,L)
                importance = scores.max(dim=-1).values - scores.mean(dim=-1)  # (B,H,L)
                u = int(min(L, probs_u))
                top_q = importance.topk(k=u, dim=-1).indices  # (B,H,u)

                scores_top = scores.gather(
                    dim=2, index=top_q.unsqueeze(-1).expand(-1, -1, -1, int(L))
                )  # (B,H,u,L)
                w = torch.softmax(scores_top, dim=-1)
                out_top = w @ v  # (B,H,u,dh)

                base = v.mean(dim=2, keepdim=True).expand(-1, -1, int(L), -1).clone()
                out = base.scatter(
                    dim=2,
                    index=top_q.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1]),
                    src=out_top,
                )
                return self.out(self._merge_heads(out))

            scores = torch.einsum(_EINSUM_QK_SCORES, q * scale, k)
            if attn_s == "local":
                w = int(local_window)
                idx = torch.arange(L, device=xb.device)
                dist = (idx.reshape(-1, 1) - idx.reshape(1, -1)).abs()
                mask = dist > int(w)
                scores = scores.masked_fill(mask.reshape(1, 1, L, L), float("-inf"))
            if attn_s == "logsparse":
                w = int(local_window)
                idx = torch.arange(L, device=xb.device)
                dist = (idx.reshape(-1, 1) - idx.reshape(1, -1)).abs()
                pow2 = (dist > 0) & ((dist & (dist - 1)) == 0)
                allowed = (dist <= int(w)) | pow2
                scores = scores.masked_fill((~allowed).reshape(1, 1, L, L), float("-inf"))
            if attn_s == "longformer":
                # Longformer-style sliding window + global tokens (lite).
                #
                # For forecasting we treat the last `h` horizon tokens as global queries, so every
                # prediction position can attend to the full context efficiently.
                w = int(local_window)
                idx = torch.arange(L, device=xb.device)
                dist = (idx.reshape(-1, 1) - idx.reshape(1, -1)).abs()
                local_allowed = dist <= int(w)

                global_mask = torch.zeros((int(L),), dtype=torch.bool, device=xb.device)
                if int(h) > 0:
                    global_mask[int(L) - int(h) :] = True
                last_ctx = int(L) - int(h) - 1
                if last_ctx >= 0:
                    global_mask[int(last_ctx)] = True

                allowed = (
                    local_allowed | global_mask.reshape(int(L), 1) | global_mask.reshape(1, int(L))
                )
                scores = scores.masked_fill((~allowed).reshape(1, 1, int(L), int(L)), float("-inf"))
            if attn_s == "bigbird":
                # BigBird-style random + local + global (lite).
                w = int(local_window)
                idx = torch.arange(L, device=xb.device)
                dist = (idx.reshape(-1, 1) - idx.reshape(1, -1)).abs()
                local_allowed = dist <= int(w)

                global_mask = torch.zeros((int(L),), dtype=torch.bool, device=xb.device)
                if int(h) > 0:
                    global_mask[int(L) - int(h) :] = True
                last_ctx = int(L) - int(h) - 1
                if last_ctx >= 0:
                    global_mask[int(last_ctx)] = True

                allowed = (
                    local_allowed | global_mask.reshape(int(L), 1) | global_mask.reshape(1, int(L))
                )
                if self.bigbird_rand.numel() > 0:
                    allowed = allowed | self.bigbird_rand.to(device=xb.device)
                scores = scores.masked_fill((~allowed).reshape(1, 1, int(L), int(L)), float("-inf"))
            w = torch.softmax(scores, dim=-1)
            out = torch.einsum(_EINSUM_ATTN_OUT, w, v)
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
            self.head = nn.Linear(d, out_dim)

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

            yhat = self.head(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    if qs:
        if yhat_scaled.shape != (int(len(pred_uids)), h, out_dim):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}, {out_dim}), got {yhat_scaled.shape}"
            )

        yhat_all = yhat_scaled * pred_std.reshape(-1, 1, 1) + pred_mean.reshape(-1, 1, 1)
        q_point = _pick_point_quantile(qs)
        point_idx = int(qs.index(q_point))
        yhat_point = yhat_all[:, :, point_idx]

        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                row = {"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat_point[i, j])}
                for k, qv in enumerate(qs):
                    row[_quantile_col(float(qv))] = float(yhat_all[i, j, k])
                rows.append(row)
    else:
        if yhat_scaled.shape != (int(len(pred_uids)), h):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
            )

        yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)
        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def torch_xformer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    bigbird_random_k: int = 8,
    performer_features: int = 64,
    linformer_k: int = 32,
    nystrom_landmarks: int = 16,
    reformer_bucket_size: int = 8,
    reformer_n_hashes: int = 1,
    probsparse_top_u: int = 32,
    autocorr_top_k: int = 4,
    residual_gating: bool = False,
    drop_path: float = 0.0,
    quantiles: Any = (),
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
            static_cols=static_cols,
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
            bigbird_random_k=int(bigbird_random_k),
            performer_features=int(performer_features),
            linformer_k=int(linformer_k),
            nystrom_landmarks=int(nystrom_landmarks),
            reformer_bucket_size=int(reformer_bucket_size),
            reformer_n_hashes=int(reformer_n_hashes),
            probsparse_top_u=int(probsparse_top_u),
            autocorr_top_k=int(autocorr_top_k),
            residual_gating=bool(residual_gating),
            drop_path=float(drop_path),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_rnn_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    cell: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    input_dim = int(x_train.shape[2])

    cell_s = str(cell).lower().strip()
    if cell_s not in {"lstm", "gru"}:
        raise ValueError("cell must be one of: lstm, gru")
    hidden = int(hidden_size)
    layers = int(num_layers)
    if hidden <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)
    rnn_drop = drop if layers > 1 else 0.0

    class _RNNGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            in_dim = int(input_dim + id_emb_dim)
            if cell_s == "lstm":
                self.rnn = _make_manual_lstm(
                    input_size=int(in_dim),
                    hidden_size=int(hidden),
                    num_layers=int(layers),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
            else:
                self.rnn = _make_manual_gru(
                    input_size=int(in_dim),
                    hidden_size=int(hidden),
                    num_layers=int(layers),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
            self.head = nn.Linear(hidden, out_dim)

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z, _st = self.rnn(x2)
            yhat = self.head(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    if qs:
        if yhat_scaled.shape != (int(len(pred_uids)), h, out_dim):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}, {out_dim}), got {yhat_scaled.shape}"
            )

        yhat_all = yhat_scaled * pred_std.reshape(-1, 1, 1) + pred_mean.reshape(-1, 1, 1)
        q_point = _pick_point_quantile(qs)
        point_idx = int(qs.index(q_point))
        yhat_point = yhat_all[:, :, point_idx]

        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                row = {"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat_point[i, j])}
                for k, qv in enumerate(qs):
                    row[_quantile_col(float(qv))] = float(yhat_all[i, j, k])
                rows.append(row)
    else:
        if yhat_scaled.shape != (int(len(pred_uids)), h):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
            )

        yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)
        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def torch_rnn_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    cell: str = "lstm",
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    id_emb_dim: int = 8,
    quantiles: Any = (),
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
            static_cols=static_cols,
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
            device=str(device),
            cell=str(cell),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_retnet_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
    add_time_features: bool,
    normalize: bool,
    max_train_size: int | None,
    sample_step: int,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    d_model: int,
    nhead: int,
    num_layers: int,
    ffn_dim: int,
    id_emb_dim: int,
    dropout: float,
    quantiles: Any,
) -> pd.DataFrame:
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    seq_len = int(context_length) + h
    input_dim = int(x_train.shape[2])

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
        raise ValueError(_D_MODEL_DIVISIBLE_BY_NHEAD_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if hidden <= 0:
        raise ValueError(_FFN_DIM_MIN_MSG)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    head_dim = d // heads

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
            batch, token_count, _dim = xb.shape

            def _reshape(z: Any) -> Any:
                return z.reshape(batch, token_count, heads, head_dim).permute(0, 2, 1, 3)

            q = F.elu(_reshape(self.q_proj(xb))) + 1.0
            k = F.elu(_reshape(self.k_proj(xb))) + 1.0
            v = _reshape(self.v_proj(xb))

            decay = torch.sigmoid(self.decay_logits).view(1, heads, 1, 1)
            state = torch.zeros(
                (batch, heads, head_dim, head_dim), device=xb.device, dtype=xb.dtype
            )
            key_state = torch.zeros((batch, heads, head_dim), device=xb.device, dtype=xb.dtype)
            outs: list[Any] = []
            for idx in range(token_count):
                k_t = k[:, :, idx, :]
                v_t = v[:, :, idx, :]
                state = decay * state + torch.einsum("bhd,bhe->bhde", k_t, v_t)
                key_state = decay.squeeze(-1) * key_state + k_t
                numer = torch.einsum("bhd,bhde->bhe", q[:, :, idx, :], state)
                denom = torch.einsum("bhd,bhd->bh", q[:, :, idx, :], key_state).unsqueeze(-1)
                outs.append(numer / (denom + 1e-6))

            out = torch.stack(outs, dim=2)
            out = out.permute(0, 2, 1, 3).reshape(batch, token_count, d)
            return self.out_proj(out / math.sqrt(float(head_dim)))

        def forward(self, xb: Any) -> Any:
            xb = xb + self.drop(self._retention(self.norm1(xb)))
            xb = xb + self.drop(self.ffn(self.norm2(xb)))
            return xb

    class _RetNetGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            pe = _make_positional_encoding(int(seq_len), d)
            self.register_buffer("pe", torch.tensor(pe, dtype=torch.float32), persistent=False)
            self.blocks = nn.ModuleList([_RetentionBlock() for _ in range(layers)])
            self.out = nn.Linear(d, out_dim)

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            z = self.in_proj(torch.cat([xb, emb_t], dim=-1)) + self.pe.unsqueeze(0)
            for blk in self.blocks:
                z = blk(z)
            yhat = self.out(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _RetNetGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=np.asarray(yhat_scaled, dtype=float),
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_retnet_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    ffn_dim: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    RetNet-style global/panel forecaster (lite).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_retnet_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            ffn_dim=int(ffn_dim),
            id_emb_dim=int(id_emb_dim),
            dropout=float(dropout),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_patchtst_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    id_emb_dim: int,
    patch_len: int,
    stride: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    PatchTST-style global/panel model (lite).

    Notes:
      - Builds patch tokens from the full (context + horizon) input sequence.
      - Uses pooled patch representation -> direct multi-horizon head.
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_DIVISIBLE_BY_NHEAD_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)

    p_len = int(patch_len)
    if p_len <= 0:
        raise ValueError("patch_len must be >= 1")
    step = int(stride)
    if step <= 0:
        raise ValueError(_STRIDE_MIN_MSG)
    if p_len > seq_len:
        raise ValueError("patch_len must be <= (context_length + horizon)")
    n_patches = 1 + (seq_len - p_len) // step
    if n_patches <= 0:
        raise ValueError("Internal error: computed n_patches <= 0")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    patch_in_dim = int((input_dim + int(id_emb_dim)) * p_len)

    class _PatchTSTGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.patch_proj = nn.Linear(patch_in_dim, d)
            self.pos = nn.Parameter(torch.zeros((1, int(n_patches), d), dtype=torch.float32))
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
            self.head = nn.Linear(d, int(h * out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)  # (B, T, C)

            # (B, n_patches, patch_len, C)
            patches = x2.unfold(dimension=1, size=p_len, step=step)
            patches = patches.contiguous().reshape(patches.shape[0], patches.shape[1], -1)

            z = self.patch_proj(patches) + self.pos
            z = self.enc(z)
            pooled = self.norm(z.mean(dim=1))
            out = self.head(pooled).reshape(-1, h, out_dim)
            return out.squeeze(-1) if out_dim == 1 else out

    model = _PatchTSTGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    if qs:
        if yhat_scaled.shape != (int(len(pred_uids)), h, out_dim):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}, {out_dim}), got {yhat_scaled.shape}"
            )

        yhat_all = yhat_scaled * pred_std.reshape(-1, 1, 1) + pred_mean.reshape(-1, 1, 1)
        q_point = _pick_point_quantile(qs)
        point_idx = int(qs.index(q_point))
        yhat_point = yhat_all[:, :, point_idx]

        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                row = {"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat_point[i, j])}
                for k, qv in enumerate(qs):
                    row[_quantile_col(float(qv))] = float(yhat_all[i, j, k])
                rows.append(row)
    else:
        if yhat_scaled.shape != (int(len(pred_uids)), h):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
            )

        yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)
        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def torch_patchtst_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    patch_len: int = 16,
    stride: int = 8,
    quantiles: Any = (),
) -> Any:
    """
    PatchTST-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_patchtst_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            patch_len=int(patch_len),
            stride=int(stride),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_crossformer_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    id_emb_dim: int,
    segment_len: int,
    stride: int,
    num_scales: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    Crossformer-style global/panel model (lite).

    Builds multi-scale segment tokens from the full (context + horizon) input sequence,
    mixes tokens with a Transformer encoder, then uses a pooled representation -> direct
    multi-horizon head.
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = int(ctx + h)
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_DIVISIBLE_BY_NHEAD_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)

    base_seg = int(segment_len)
    base_stride = int(stride)
    n_scales_req = int(num_scales)
    if base_seg <= 0:
        raise ValueError("segment_len must be >= 1")
    if base_stride <= 0:
        raise ValueError(_STRIDE_MIN_MSG)
    if n_scales_req <= 0:
        raise ValueError("num_scales must be >= 1")
    if base_seg > seq_len:
        raise ValueError("segment_len must be <= (context_length + horizon)")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    # Build scale configs (skip scales that don't fit).
    scales: list[tuple[int, int, int]] = []  # (seg_len, step, n_tokens)
    for i in range(int(n_scales_req)):
        seg_i = int(base_seg * (2**i))
        step_i = int(base_stride * (2**i))
        if seg_i > seq_len:
            break
        if step_i <= 0:
            continue
        n_tokens_i = 1 + (seq_len - seg_i) // step_i
        if n_tokens_i <= 0:
            continue
        scales.append((seg_i, step_i, int(n_tokens_i)))

    if not scales:
        raise ValueError(
            "Invalid (segment_len, stride, num_scales) configuration for given context_length+horizon"
        )

    class _CrossFormerGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.scale_proj = nn.ModuleList(
                [
                    nn.Linear(int((input_dim + int(id_emb_dim)) * seg_i), d)
                    for seg_i, _step_i, _nt_i in scales
                ]
            )
            self.scale_pos = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros((1, int(nt_i), d), dtype=torch.float32))
                    for _seg_i, _step_i, nt_i in scales
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
            self.head = nn.Linear(d, int(h * out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)  # (B, T, C)

            tokens: list[Any] = []
            for si, (seg_i, step_i, nt_i) in enumerate(scales):
                patches = x2.unfold(dimension=1, size=int(seg_i), step=int(step_i))
                patches = patches.contiguous().reshape(patches.shape[0], patches.shape[1], -1)
                if patches.shape[1] != int(nt_i):
                    patches = patches[:, : int(nt_i), :]

                z = self.scale_proj[si](patches)
                z = z + self.scale_pos[si] + self.scale_emb.weight[int(si)].reshape(1, 1, d)
                tokens.append(self.drop(z))

            zcat = torch.cat(tokens, dim=1)
            zcat = self.enc(zcat)
            pooled = self.norm(zcat.mean(dim=1))
            out = self.head(pooled).reshape(-1, h, out_dim)
            return out.squeeze(-1) if out_dim == 1 else out

    model = _CrossFormerGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_crossformer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    segment_len: int = 16,
    stride: int = 16,
    num_scales: int = 3,
    quantiles: Any = (),
) -> Any:
    """
    Crossformer-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_crossformer_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            segment_len=int(segment_len),
            stride=int(stride),
            num_scales=int(num_scales),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_pyraformer_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    id_emb_dim: int,
    segment_len: int,
    stride: int,
    num_levels: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    Pyraformer-style global/panel model (lite).

    Builds a pyramid of segment tokens:
      - level 0: segmented tokens from the full (context + horizon) sequence
      - level 1..L: pooled tokens (factor 2) from previous level

    Then mixes all pyramid tokens with a Transformer encoder and predicts the horizon from a pooled
    representation.
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = int(ctx + h)
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_DIVISIBLE_BY_NHEAD_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)

    seg = int(segment_len)
    step = int(stride)
    levels_req = int(num_levels)
    if seg <= 0:
        raise ValueError("segment_len must be >= 1")
    if step <= 0:
        raise ValueError(_STRIDE_MIN_MSG)
    if levels_req <= 0:
        raise ValueError("num_levels must be >= 1")
    if seg > seq_len:
        raise ValueError("segment_len must be <= (context_length + horizon)")

    n0 = 1 + (seq_len - seg) // step
    if n0 <= 0:
        raise ValueError("Invalid segment configuration: produces no tokens")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    level_sizes: list[int] = [int(n0)]
    for _i in range(int(levels_req) - 1):
        nxt = int(level_sizes[-1] // 2)
        if nxt <= 0:
            break
        level_sizes.append(nxt)

    patch_in_dim = int((input_dim + int(id_emb_dim)) * seg)

    class _PyraFormerGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.patch_proj = nn.Linear(int(patch_in_dim), d)
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
            self.head = nn.Linear(d, int(h * out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)  # (B, T, C)

            patches = x2.unfold(dimension=1, size=int(seg), step=int(step))
            patches = patches[:, : int(level_sizes[0]), :, :]
            patches = patches.contiguous().reshape(patches.shape[0], patches.shape[1], -1)

            z0 = self.patch_proj(patches)
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
            out = self.head(pooled).reshape(-1, h, out_dim)
            return out.squeeze(-1) if out_dim == 1 else out

    model = _PyraFormerGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_pyraformer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    segment_len: int = 16,
    stride: int = 16,
    num_levels: int = 3,
    quantiles: Any = (),
) -> Any:
    """
    Pyraformer-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_pyraformer_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            segment_len=int(segment_len),
            stride=int(stride),
            num_levels=int(num_levels),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_tsmixer_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    num_blocks: int,
    token_mixing_hidden: int,
    channel_mixing_hidden: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    TSMixer-style global/panel model (lite).

    Uses token mixing (Linear over time) + channel mixing (MLP over channels),
    then a token-wise head for the last `horizon` steps.
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    if d <= 0:
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

    class _MixerBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm_t = nn.LayerNorm(d)
            self.token_mlp = nn.Sequential(
                nn.Linear(seq_len, int(token_mixing_hidden)),
                nn.ReLU(),
                nn.Dropout(p=drop),
                nn.Linear(int(token_mixing_hidden), seq_len),
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

    class _TSMixerGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.embed = nn.Linear(int(input_dim + id_emb_dim), d)
            self.blocks = nn.ModuleList([_MixerBlock() for _ in range(int(num_blocks))])
            self.head = nn.Linear(d, out_dim)

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.embed(x2)
            for blk in self.blocks:
                z = blk(z)
            yhat = self.head(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _TSMixerGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    if qs:
        if yhat_scaled.shape != (int(len(pred_uids)), h, out_dim):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}, {out_dim}), got {yhat_scaled.shape}"
            )

        yhat_all = yhat_scaled * pred_std.reshape(-1, 1, 1) + pred_mean.reshape(-1, 1, 1)
        q_point = _pick_point_quantile(qs)
        point_idx = int(qs.index(q_point))
        yhat_point = yhat_all[:, :, point_idx]

        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                row = {"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat_point[i, j])}
                for k, qv in enumerate(qs):
                    row[_quantile_col(float(qv))] = float(yhat_all[i, j, k])
                rows.append(row)
    else:
        if yhat_scaled.shape != (int(len(pred_uids)), h):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
            )

        yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)
        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def torch_tsmixer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    num_blocks: int = 4,
    token_mixing_hidden: int = 128,
    channel_mixing_hidden: int = 128,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    TSMixer-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_tsmixer_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            num_blocks=int(num_blocks),
            token_mixing_hidden=int(token_mixing_hidden),
            channel_mixing_hidden=int(channel_mixing_hidden),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_itransformer_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    iTransformer-style global/panel model (lite).

    "Inverted" tokens: variables/features are tokens, time is the embedding dimension.
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_DIVISIBLE_BY_NHEAD_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    class _ITransformerGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            # Tokens = (input_dim + id_emb_dim), each token holds a length=seq_len vector.
            self.in_proj = nn.Linear(seq_len, d)
            self.token_pos = nn.Parameter(
                torch.zeros((1, int(input_dim + id_emb_dim), d), dtype=torch.float32)
            )
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
            self.out = nn.Linear(d, int(h * out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)  # (B, T, C2)
            xinv = x2.transpose(1, 2)  # (B, C2, T)
            z = self.in_proj(xinv) + self.token_pos  # (B, C2, d)
            z = self.enc(z)
            y_token = z[:, 0, :]  # token 0 corresponds to the y channel
            out = self.out(y_token).reshape(-1, h, out_dim)
            return out.squeeze(-1) if out_dim == 1 else out

    model = _ITransformerGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    if qs:
        if yhat_scaled.shape != (int(len(pred_uids)), h, out_dim):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}, {out_dim}), got {yhat_scaled.shape}"
            )

        yhat_all = yhat_scaled * pred_std.reshape(-1, 1, 1) + pred_mean.reshape(-1, 1, 1)
        q_point = _pick_point_quantile(qs)
        point_idx = int(qs.index(q_point))
        yhat_point = yhat_all[:, :, point_idx]

        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                row = {"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat_point[i, j])}
                for k, qv in enumerate(qs):
                    row[_quantile_col(float(qv))] = float(yhat_all[i, j, k])
                rows.append(row)
    else:
        if yhat_scaled.shape != (int(len(pred_uids)), h):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
            )

        yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)
        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def torch_itransformer_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    iTransformer-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_itransformer_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_timesnet_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    num_layers: int,
    top_k: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    TimesNet-style global/panel model (lite).

    Detects dominant periods from rFFT of the context y, then applies a shared
    Conv2D block on period-reshaped views (weighted sum across top periods).
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)

    k = int(top_k)
    if k <= 0:
        raise ValueError("top_k must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    def _detect_periods(y_ctx: Any) -> tuple[list[int], Any]:
        # y_ctx: (B, ctx)
        # Return (periods, weights) where weights is a 1D tensor on y_ctx.device
        # and sum(weights)=1.
        amp = torch.fft.rfft(y_ctx, dim=1).abs().mean(dim=0)  # (ctx//2+1,)
        if int(amp.numel()) <= 1:
            w = torch.ones((1,), dtype=torch.float32, device=y_ctx.device)
            return [1], w

        amp = amp.to(dtype=torch.float32)
        amp = amp.clone()
        amp[0] = 0.0  # ignore DC

        k_eff = min(int(k), int(amp.numel() - 1))
        vals, idx = torch.topk(amp, k=k_eff, largest=True)

        period_to_val: dict[int, float] = {}
        for f_i, v_i in zip(idx.tolist(), vals.tolist(), strict=True):
            f = int(f_i)
            if f <= 0:
                continue
            p = int(round(float(ctx) / float(f)))
            p = max(1, min(int(p), int(seq_len)))
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
            # z: (B, T, d)
            B, T, D = z.shape
            out = torch.zeros_like(z)
            for w, p in zip(weights, periods, strict=True):
                pp = int(p)
                if pp <= 0:
                    continue
                pad_len = (pp - (int(T) % pp)) % pp
                if pad_len:
                    pad = z[:, -1:, :].expand(-1, int(pad_len), -1)
                    z_pad = torch.cat([z, pad], dim=1)
                else:
                    z_pad = z

                L = int(z_pad.shape[1])
                z2 = z_pad.reshape(int(B), L // pp, pp, int(D)).permute(0, 3, 1, 2)
                z2 = self.conv(z2)
                z2 = z2.permute(0, 2, 3, 1).reshape(int(B), L, int(D))
                z2 = z2[:, : int(T), :]
                out = out + w * z2

            return self.norm(z + out)

    class _TimesNetGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.blocks = nn.ModuleList([_TimesBlock() for _ in range(int(num_layers))])
            self.head = nn.Linear(d, out_dim)

        def forward(self, xb: Any, ids: Any) -> Any:
            y_ctx = xb[:, : int(ctx), 0]
            with torch.no_grad():
                periods, weights = _detect_periods(y_ctx)

            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2)
            for blk in self.blocks:
                z = blk(z, periods, weights)

            yhat = self.head(z[:, -int(h) :, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _TimesNetGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    if qs:
        if yhat_scaled.shape != (int(len(pred_uids)), h, out_dim):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}, {out_dim}), got {yhat_scaled.shape}"
            )

        yhat_all = yhat_scaled * pred_std.reshape(-1, 1, 1) + pred_mean.reshape(-1, 1, 1)
        q_point = _pick_point_quantile(qs)
        point_idx = int(qs.index(q_point))
        yhat_point = yhat_all[:, :, point_idx]

        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                row = {"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat_point[i, j])}
                for kk, qv in enumerate(qs):
                    row[_quantile_col(float(qv))] = float(yhat_all[i, j, kk])
                rows.append(row)
    else:
        if yhat_scaled.shape != (int(len(pred_uids)), h):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
            )

        yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)
        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def torch_timesnet_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    num_layers: int = 2,
    top_k: int = 3,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    TimesNet-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_timesnet_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            num_layers=int(num_layers),
            top_k=int(top_k),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _make_pred_df_from_scaled(
    *,
    yhat_scaled: np.ndarray,
    pred_uids: list[str],
    pred_ds_list: list[np.ndarray],
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    horizon: int,
    qs: tuple[float, ...],
) -> pd.DataFrame:
    h = int(horizon)
    out_dim = int(len(qs)) if qs else 1

    rows: list[dict[str, Any]] = []
    if qs:
        if yhat_scaled.shape != (int(len(pred_uids)), h, out_dim):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}, {out_dim}), got {yhat_scaled.shape}"
            )

        yhat_all = yhat_scaled * pred_std.reshape(-1, 1, 1) + pred_mean.reshape(-1, 1, 1)
        q_point = _pick_point_quantile(qs)
        point_idx = int(qs.index(q_point))
        yhat_point = yhat_all[:, :, point_idx]

        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                row = {"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat_point[i, j])}
                for kk, qv in enumerate(qs):
                    row[_quantile_col(float(qv))] = float(yhat_all[i, j, kk])
                rows.append(row)
    else:
        if yhat_scaled.shape != (int(len(pred_uids)), h):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
            )

        yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)
        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def _predict_torch_tcn_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    channels: Any,
    kernel_size: int,
    dilation_base: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    TCN global/panel model (causal dilated Conv1D residual blocks).

    The model consumes the full (context + horizon) token sequence:
      - context tokens contain observed y
      - horizon tokens contain y=0 but include known future covariates/time features
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    if isinstance(channels, int):
        chs = (int(channels),)
    elif isinstance(channels, str):
        parts = [p.strip() for p in channels.split(",") if p.strip()]
        chs = tuple(int(p) for p in parts)
    elif isinstance(channels, list | tuple):
        chs = tuple(int(c) for c in channels)
    else:
        chs = (int(channels),)

    chs = tuple(int(c) for c in chs if int(c) > 0)
    if not chs:
        raise ValueError("channels must contain at least one positive int")

    k = int(kernel_size)
    if k <= 0:
        raise ValueError(_KERNEL_SIZE_MIN_MSG)
    dbase = int(dilation_base)
    if dbase <= 0:
        raise ValueError("dilation_base must be >= 1")
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    class _CausalConv1d(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, *, dilation: int) -> None:
            super().__init__()
            self.left_pad = int((k - 1) * dilation)
            self.conv = nn.Conv1d(
                in_channels=int(in_ch),
                out_channels=int(out_ch),
                kernel_size=int(k),
                dilation=int(dilation),
                padding=0,
            )

        def forward(self, xch: Any) -> Any:
            # Pad left only to keep causality.
            if int(self.left_pad) > 0:
                xch = F.pad(xch, (int(self.left_pad), 0))
            return self.conv(xch)

    class _TCNBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, *, dilation: int) -> None:
            super().__init__()
            self.conv1 = _CausalConv1d(int(in_ch), int(out_ch), dilation=int(dilation))
            self.conv2 = _CausalConv1d(int(out_ch), int(out_ch), dilation=int(dilation))
            self.act = nn.ReLU()
            self.drop = nn.Dropout(p=float(drop))
            self.skip = (
                nn.Identity()
                if int(in_ch) == int(out_ch)
                else nn.Conv1d(int(in_ch), int(out_ch), kernel_size=1)
            )

        def forward(self, xch: Any) -> Any:
            z = self.drop(self.act(self.conv1(xch)))
            z = self.drop(self.act(self.conv2(z)))
            return z + self.skip(xch)

    class _TCNGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            blocks: list[Any] = []
            in_ch = int(input_dim + int(id_emb_dim))
            for i, out_ch in enumerate(chs):
                dilation = int(dbase ** int(i))
                blocks.append(_TCNBlock(in_ch, int(out_ch), dilation=int(dilation)))
                in_ch = int(out_ch)
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Linear(int(in_ch), int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)  # (B,E)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)  # (B,T,C)
            xch = x2.transpose(1, 2)  # (B,C,T)
            z = self.blocks(xch)  # (B,C,T)
            zt = z.transpose(1, 2)  # (B,T,C)
            out = self.head(zt[:, -seq_len:, :])  # (B,seq_len,out_dim)
            yhat = out[:, -h:, :]  # (B,h,out_dim)
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _TCNGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_tcn_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    channels: Any = (64, 64, 64),
    kernel_size: int = 3,
    dilation_base: int = 2,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Torch TCN global/panel forecaster.

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_tcn_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            channels=channels,
            kernel_size=int(kernel_size),
            dilation_base=int(dilation_base),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_nbeats_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    num_blocks: int,
    num_layers: int,
    layer_width: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    N-BEATS-style global/panel model (generic residual MLP blocks).

    This implementation flattens the full (context+horizon) token grid and learns
    backcast/forecast residual updates through MLP blocks.
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    if int(num_blocks) <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(layer_width) <= 0:
        raise ValueError("layer_width must be >= 1")
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    backcast_dim = int(seq_len * input_dim + int(id_emb_dim))
    forecast_dim = int(h * out_dim)

    class _NBeatsBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            in_dim = int(backcast_dim)
            for _ in range(int(num_layers)):
                layers.append(nn.Linear(int(in_dim), int(layer_width)))
                layers.append(nn.ReLU())
                if float(drop) > 0.0:
                    layers.append(nn.Dropout(p=float(drop)))
                in_dim = int(layer_width)
            self.mlp = nn.Sequential(*layers)
            self.theta = nn.Linear(int(layer_width), int(backcast_dim + forecast_dim))

        def forward(self, xb: Any) -> tuple[Any, Any]:
            z = self.mlp(xb)
            theta = self.theta(z)
            back = theta[:, : int(backcast_dim)]
            fc = theta[:, int(backcast_dim) :]
            return back, fc

    class _NBeatsGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.blocks = nn.ModuleList([_NBeatsBlock() for _ in range(int(num_blocks))])

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)  # (B,E)
            x_flat = xb.reshape(xb.shape[0], -1)  # (B, seq_len*input_dim)
            residual = torch.cat([x_flat, emb], dim=-1)  # (B, backcast_dim)

            forecast = torch.zeros(
                (xb.shape[0], int(forecast_dim)), device=xb.device, dtype=xb.dtype
            )
            for blk in self.blocks:
                back, fc = blk(residual)
                residual = residual - back
                forecast = forecast + fc

            out = forecast.reshape(-1, h, out_dim)
            return out.squeeze(-1) if out_dim == 1 else out

    model = _NBeatsGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_nbeats_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    num_blocks: int = 3,
    num_layers: int = 2,
    layer_width: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Torch N-BEATS-style global/panel forecaster.

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_nbeats_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            num_blocks=int(num_blocks),
            num_layers=int(num_layers),
            layer_width=int(layer_width),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_nhits_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    pool_sizes: Any,
    num_blocks: int,
    num_layers: int,
    layer_width: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    N-HiTS-style global/panel model (lite, residual multi-rate MLP).

    This variant focuses on the target channel y only (context window), with an id
    embedding as a conditioning vector.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)

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
    if any(int(p) > ctx for p in pools):
        raise ValueError("pool_sizes values must be <= context_length")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    forecast_dim = int(h * out_dim)

    class _NHITSBlock(nn.Module):
        def __init__(self, pool: int) -> None:
            super().__init__()
            self.pool = int(pool)
            n_low = 1 + (ctx - int(pool)) // int(pool)
            if n_low <= 0:
                raise ValueError("Invalid pool configuration")

            layers: list[Any] = []
            in_dim = int(n_low + int(id_emb_dim))
            for _ in range(int(num_layers)):
                layers.append(nn.Linear(int(in_dim), int(layer_width)))
                layers.append(nn.ReLU())
                if float(drop) > 0.0:
                    layers.append(nn.Dropout(p=float(drop)))
                in_dim = int(layer_width)
            self.mlp = nn.Sequential(*layers)
            self.theta = nn.Linear(int(layer_width), int(n_low + forecast_dim))

        def forward(self, residual: Any, emb: Any) -> tuple[Any, Any]:
            # residual: (B, ctx)
            x1 = residual.unsqueeze(1)
            low = F.avg_pool1d(x1, kernel_size=int(self.pool), stride=int(self.pool)).squeeze(1)
            z_in = torch.cat([low, emb], dim=-1)
            z = self.mlp(z_in)
            theta = self.theta(z)
            n_low = int(low.shape[1])
            back_low = theta[:, :n_low]
            fc = theta[:, n_low:]

            back_up = F.interpolate(
                back_low.unsqueeze(1),
                size=int(ctx),
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            return back_up, fc

    class _NHITSGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            blocks: list[Any] = []
            for i in range(int(num_blocks)):
                blocks.append(_NHITSBlock(pool=pools[int(i) % len(pools)]))
            self.blocks = nn.ModuleList(blocks)

        def forward(self, xb: Any, ids: Any) -> Any:
            # Use only target y context as residual series.
            residual = xb[:, :ctx, 0]  # (B, ctx)
            emb = self.id_emb(ids)  # (B, E)

            forecast = torch.zeros(
                (xb.shape[0], int(forecast_dim)), device=xb.device, dtype=xb.dtype
            )
            for blk in self.blocks:
                back, fc = blk(residual, emb)
                residual = residual - back
                forecast = forecast + fc

            out = forecast.reshape(-1, h, out_dim)
            return out.squeeze(-1) if out_dim == 1 else out

    model = _NHITSGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_nhits_global_forecaster(
    *,
    context_length: int = 192,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    pool_sizes: Any = (1, 2, 4),
    num_blocks: int = 6,
    num_layers: int = 2,
    layer_width: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Torch N-HiTS-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_nhits_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            pool_sizes=pool_sizes,
            num_blocks=int(num_blocks),
            num_layers=int(num_layers),
            layer_width=int(layer_width),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_tide_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    hidden_size: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    TiDE-style global/panel model (lite, deterministic or quantile regression).

    - Encoder: MLP over flattened context tokens (+ id embedding)
    - Decoder: step embeddings + known future covariates/time features -> per-step MLP head
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    hidden = int(hidden_size)
    if hidden <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    ctx = int(context_length)
    enc_in_dim = int(ctx * input_dim + int(id_emb_dim))

    class _TiDEGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.enc = nn.Sequential(
                nn.Linear(enc_in_dim, int(hidden)),
                nn.ReLU(),
                nn.Dropout(p=float(drop)),
                nn.Linear(int(hidden), int(d)),
            )
            self.step_emb = nn.Embedding(int(h), int(d))
            self.fut_proj = nn.Linear(int(input_dim), int(d))
            self.dec = nn.Sequential(
                nn.Linear(int(d), int(hidden)),
                nn.ReLU(),
                nn.Dropout(p=float(drop)),
                nn.Linear(int(hidden), int(out_dim)),
            )

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)  # (B,E)
            ctx_flat = xb[:, :ctx, :].reshape(xb.shape[0], -1)
            ctx_vec = self.enc(torch.cat([ctx_flat, emb], dim=-1))  # (B,d)

            steps = self.step_emb(torch.arange(h, device=xb.device, dtype=torch.long))  # (h,d)
            fut = xb[:, ctx:, :]  # (B,h,input_dim)
            fut_e = self.fut_proj(fut)  # (B,h,d)

            z = ctx_vec.unsqueeze(1) + steps.unsqueeze(0) + fut_e
            out = self.dec(z)  # (B,h,out_dim)
            return out.squeeze(-1) if out_dim == 1 else out

    model = _TiDEGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_tide_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    hidden_size: int = 128,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Torch TiDE-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_tide_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            hidden_size=int(hidden_size),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_wavenet_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    channels: int,
    num_layers: int,
    kernel_size: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    WaveNet-style global/panel model (gated dilated causal CNN).

    Uses the full (context + horizon) token sequence with known future covariates/time features.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    c = int(channels)
    if c <= 0:
        raise ValueError(_CHANNELS_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    k = int(kernel_size)
    if k <= 0:
        raise ValueError(_KERNEL_SIZE_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    in_ch = int(input_dim + int(id_emb_dim))

    class _WaveNetGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Conv1d(int(in_ch), int(c), kernel_size=1)

            self.filter_convs = nn.ModuleList()
            self.gate_convs = nn.ModuleList()
            self.res_convs = nn.ModuleList()
            self.skip_convs = nn.ModuleList()
            self.drop = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

            for i in range(int(num_layers)):
                dilation = 2 ** int(i)
                self.filter_convs.append(
                    nn.Conv1d(int(c), int(c), kernel_size=int(k), dilation=int(dilation))
                )
                self.gate_convs.append(
                    nn.Conv1d(int(c), int(c), kernel_size=int(k), dilation=int(dilation))
                )
                self.res_convs.append(nn.Conv1d(int(c), int(c), kernel_size=1))
                self.skip_convs.append(nn.Conv1d(int(c), int(c), kernel_size=1))

            self.out1 = nn.Conv1d(int(c), int(c), kernel_size=1)
            self.out2 = nn.Conv1d(int(c), int(c), kernel_size=1)
            self.head = nn.Linear(int(c), int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)  # (B,E)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)  # (B,T,C)

            xch = x2.transpose(1, 2)  # (B,C,T)
            xh = self.in_proj(xch)
            skip = None

            for i in range(int(num_layers)):
                dilation = 2 ** int(i)
                pad = int(dilation) * (int(k) - 1)
                xpad = F.pad(xh, (int(pad), 0), mode="constant", value=0.0)
                f = torch.tanh(self.filter_convs[i](xpad))
                g = torch.sigmoid(self.gate_convs[i](xpad))
                z = self.drop(f * g)
                s = self.skip_convs[i](z)
                skip = s if skip is None else skip + s
                xh = self.res_convs[i](z) + xh

            out = torch.relu(skip)
            out = torch.relu(self.out1(out))
            out = self.out2(out)  # (B,c,T)

            zt = out.transpose(1, 2)  # (B,T,c)
            yhat = self.head(zt[:, -int(seq_len) :, :])[:, -h:, :]
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _WaveNetGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_wavenet_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    channels: int = 32,
    num_layers: int = 6,
    kernel_size: int = 2,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Torch WaveNet-style global/panel forecaster.

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_wavenet_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            channels=int(channels),
            num_layers=int(num_layers),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_resnet1d_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    channels: int,
    num_blocks: int,
    kernel_size: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    ResNet-1D global/panel model (token-wise residual Conv1D blocks).
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    c = int(channels)
    if c <= 0:
        raise ValueError(_CHANNELS_MIN_MSG)
    if int(num_blocks) <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    k = int(kernel_size)
    if k <= 0:
        raise ValueError(_KERNEL_SIZE_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    pad = int(k) // 2
    in_ch = int(input_dim + int(id_emb_dim))

    class _ResBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(int(c), int(c), kernel_size=int(k), padding=int(pad))
            self.conv2 = nn.Conv1d(int(c), int(c), kernel_size=int(k), padding=int(pad))
            self.act = nn.ReLU()
            self.drop = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

        def forward(self, xch: Any) -> Any:
            z = self.act(self.conv1(xch))
            z = self.drop(z)
            z = self.conv2(z)
            return self.act(xch + z)

    class _ResNet1DGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Conv1d(int(in_ch), int(c), kernel_size=1)
            self.blocks = nn.ModuleList([_ResBlock() for _ in range(int(num_blocks))])
            self.head = nn.Linear(int(c), int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)  # (B,E)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)  # (B,T,C)

            xch = x2.transpose(1, 2)  # (B,C,T)
            z = self.in_proj(xch)
            for blk in self.blocks:
                z = blk(z)

            zt = z.transpose(1, 2)  # (B,T,c)
            yhat = self.head(zt[:, -int(seq_len) :, :])[:, -h:, :]
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _ResNet1DGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_resnet1d_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    channels: int = 32,
    num_blocks: int = 4,
    kernel_size: int = 3,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Torch ResNet-1D global/panel forecaster.

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_resnet1d_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            channels=int(channels),
            num_blocks=int(num_blocks),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_inception_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    channels: int,
    num_blocks: int,
    kernel_sizes: Any,
    bottleneck_channels: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    InceptionTime-style global/panel model (lite).
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    c = int(channels)
    if c <= 0:
        raise ValueError(_CHANNELS_MIN_MSG)
    if int(num_blocks) <= 0:
        raise ValueError(_NUM_BLOCKS_MIN_MSG)
    b = int(bottleneck_channels)
    if b <= 0:
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

    ks = tuple(int(k) for k in ks if int(k) > 0)
    if not ks:
        raise ValueError("kernel_sizes must contain at least one positive int")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    in_ch = int(input_dim + int(id_emb_dim))

    class _InceptionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bottleneck = nn.Conv1d(int(c), int(b), kernel_size=1)
            self.convs = nn.ModuleList(
                [nn.Conv1d(int(b), int(c), kernel_size=int(k), padding=int(k) // 2) for k in ks]
            )
            self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            self.pool_conv = nn.Conv1d(int(c), int(c), kernel_size=1)
            self.proj = nn.Conv1d(int(c) * (len(ks) + 1), int(c), kernel_size=1)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

        def forward(self, xch: Any) -> Any:  # (B,c,T)
            z0 = self.bottleneck(xch)
            outs = [conv(z0) for conv in self.convs]
            outs.append(self.pool_conv(self.pool(xch)))
            z = torch.cat(outs, dim=1)
            z = self.act(self.proj(z))
            return self.drop(z)

    class _InceptionGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Conv1d(int(in_ch), int(c), kernel_size=1)
            self.blocks = nn.ModuleList([_InceptionBlock() for _ in range(int(num_blocks))])
            self.head = nn.Linear(int(c), int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)  # (B,T,C)

            xch = x2.transpose(1, 2)
            z = self.in_proj(xch)
            for blk in self.blocks:
                z = z + blk(z)

            zt = z.transpose(1, 2)
            yhat = self.head(zt[:, -int(seq_len) :, :])[:, -h:, :]
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _InceptionGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_inception_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    channels: int = 32,
    num_blocks: int = 3,
    kernel_sizes: Any = (3, 5, 7),
    bottleneck_channels: int = 16,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Torch InceptionTime-style global/panel forecaster.

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_inception_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            channels=int(channels),
            num_blocks=int(num_blocks),
            kernel_sizes=kernel_sizes,
            bottleneck_channels=int(bottleneck_channels),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_lstnet_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    cnn_channels: int,
    kernel_size: int,
    rnn_hidden: int,
    skip: int,
    highway_window: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    LSTNet-style global/panel model (CNN + GRU + skip GRU + highway, lite).

    This is a deterministic/quantile-regression direct multi-horizon model.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    input_dim = int(x_train.shape[2])

    c = int(cnn_channels)
    if c <= 0:
        raise ValueError("cnn_channels must be >= 1")
    k = int(kernel_size)
    if k <= 0:
        raise ValueError(_KERNEL_SIZE_MIN_MSG)
    hidden = int(rnn_hidden)
    if hidden <= 0:
        raise ValueError("rnn_hidden must be >= 1")
    skip_int = int(skip)
    if skip_int < 0:
        raise ValueError("skip must be >= 0")
    hw = int(highway_window)
    if hw < 0:
        raise ValueError("highway_window must be >= 0")
    if hw > 0 and int(ctx) < int(hw):
        raise ValueError("highway_window must be <= context_length")
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    in_ch = int(input_dim + int(id_emb_dim))

    class _LSTNetGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.conv = nn.Conv1d(int(in_ch), int(c), kernel_size=int(k))
            self.drop = nn.Dropout(p=float(drop))
            self.gru = _make_manual_gru(
                input_size=int(c),
                hidden_size=int(hidden),
                num_layers=1,
                dropout=0.0,
                bidirectional=False,
            )
            self.skip = int(skip_int)
            self.skip_gru = (
                None
                if self.skip <= 0
                else _make_manual_gru(
                    input_size=int(c),
                    hidden_size=int(hidden),
                    num_layers=1,
                    dropout=0.0,
                    bidirectional=False,
                )
            )
            feat_dim = int(hidden) * (2 if self.skip_gru is not None else 1)
            self.proj = nn.Linear(int(feat_dim), int(h * out_dim))

            self.hw = int(hw)
            self.highway = None if self.hw <= 0 else nn.Linear(int(self.hw), int(h))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)  # (B,E)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)  # (B,T,C)

            xch = x2.transpose(1, 2)  # (B,C,T)
            z = F.relu(self.conv(xch))  # (B, cnn_channels, T')
            z = self.drop(z)
            zt = z.transpose(1, 2)  # (B,T',cnn_channels)

            _out, h_main = self.gru(zt)
            h_main = h_main[-1]  # (B,H)

            if self.skip_gru is not None:
                T = int(zt.shape[1])
                s = int(self.skip)
                n = T // s
                if n <= 0:
                    h_skip = torch.zeros_like(h_main)
                else:
                    z2 = zt[:, -n * s :, :].reshape(zt.shape[0], n, s, zt.shape[2])
                    z3 = z2[:, :, -1, :]  # (B,n,C)
                    _out2, h_s = self.skip_gru(z3)
                    h_skip = h_s[-1]
                feat = torch.cat([h_main, h_skip], dim=-1)
            else:
                feat = h_main

            out = self.proj(feat).reshape(-1, h, out_dim)  # (B,h,out_dim)

            if self.highway is not None:
                last = xb[:, int(ctx) - int(self.hw) : int(ctx), 0]
                out = out + self.highway(last).unsqueeze(-1)

            return out.squeeze(-1) if out_dim == 1 else out

    model = _LSTNetGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_lstnet_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    cnn_channels: int = 16,
    kernel_size: int = 6,
    rnn_hidden: int = 32,
    skip: int = 24,
    highway_window: int = 24,
    id_emb_dim: int = 8,
    dropout: float = 0.2,
    quantiles: Any = (),
) -> Any:
    """
    Torch LSTNet-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_lstnet_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            cnn_channels=int(cnn_channels),
            kernel_size=int(kernel_size),
            rnn_hidden=int(rnn_hidden),
            skip=int(skip),
            highway_window=int(highway_window),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_fnet_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    FNet-style global/panel model (FFT token mixing instead of attention, lite).
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    class _FNetLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(
                nn.Linear(d, int(dim_feedforward)),
                nn.GELU(),
                nn.Dropout(p=float(drop)),
                nn.Linear(int(dim_feedforward), d),
                nn.Dropout(p=float(drop)),
            )

        def forward(self, x: Any) -> Any:  # (B, T, d)
            z = self.norm1(x)
            z = torch.fft.fft(z, dim=1).real
            x = x + z
            z = self.norm2(x)
            x = x + self.ff(z)
            return x

    class _FNetGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.pos = nn.Parameter(torch.zeros((1, int(seq_len), d), dtype=torch.float32))
            self.layers = nn.ModuleList([_FNetLayer() for _ in range(int(num_layers))])
            self.out = nn.Linear(d, int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2) + self.pos
            for layer in self.layers:
                z = layer(z)
            yhat = self.out(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _FNetGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_fnet_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Torch FNet-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_fnet_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_gmlp_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    num_layers: int,
    ffn_dim: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    gMLP-style global/panel model (token mixing via spatial gating, lite).
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

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

    class _SpatialGatingUnit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(ffn)
            self.proj = nn.Linear(int(seq_len), int(seq_len))

        def forward(self, v: Any) -> Any:  # (B, T, ffn)
            z = self.norm(v)
            z = z.transpose(1, 2)  # (B, ffn, T)
            z = self.proj(z)
            return z.transpose(1, 2)

    class _gMLPLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, 2 * ffn)
            self.act = nn.GELU()
            self.sgu = _SpatialGatingUnit()
            self.fc2 = nn.Linear(ffn, d)
            self.drop = nn.Dropout(p=float(drop))

        def forward(self, x: Any) -> Any:
            z = self.norm(x)
            z = self.fc1(z)
            u, v = z.chunk(2, dim=-1)
            u = self.act(u)
            v = self.sgu(v)
            z2 = u * v
            z2 = self.fc2(self.drop(z2))
            return x + self.drop(z2)

    class _gMLPGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.pos = nn.Parameter(torch.zeros((1, int(seq_len), d), dtype=torch.float32))
            self.layers = nn.ModuleList([_gMLPLayer() for _ in range(int(num_layers))])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2) + self.pos
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            yhat = self.out(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _gMLPGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_gmlp_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    num_layers: int = 4,
    ffn_dim: int = 128,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Torch gMLP-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_gmlp_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            num_layers=int(num_layers),
            ffn_dim=int(ffn_dim),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_ssm_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    num_layers: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    Diagonal state-space global/panel model (SSM, lite).

    Implements a stable exponential-decay state update per feature dimension.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    class _SSMBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.u_proj = nn.Linear(d, d)
            self.gate = nn.Linear(d, d)
            self.log_decay = nn.Parameter(torch.zeros((d,), dtype=torch.float32))
            self.out_proj = nn.Linear(d, d)
            self.drop = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

        def forward(self, x: Any) -> Any:  # (B, T, d)
            z = self.norm(x)
            u = self.u_proj(z)
            g = torch.sigmoid(self.gate(z))
            a = torch.exp(-F.softplus(self.log_decay)).reshape(1, 1, -1)  # (1,1,d) in (0,1)

            B = int(u.shape[0])
            T = int(u.shape[1])
            s = torch.zeros((B, d), device=u.device, dtype=u.dtype)
            outs: list[Any] = []
            for t in range(T):
                s = a[:, 0, :] * s + (1.0 - a[:, 0, :]) * u[:, t, :]
                y = self.out_proj(s)
                outs.append(g[:, t, :] * y + (1.0 - g[:, t, :]) * u[:, t, :])

            y_seq = torch.stack(outs, dim=1)
            y_seq = self.drop(y_seq)
            return x + y_seq

    class _SSMGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.pos = nn.Parameter(torch.zeros((1, int(seq_len), d), dtype=torch.float32))
            self.layers = nn.ModuleList([_SSMBlock() for _ in range(int(num_layers))])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2) + self.pos
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            yhat = self.out(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _SSMGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_ssm_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    num_layers: int = 4,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Torch diagonal state-space global/panel forecaster (SSM, lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_ssm_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            num_layers=int(num_layers),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_mamba_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    num_layers: int,
    conv_kernel: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    Mamba-style selective SSM global/panel model (lite).

    Implements:
      - causal depthwise conv for short-range mixing
      - input-dependent exponential-decay state update per channel
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    k = int(conv_kernel)
    if k <= 0:
        raise ValueError("conv_kernel must be >= 1")
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

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

        def forward(self, x: Any) -> Any:  # (B, T, d)
            z = self.norm(x)
            u = self.u_proj(z)  # (B, T, d)

            # Causal depthwise conv on u (short-range mixing).
            u_ch = u.transpose(1, 2)  # (B, d, T)
            u_pad = F.pad(u_ch, (int(k) - 1, 0))
            u = self.dwconv(u_pad).transpose(1, 2)  # (B, T, d)

            delta = F.softplus(self.delta_proj(z))  # (B, T, d)
            a = torch.exp(-delta * F.softplus(self.log_A).reshape(1, 1, -1))  # (B, T, d)
            g = torch.sigmoid(self.gate_proj(z))

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
            return x + y_seq

    class _MambaGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.pos = nn.Parameter(torch.zeros((1, int(seq_len), d), dtype=torch.float32))
            self.layers = nn.ModuleList([_MambaBlock() for _ in range(int(num_layers))])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2) + self.pos
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            yhat = self.out(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _MambaGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_mamba_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    num_layers: int = 4,
    conv_kernel: int = 3,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Mamba-style selective SSM global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_mamba_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            num_layers=int(num_layers),
            conv_kernel=int(conv_kernel),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_rwkv_global(
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    num_layers: int,
    ffn_dim: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    RWKV-style time-mix + channel-mix global/panel model (lite).
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
    input_dim = int(x_train.shape[2])

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

        def _time_mix(self, x: Any) -> Any:
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

        def _channel_mix(self, x: Any) -> Any:
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

    class _RWKVGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.pos = nn.Parameter(torch.zeros((1, int(seq_len), d), dtype=torch.float32))
            self.layers = nn.ModuleList([_RWKVBlock() for _ in range(int(num_layers))])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2) + self.pos
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            yhat = self.out(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _RWKVGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_rwkv_global_forecaster(
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
    device: str = "cpu",
    d_model: int = 64,
    num_layers: int = 4,
    ffn_dim: int = 128,
    id_emb_dim: int = 8,
    dropout: float = 0.0,
    quantiles: Any = (),
) -> Any:
    """
    RWKV-style time-mix + channel-mix global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_rwkv_global(
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
            device=str(device),
            d_model=int(d_model),
            num_layers=int(num_layers),
            ffn_dim=int(ffn_dim),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_hyena_global(
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    num_layers: int,
    ffn_dim: int,
    kernel_size: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    Hyena-style long convolution global/panel model (lite).

    Uses a depthwise causal Conv1D (long kernel) with gating for token mixing,
    followed by a channel-mixing FFN.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
    input_dim = int(x_train.shape[2])

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

    class _HyenaBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.in_proj = nn.Linear(d, 2 * d)
            self.dwconv = nn.Conv1d(d, d, kernel_size=int(k), groups=d)
            self.out_proj = nn.Linear(d, d)
            self.drop1 = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

            self.norm2 = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, hidden)
            self.fc2 = nn.Linear(hidden, d)
            self.drop2 = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

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

    class _HyenaGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.pos = nn.Parameter(torch.zeros((1, int(seq_len), d), dtype=torch.float32))
            self.layers = nn.ModuleList([_HyenaBlock() for _ in range(int(num_layers))])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2) + self.pos
            for layer in self.layers:
                z = layer(z)
            z = self.norm(z)
            yhat = self.out(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _HyenaGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_hyena_global_forecaster(
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
    device: str = "cpu",
    d_model: int = 64,
    num_layers: int = 4,
    ffn_dim: int = 128,
    kernel_size: int = 64,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Hyena-style long convolution global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_hyena_global(
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
            device=str(device),
            d_model=int(d_model),
            num_layers=int(num_layers),
            ffn_dim=int(ffn_dim),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_dilated_rnn_global(
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    cell: str,
    d_model: int,
    num_layers: int,
    dilation_base: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    Dilated RNN (lite) trained globally across a panel/long-format dataset.

    Uses fixed dilations per layer (1, dilation_base, dilation_base^2, ...) to expand
    receptive field while keeping the sequence length unchanged.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
    seq_len = int(ctx + h)
    input_dim = int(x_train.shape[2])

    cell_s = str(cell).lower().strip()
    if cell_s not in {"gru", "lstm"}:
        raise ValueError("cell must be one of: gru, lstm")

    d = int(d_model)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    L = int(num_layers)
    if L <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    base = int(dilation_base)
    if base <= 1:
        raise ValueError("dilation_base must be >= 2")
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    class _DilatedRNNGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.pos = nn.Parameter(torch.zeros((1, int(seq_len), d), dtype=torch.float32))

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
            self.drop = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()
            self.out = nn.Linear(d, int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:  # xb: (B, T, input_dim)
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2) + self.pos  # (B, T, d)

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

            yhat = self.out(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _DilatedRNNGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_dilated_rnn_global_forecaster(
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
    device: str = "cpu",
    cell: str = "gru",
    d_model: int = 64,
    num_layers: int = 3,
    dilation_base: int = 2,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Dilated RNN global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_dilated_rnn_global(
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
            device=str(device),
            cell=str(cell),
            d_model=int(d_model),
            num_layers=int(num_layers),
            dilation_base=int(dilation_base),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_kan_global(
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    num_layers: int,
    grid_size: int,
    grid_range: float,
    dropout: float,
    linear_skip: bool,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    KAN (lite) trained globally across panel series.

    Uses triangular spline basis features on a fixed grid, with per-output learnable
    coefficients, plus an optional linear skip connection.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
    seq_len = int(ctx + h)
    input_dim = int(x_train.shape[2])

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

    flat_dim = int(seq_len * input_dim)
    in_dim = int(flat_dim + int(id_emb_dim))

    class _KANSplineLayer(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            knots = torch.linspace(-grid_r, grid_r, int(K), dtype=torch.float32)
            self.register_buffer("knots", knots, persistent=False)
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
            xk = xb.unsqueeze(-1)
            basis = torch.relu(1.0 - torch.abs(xk - self.knots.reshape(1, 1, -1)) * self.inv_delta)
            y = torch.einsum("bik,oik->bo", basis, self.coeff)
            if self.linear is not None:
                y = y + self.linear(xb)
            return y + self.bias

    class _KANGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            layers: list[Any] = []
            cur = int(in_dim)
            for _i in range(int(L)):
                layers.append(_KANSplineLayer(cur, d))
                layers.append(nn.LayerNorm(d))
                layers.append(nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity())
                cur = d
            self.net = nn.Sequential(*layers)
            self.out = nn.Linear(d, int(h * out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            B = int(xb.shape[0])
            x_flat = xb.reshape(B, -1)
            emb = self.id_emb(ids)
            z0 = torch.cat([x_flat, emb], dim=-1)
            z = self.net(z0)
            z = F.gelu(z)
            y = self.out(z)
            if out_dim == 1:
                return y.reshape(B, int(h))
            return y.reshape(B, int(h), int(out_dim))

    model = _KANGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_kan_global_forecaster(
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
    device: str = "cpu",
    d_model: int = 64,
    num_layers: int = 2,
    grid_size: int = 16,
    grid_range: float = 2.0,
    linear_skip: bool = True,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    KAN-style spline MLP global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_kan_global(
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
            device=str(device),
            d_model=int(d_model),
            num_layers=int(num_layers),
            grid_size=int(grid_size),
            grid_range=float(grid_range),
            dropout=float(dropout),
            linear_skip=bool(linear_skip),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_scinet_global(
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    num_stages: int,
    conv_kernel: int,
    ffn_dim: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    SCINet-style (lite) global/panel model.

    Repeatedly splits tokens into even/odd subsequences, applies convolutional interaction,
    merges back, and predicts the horizon tokens.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
    seq_len = int(ctx + h)
    input_dim = int(x_train.shape[2])

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
            x_ch = xbt.transpose(1, 2)
            x_pad = F.pad(x_ch, (int(pad_left), int(pad_right)))
            return conv(x_pad).transpose(1, 2)

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

            out = torch.zeros_like(z)
            out[:, ::2, :] = even2
            out[:, 1::2, :] = odd2

            xb = xb + self.drop1(self.out_proj(out))

            z2 = self.norm2(xb)
            y2 = F.gelu(self.fc1(z2))
            y2 = self.fc2(self.drop2(y2))
            xb = xb + self.drop2(y2)
            return xb

    class _SCINetGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.pos = nn.Parameter(torch.zeros((1, int(seq_len), d), dtype=torch.float32))
            self.blocks = nn.ModuleList([_SCIBlock() for _ in range(int(stages))])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2) + self.pos
            for blk in self.blocks:
                z = blk(z)
            z = self.norm(z)
            yhat = self.out(z[:, -h:, :])
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _SCINetGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_scinet_global_forecaster(
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
    device: str = "cpu",
    d_model: int = 64,
    num_stages: int = 3,
    conv_kernel: int = 5,
    ffn_dim: int = 128,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    SCINet-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_scinet_global(
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
            device=str(device),
            d_model=int(d_model),
            num_stages=int(num_stages),
            conv_kernel=int(conv_kernel),
            ffn_dim=int(ffn_dim),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_etsformer_global(
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    alpha_init: float,
    beta_init: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    ETSformer-style exponential smoothing + Transformer residual model (lite), global/panel.

    - Holt-style baseline forecast from the context tokens (learned alpha/beta).
    - Transformer on residual tokens (with future covariates/time features available).
    - Output = baseline + residual_adjustment.
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
    seq_len = int(ctx + h)
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_DIVISIBLE_BY_NHEAD_MSG)
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

    def _logit(p: float) -> float:
        p2 = min(max(float(p), 1e-4), 1.0 - 1e-4)
        return math.log(p2 / (1.0 - p2))

    alpha_logit_init = _logit(alpha0)
    beta_logit_init = _logit(beta0)

    class _ETSformerGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init, dtype=torch.float32))
            self.beta_logit = nn.Parameter(torch.tensor(beta_logit_init, dtype=torch.float32))
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.pos = nn.Parameter(torch.zeros((1, int(seq_len), d), dtype=torch.float32))
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=int(dim_feedforward),
                dropout=float(drop),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=int(num_layers))
            self.out = nn.Linear(d, int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:  # xb: (B, seq_len, input_dim)
            y_ctx = xb[:, :ctx, 0]  # (B, ctx)
            B = int(y_ctx.shape[0])

            alpha = torch.sigmoid(self.alpha_logit)
            beta = torch.sigmoid(self.beta_logit)

            level = y_ctx[:, 0]
            if int(ctx) >= 2:
                b = y_ctx[:, 1] - y_ctx[:, 0]
            else:
                b = torch.zeros((B,), device=y_ctx.device, dtype=y_ctx.dtype)

            levels: list[Any] = [level]
            trends: list[Any] = [b]
            for t in range(1, int(ctx)):
                yt = y_ctx[:, t]
                level_new = alpha * yt + (1.0 - alpha) * (level + b)
                b_new = beta * (level_new - level) + (1.0 - beta) * b
                level, b = level_new, b_new
                levels.append(level)
                trends.append(b)

            level_seq = torch.stack(levels, dim=1)  # (B, ctx)
            trend_seq = torch.stack(trends, dim=1)  # (B, ctx)

            fitted = torch.empty_like(y_ctx)
            fitted[:, 0] = level_seq[:, 0]
            if int(ctx) > 1:
                fitted[:, 1:] = level_seq[:, :-1] + trend_seq[:, :-1]
            resid_ctx = y_ctx - fitted

            resid_y = torch.cat(
                [resid_ctx, torch.zeros((B, int(h)), device=xb.device, dtype=xb.dtype)], dim=1
            ).unsqueeze(-1)

            if int(input_dim) > 1:
                xb_resid = torch.cat([resid_y, xb[:, :, 1:]], dim=-1)
            else:
                xb_resid = resid_y

            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb_resid.shape[1], -1)
            x2 = torch.cat([xb_resid, emb_t], dim=-1)

            steps = torch.arange(1, int(h) + 1, device=xb.device, dtype=xb.dtype).reshape(1, -1)
            baseline = level_seq[:, -1].unsqueeze(1) + steps * trend_seq[:, -1].unsqueeze(1)

            z = self.in_proj(x2) + self.pos
            z = self.enc(z)
            delta = self.out(z[:, -h:, :])
            if out_dim == 1:
                return baseline + delta.squeeze(-1)
            return baseline.unsqueeze(-1) + delta

    model = _ETSformerGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_etsformer_global_forecaster(
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    alpha_init: float = 0.3,
    beta_init: float = 0.1,
    id_emb_dim: int = 8,
    quantiles: Any = (),
) -> Any:
    """
    ETSformer-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_etsformer_global(
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            alpha_init=float(alpha_init),
            beta_init=float(beta_init),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_esrnn_global(
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    cell: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    alpha_init: float,
    beta_init: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    ESRNN-style hybrid (lite), global/panel:
      - Holt-style exponential smoothing baseline on the context y channel
      - RNN residual model over (residual_y + covariates/time features)
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
    input_dim = int(x_train.shape[2])

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

    def _logit(p: float) -> float:
        p2 = min(max(float(p), 1e-4), 1.0 - 1e-4)
        return math.log(p2 / (1.0 - p2))

    alpha_logit_init = _logit(alpha0)
    beta_logit_init = _logit(beta0)

    class _ESRNNGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init, dtype=torch.float32))
            self.beta_logit = nn.Parameter(torch.tensor(beta_logit_init, dtype=torch.float32))
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.drop = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()
            if cell_s == "gru":
                rnn_drop = float(drop) if int(L) > 1 else 0.0
                self.rnn = _make_manual_gru(
                    input_size=int(d),
                    hidden_size=int(d),
                    num_layers=int(L),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
            else:
                rnn_drop = float(drop) if int(L) > 1 else 0.0
                self.rnn = _make_manual_lstm(
                    input_size=int(d),
                    hidden_size=int(d),
                    num_layers=int(L),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:  # xb: (B, seq_len, input_dim)
            y_ctx = xb[:, :ctx, 0]  # (B, ctx)
            B = int(y_ctx.shape[0])

            alpha = torch.sigmoid(self.alpha_logit)
            beta = torch.sigmoid(self.beta_logit)

            level = y_ctx[:, 0]
            if int(ctx) >= 2:
                b = y_ctx[:, 1] - y_ctx[:, 0]
            else:
                b = torch.zeros((B,), device=y_ctx.device, dtype=y_ctx.dtype)

            levels: list[Any] = [level]
            trends: list[Any] = [b]
            for t in range(1, int(ctx)):
                yt = y_ctx[:, t]
                level_new = alpha * yt + (1.0 - alpha) * (level + b)
                b_new = beta * (level_new - level) + (1.0 - beta) * b
                level, b = level_new, b_new
                levels.append(level)
                trends.append(b)

            level_seq = torch.stack(levels, dim=1)  # (B, ctx)
            trend_seq = torch.stack(trends, dim=1)  # (B, ctx)

            fitted = torch.empty_like(y_ctx)
            fitted[:, 0] = level_seq[:, 0]
            if int(ctx) > 1:
                fitted[:, 1:] = level_seq[:, :-1] + trend_seq[:, :-1]
            resid_ctx = y_ctx - fitted

            resid_y = torch.cat(
                [resid_ctx, torch.zeros((B, int(h)), device=xb.device, dtype=xb.dtype)], dim=1
            ).unsqueeze(-1)

            if int(input_dim) > 1:
                xb_resid = torch.cat([resid_y, xb[:, :, 1:]], dim=-1)
            else:
                xb_resid = resid_y

            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb_resid.shape[1], -1)
            x2 = torch.cat([xb_resid, emb_t], dim=-1)

            steps = torch.arange(1, int(h) + 1, device=xb.device, dtype=xb.dtype).reshape(1, -1)
            baseline = level_seq[:, -1].unsqueeze(1) + steps * trend_seq[:, -1].unsqueeze(1)

            z0 = self.drop(self.in_proj(x2))
            out, _st = self.rnn(z0)
            out = self.norm(out)
            delta = self.out(out[:, -h:, :])
            if out_dim == 1:
                return baseline + delta.squeeze(-1)
            return baseline.unsqueeze(-1) + delta

    model = _ESRNNGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_esrnn_global_forecaster(
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
    device: str = "cpu",
    cell: str = "gru",
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    alpha_init: float = 0.3,
    beta_init: float = 0.1,
    id_emb_dim: int = 8,
    quantiles: Any = (),
) -> Any:
    """
    ESRNN-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_esrnn_global(
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
            device=str(device),
            cell=str(cell),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            alpha_init=float(alpha_init),
            beta_init=float(beta_init),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_transformer_encdec_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    Encoder-decoder Transformer global/panel model (lite).

    - Encoder attends over the context tokens.
    - Decoder attends over horizon tokens + cross-attends to the encoder.
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    seq_len = ctx + h
    input_dim = int(x_train.shape[2])

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError(_D_MODEL_MIN_MSG)
    if heads <= 0:
        raise ValueError(_NHEAD_MIN_MSG)
    if d % heads != 0:
        raise ValueError(_D_MODEL_DIVISIBLE_BY_NHEAD_MSG)
    if int(num_layers) <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    if int(dim_feedforward) <= 0:
        raise ValueError(_DIM_FEEDFORWARD_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    class _EncoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=heads, dropout=float(drop), batch_first=True
            )
            self.drop = nn.Dropout(p=float(drop))
            self.ff = nn.Sequential(
                nn.Linear(d, int(dim_feedforward)),
                nn.GELU(),
                nn.Dropout(p=float(drop)),
                nn.Linear(int(dim_feedforward), d),
                nn.Dropout(p=float(drop)),
            )

        def forward(self, x: Any) -> Any:
            z = self.norm1(x)
            a, _w = self.attn(z, z, z, need_weights=False)
            x = x + self.drop(a)
            z = self.norm2(x)
            x = x + self.ff(z)
            return x

    class _DecoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)
            self.norm3 = nn.LayerNorm(d)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=heads, dropout=float(drop), batch_first=True
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=heads, dropout=float(drop), batch_first=True
            )
            self.drop = nn.Dropout(p=float(drop))
            self.ff = nn.Sequential(
                nn.Linear(d, int(dim_feedforward)),
                nn.GELU(),
                nn.Dropout(p=float(drop)),
                nn.Linear(int(dim_feedforward), d),
                nn.Dropout(p=float(drop)),
            )

        def forward(self, x: Any, mem: Any, *, attn_mask: Any) -> Any:
            z = self.norm1(x)
            a, _w1 = self.self_attn(z, z, z, attn_mask=attn_mask, need_weights=False)
            x = x + self.drop(a)
            z = self.norm2(x)
            a2, _w2 = self.cross_attn(z, mem, mem, need_weights=False)
            x = x + self.drop(a2)
            z = self.norm3(x)
            x = x + self.ff(z)
            return x

    class _TransformerEncDecGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), d)
            self.pos = nn.Parameter(torch.zeros((1, int(seq_len), d), dtype=torch.float32))
            self.enc_layers = nn.ModuleList([_EncoderLayer() for _ in range(int(num_layers))])
            self.dec_layers = nn.ModuleList([_DecoderLayer() for _ in range(int(num_layers))])
            self.enc_norm = nn.LayerNorm(d)
            self.dec_norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, int(out_dim))

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2) + self.pos

            enc = z[:, :ctx, :]
            dec = z[:, ctx:, :]

            for layer in self.enc_layers:
                enc = layer(enc)
            enc = self.enc_norm(enc)

            attn_mask = torch.triu(
                torch.ones((int(h), int(h)), device=dec.device, dtype=torch.bool), diagonal=1
            )
            for layer in self.dec_layers:
                dec = layer(dec, enc, attn_mask=attn_mask)
            dec = self.dec_norm(dec)

            yhat = self.out(dec)
            return yhat.squeeze(-1) if out_dim == 1 else yhat

    model = _TransformerEncDecGlobal()

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
    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_transformer_encdec_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    quantiles: Any = (),
) -> Any:
    """
    Torch encoder-decoder Transformer global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_transformer_encdec_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_nlinear_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    NLinear-style global/panel model (last-value centering + linear head, lite).

    This implementation focuses on the target y context only, optionally conditioned
    on a series-id embedding.
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    forecast_dim = int(h * out_dim)

    class _NLinearGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.fc = nn.Linear(int(ctx), int(forecast_dim))
            self.id_proj = nn.Linear(int(id_emb_dim), int(forecast_dim), bias=False)
            self.drop = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

        def forward(self, xb: Any, ids: Any) -> Any:
            y_ctx = xb[:, :ctx, 0]  # (B, ctx)
            last = y_ctx[:, -1:].detach()  # (B,1)
            centered = y_ctx - last
            emb = self.id_emb(ids)
            out = self.fc(centered) + self.id_proj(emb)
            out = self.drop(out).reshape(-1, h, out_dim)
            if out_dim == 1:
                return out.squeeze(-1) + last
            return out + last.unsqueeze(1)

    model = _NLinearGlobal()

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

    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_nlinear_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    id_emb_dim: int = 8,
    dropout: float = 0.0,
    quantiles: Any = (),
) -> Any:
    """
    Torch NLinear-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_nlinear_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_dlinear_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    ma_window: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    DLinear-style global/panel model (moving-average decomposition + linear heads, lite).
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)

    w = int(ma_window)
    if w <= 0:
        raise ValueError("ma_window must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)

    forecast_dim = int(h * out_dim)

    class _DLinearGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.seasonal_fc = nn.Linear(int(ctx), int(forecast_dim))
            self.trend_fc = nn.Linear(int(ctx), int(forecast_dim))
            self.id_proj = nn.Linear(int(id_emb_dim), int(forecast_dim), bias=False)
            self.drop = nn.Dropout(p=float(drop)) if float(drop) > 0.0 else nn.Identity()

        def forward(self, xb: Any, ids: Any) -> Any:
            y_ctx = xb[:, :ctx, 0]  # (B, ctx)
            pad = int(w) // 2
            y_in = y_ctx.unsqueeze(1)
            y_pad = F.pad(y_in, (pad, pad), mode="replicate")
            trend = F.avg_pool1d(y_pad, kernel_size=int(w), stride=1).squeeze(1)
            seasonal = y_ctx - trend

            emb = self.id_emb(ids)
            out = self.seasonal_fc(seasonal) + self.trend_fc(trend) + self.id_proj(emb)
            out = self.drop(out).reshape(-1, h, out_dim)
            return out.squeeze(-1) if out_dim == 1 else out

    model = _DLinearGlobal()

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

    loss_fn_override = None if not qs else _make_pinball_loss(qs)
    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=loss_fn_override,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_dlinear_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    ma_window: int = 7,
    id_emb_dim: int = 8,
    dropout: float = 0.0,
    quantiles: Any = (),
) -> Any:
    """
    Torch DLinear-style global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_dlinear_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            ma_window=int(ma_window),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_deepar_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    hidden_size: int,
    num_layers: int,
    dropout: float,
    id_emb_dim: int,
    quantiles: Any,
) -> pd.DataFrame:
    """
    DeepAR-style global/panel Gaussian RNN (direct multi-horizon, lite).

    Trains with Gaussian NLL on the multi-horizon target. If `quantiles` is set,
    outputs `yhat_pXX` columns derived from the Normal distribution.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    input_dim = int(x_train.shape[2])

    hidden = int(hidden_size)
    if hidden <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    layers = int(num_layers)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)
    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)
    rnn_drop = drop if layers > 1 else 0.0

    class _DeepARGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), int(hidden))
            self.rnn = _make_manual_gru(
                input_size=int(hidden),
                hidden_size=int(hidden),
                num_layers=int(layers),
                dropout=float(rnn_drop),
                bidirectional=False,
            )
            self.head = nn.Linear(int(hidden), 2)  # mu, raw_sigma

        def forward(self, xb: Any, ids: Any) -> Any:
            emb = self.id_emb(ids)
            emb_t = emb.unsqueeze(1).expand(-1, xb.shape[1], -1)
            x2 = torch.cat([xb, emb_t], dim=-1)
            z = self.in_proj(x2)
            out, _hn = self.rnn(z)
            params = self.head(out[:, -h:, :])  # (B,h,2)
            return params

    def _gaussian_nll(params: Any, yb: Any) -> Any:
        mu = params[:, :, 0]
        sigma = F.softplus(params[:, :, 1]) + 1e-3
        z = (yb - mu) / sigma
        const = 0.5 * math.log(2.0 * math.pi)
        return const + torch.log(sigma) + 0.5 * (z**2)

    model = _DeepARGlobal()

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

    model = _train_loop_global(
        model,
        x_train,
        ids_train,
        y_train,
        cfg=cfg,
        device=str(device),
        loss_fn_override=_gaussian_nll,
    )

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        params_scaled = model(x_pred_tensor, idp).detach().cpu().numpy()

    if params_scaled.shape != (int(len(pred_uids)), h, 2):
        raise ValueError(
            f"Expected DeepAR params shape ({len(pred_uids)}, {h}, 2), got {params_scaled.shape}"
        )

    mu_scaled = params_scaled[:, :, 0]
    sigma_scaled = np.log1p(np.exp(params_scaled[:, :, 1])) + 1e-3  # softplus

    if qs:
        nd = NormalDist()
        z = np.asarray([float(nd.inv_cdf(float(q))) for q in qs], dtype=float)  # (K,)
        yhat_scaled = mu_scaled[:, :, None] + sigma_scaled[:, :, None] * z.reshape(1, 1, -1)
    else:
        yhat_scaled = mu_scaled

    return _make_pred_df_from_scaled(
        yhat_scaled=yhat_scaled,
        pred_uids=pred_uids,
        pred_ds_list=pred_ds_list,
        pred_mean=pred_mean,
        pred_std=pred_std,
        horizon=h,
        qs=qs,
    )


def torch_deepar_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    hidden_size: int = 64,
    num_layers: int = 1,
    id_emb_dim: int = 8,
    dropout: float = 0.0,
    quantiles: Any = (),
) -> Any:
    """
    Torch DeepAR-style global/panel forecaster (Gaussian NLL, lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_deepar_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            quantiles=quantiles,
        )

    return _f


def _predict_torch_seq2seq_global(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    context_length: int,
    x_cols: Any,
    static_cols: Any,
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
    checkpoint_dir: str,
    save_best_checkpoint: bool,
    save_last_checkpoint: bool,
    resume_checkpoint_path: str,
    resume_checkpoint_strict: bool,
    device: str,
    # model params
    cell: str,
    attention: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    id_emb_dim: int,
    teacher_forcing: float,
    teacher_forcing_final: float | None,
    quantiles: Any,
) -> pd.DataFrame:
    """
    Encoder-decoder Seq2Seq global/panel model (lite).

    - Encoder: context window
    - Decoder: step-wise generation over horizon, using known future covariates/time features
    - Optional Bahdanau attention over encoder states
    """
    torch = _require_torch()
    nn = torch.nn

    x_cols_tup = _normalize_x_cols(x_cols)
    static_cols_tup = _normalize_static_cols(static_cols)
    qs = _normalize_quantiles(quantiles)
    out_dim = int(len(qs)) if qs else 1
    q_point = _pick_point_quantile(qs)
    point_idx = 0 if not qs else int(qs.index(q_point))

    (
        x_train,
        ids_train,
        y_train,
        x_pred,
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
        static_cols=static_cols_tup,
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        add_time_features=bool(add_time_features),
    )

    h = int(horizon)
    ctx = int(context_length)
    input_dim = int(x_train.shape[2])

    cell_s = str(cell).lower().strip()
    if cell_s not in {"lstm", "gru"}:
        raise ValueError("cell must be one of: lstm, gru")
    attn_s = str(attention).lower().strip()
    if attn_s not in {"none", "bahdanau"}:
        raise ValueError("attention must be one of: none, bahdanau")

    hidden = int(hidden_size)
    layers = int(num_layers)
    if hidden <= 0:
        raise ValueError(_HIDDEN_SIZE_MIN_MSG)
    if layers <= 0:
        raise ValueError(_NUM_LAYERS_MIN_MSG)

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError(_DROPOUT_RANGE_MSG)
    rnn_drop = drop if layers > 1 else 0.0

    tf0 = float(teacher_forcing)
    if not (0.0 <= tf0 <= 1.0):
        raise ValueError("teacher_forcing must be in [0,1]")
    tf1 = None if teacher_forcing_final is None else float(teacher_forcing_final)
    if tf1 is not None and not (0.0 <= tf1 <= 1.0):
        raise ValueError("teacher_forcing_final must be in [0,1]")

    class _Seq2SeqGlobal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(int(n_total_series), int(id_emb_dim))
            proj_dim = int(hidden)
            self.in_proj = nn.Linear(int(input_dim + id_emb_dim), proj_dim)

            if cell_s == "lstm":
                self.enc = _make_manual_lstm(
                    input_size=int(proj_dim),
                    hidden_size=int(hidden),
                    num_layers=int(layers),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
                self.dec = _make_manual_lstm(
                    input_size=int(proj_dim),
                    hidden_size=int(hidden),
                    num_layers=int(layers),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
            else:
                self.enc = _make_manual_gru(
                    input_size=int(proj_dim),
                    hidden_size=int(hidden),
                    num_layers=int(layers),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
                self.dec = _make_manual_gru(
                    input_size=int(proj_dim),
                    hidden_size=int(hidden),
                    num_layers=int(layers),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )

            if attn_s == "bahdanau":
                self.attn_enc = nn.Linear(hidden, hidden, bias=False)
                self.attn_dec = nn.Linear(hidden, hidden, bias=False)
                self.attn_v = nn.Linear(hidden, 1, bias=False)
                self.attn_out = nn.Linear(2 * hidden, hidden)
            else:
                self.attn_enc = None
                self.attn_dec = None
                self.attn_v = None
                self.attn_out = None

            self.drop = nn.Dropout(p=drop)
            self.head = nn.Linear(hidden, out_dim)

        def _point(self, out_step: Any) -> Any:
            if out_dim == 1:
                return out_step[:, 0]
            return out_step[:, int(point_idx)]

        def _apply_attention(self, enc_out: Any, dec_h: Any) -> Any:
            if self.attn_enc is None:
                return dec_h
            e = torch.tanh(self.attn_enc(enc_out) + self.attn_dec(dec_h).unsqueeze(1))
            a = torch.softmax(self.attn_v(e), dim=1)  # (B, ctx, 1)
            ctx_vec = torch.sum(a * enc_out, dim=1)  # (B, hidden)
            fused = torch.cat([dec_h, ctx_vec], dim=1)
            return torch.tanh(self.attn_out(fused))

        def forward(
            self,
            xb: Any,
            ids: Any,
            *,
            y_true: Any | None,
            teacher_forcing_ratio: float,
        ) -> Any:
            B = int(xb.shape[0])

            emb = self.id_emb(ids)
            emb_ctx = emb.unsqueeze(1).expand(-1, int(ctx), -1)
            enc_in = torch.cat([xb[:, : int(ctx), :], emb_ctx], dim=-1)
            enc_z = self.in_proj(enc_in)

            enc_out, enc_state = self.enc(enc_z)

            # Decoder state init from encoder final state
            dec_state = enc_state

            # Start token uses last observed y in context
            y_prev = xb[:, int(ctx - 1), 0]

            preds: list[Any] = []
            tf = float(teacher_forcing_ratio)
            for t in range(int(h)):
                fut_known = xb[:, int(ctx + t), 1:]
                dec_step = torch.cat([y_prev.unsqueeze(1), fut_known], dim=1)
                emb_t = emb  # (B, E)
                dec_in = torch.cat([dec_step, emb_t], dim=1).unsqueeze(1)
                dec_z = self.in_proj(dec_in)

                dec_out, dec_state = self.dec(dec_z, dec_state)
                dec_h = dec_out[:, 0, :]
                dec_h = self._apply_attention(enc_out, dec_h)
                dec_h = self.drop(dec_h)

                out_step = self.head(dec_h)  # (B, out_dim)
                preds.append(out_step)

                if t == int(h - 1):
                    break

                y_pred_next = self._point(out_step)
                if y_true is not None and tf > 0.0:
                    coin = torch.rand((B,), device=xb.device) < tf
                    y_prev = torch.where(coin, y_true[:, int(t)], y_pred_next)
                else:
                    y_prev = y_pred_next

            pred = torch.stack(preds, dim=1)  # (B, h, out_dim)
            return pred.squeeze(-1) if out_dim == 1 else pred

    model = _Seq2SeqGlobal()

    def _train_seq2seq(model_in: Any) -> Any:
        if int(epochs) <= 0:
            raise ValueError("epochs must be >= 1")
        if float(lr) <= 0.0:
            raise ValueError("lr must be > 0")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be >= 1")
        if int(patience) <= 0:
            raise ValueError("patience must be >= 1")
        if float(val_split) < 0.0 or float(val_split) >= 0.5:
            raise ValueError("val_split must be in [0, 0.5)")
        if float(grad_clip_norm) < 0.0:
            raise ValueError("grad_clip_norm must be >= 0")
        if int(swa_start_epoch) < -1:
            raise ValueError(_SWA_START_EPOCH_MIN_MSG)
        if int(swa_start_epoch) > int(epochs):
            raise ValueError(_SWA_START_EPOCH_MAX_EPOCHS_MSG)
        if float(ema_decay) > 0.0 and int(swa_start_epoch) >= 0:
            raise ValueError(_EMA_SWA_CONFLICT_MSG)
        if int(lookahead_steps) < 0:
            raise ValueError(_LOOKAHEAD_STEPS_MIN_MSG)
        if not (0.0 < float(lookahead_alpha) <= 1.0):
            raise ValueError(_LOOKAHEAD_ALPHA_RANGE_MSG)
        if float(sam_rho) < 0.0:
            raise ValueError(_SAM_RHO_MIN_MSG)
        if float(sam_rho) > 0.0 and int(grad_accum_steps) != 1:
            raise ValueError(_SAM_REQUIRES_SINGLE_ACCUM_MSG)
        if float(sam_rho) > 0.0 and bool(amp):
            raise ValueError(_SAM_REQUIRES_AMP_DISABLED_MSG)
        if float(horizon_loss_decay) <= 0.0:
            raise ValueError(_HORIZON_LOSS_DECAY_POSITIVE_MSG)
        if not (0.0 <= float(input_dropout) < 1.0):
            raise ValueError(_INPUT_DROPOUT_RANGE_MSG)
        if not (0.0 <= float(temporal_dropout) < 1.0):
            raise ValueError(_TEMPORAL_DROPOUT_RANGE_MSG)
        if float(grad_noise_std) < 0.0:
            raise ValueError(_GRAD_NOISE_STD_MIN_MSG)
        if str(gc_mode).lower().strip() not in {"off", "all", "conv_only"}:
            raise ValueError(_GC_MODE_OPTIONS_MSG)
        if float(agc_clip_factor) < 0.0:
            raise ValueError(_AGC_CLIP_FACTOR_MIN_MSG)
        if float(agc_eps) <= 0.0:
            raise ValueError(_AGC_EPS_POSITIVE_MSG)

        torch.manual_seed(int(seed))

        dev = torch.device(str(device))
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available")

        model_in = model_in.to(dev)

        x_tensor = torch.tensor(x_train, dtype=torch.float32, device=dev)
        ids_t = torch.tensor(ids_train, dtype=torch.long, device=dev)
        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=dev)

        n = int(x_tensor.shape[0])
        val_n = 0
        if float(val_split) > 0.0 and n >= 5:
            val_n = max(1, int(round(float(val_split) * n)))
            val_n = min(val_n, n - 1)

        if val_n > 0:
            train_end = n - val_n
            x_tr, ids_tr, y_tr = x_tensor[:train_end], ids_t[:train_end], y_tensor[:train_end]
            x_va, ids_va, y_va = x_tensor[train_end:], ids_t[train_end:], y_tensor[train_end:]
        else:
            x_tr, ids_tr, y_tr = x_tensor, ids_t, y_tensor
            x_va, ids_va, y_va = None, None, None

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tr, ids_tr, y_tr),
            batch_size=int(batch_size),
            shuffle=True,
        )
        val_loader = (
            None
            if x_va is None
            else torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x_va, ids_va, y_va),
                batch_size=int(batch_size),
                shuffle=False,
            )
        )

        opt_name = str(optimizer).lower().strip()
        if opt_name in {"adam", ""}:
            opt = torch.optim.Adam(
                model_in.parameters(), lr=float(lr), weight_decay=float(weight_decay)
            )
        elif opt_name == "adamw":
            opt = torch.optim.AdamW(
                model_in.parameters(), lr=float(lr), weight_decay=float(weight_decay)
            )
        elif opt_name == "sgd":
            opt = torch.optim.SGD(
                model_in.parameters(),
                lr=float(lr),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
            )
        else:
            raise ValueError("optimizer must be one of: adam, adamw, sgd")
        base_lrs = tuple(float(group["lr"]) for group in opt.param_groups)

        strategy_cfg = type(
            "_TorchSeq2SeqStrategyConfig",
            (),
            {
                "loss": str(loss),
                "ema_decay": float(ema_decay),
                "ema_warmup_epochs": int(ema_warmup_epochs),
                "swa_start_epoch": int(swa_start_epoch),
                "lookahead_steps": int(lookahead_steps),
                "lookahead_alpha": float(lookahead_alpha),
                "sam_rho": float(sam_rho),
                "sam_adaptive": bool(sam_adaptive),
                "horizon_loss_decay": float(horizon_loss_decay),
                "input_dropout": float(input_dropout),
                "temporal_dropout": float(temporal_dropout),
                "grad_noise_std": float(grad_noise_std),
                "gc_mode": str(gc_mode),
                "agc_clip_factor": float(agc_clip_factor),
                "agc_eps": float(agc_eps),
            },
        )()

        loss_fn = _make_torch_loss_fn(
            torch,
            nn,
            cfg=strategy_cfg,
            loss_fn_override=(None if not qs else _make_pinball_loss(qs)),
        )

        sched_name = str(scheduler).lower().strip()
        if sched_name in {"none", ""}:
            sched = None
        elif sched_name == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs))
        elif sched_name == "step":
            sched = torch.optim.lr_scheduler.StepLR(
                opt, step_size=int(scheduler_step_size), gamma=float(scheduler_gamma)
            )
        else:
            raise ValueError("scheduler must be one of: none, cosine, step")

        resume_state = _load_torch_training_state(
            torch,
            model_in,
            checkpoint_path=str(resume_checkpoint_path),
            strict=bool(resume_checkpoint_strict),
            optimizer=opt,
            scheduler=sched,
            scaler=None,
        )
        start_epoch = max(0, int(resume_state.start_epoch))
        base_lrs = resume_state.base_lrs or base_lrs
        best_loss = (
            float("inf")
            if resume_state.best_monitor is None
            else float(resume_state.best_monitor)
        )
        best_state: dict[str, Any] | None = (
            None
            if resume_state.best_state is None
            else _clone_torch_state_dict_to_cpu(resume_state.best_state)
        )
        clip_cfg = type(
            "_TorchSeq2SeqClipConfig",
            (),
            {
                "grad_clip_norm": float(grad_clip_norm),
                "grad_clip_mode": str(grad_clip_mode),
                "grad_clip_value": float(grad_clip_value),
                "grad_noise_std": float(grad_noise_std),
                "gc_mode": str(gc_mode),
                "agc_clip_factor": float(agc_clip_factor),
                "agc_eps": float(agc_eps),
            },
        )()
        ema_model = _make_torch_ema_model(model_in, cfg=strategy_cfg)
        ema_active = False
        if ema_model is not None:
            if resume_state.ema_state is not None:
                ema_model.load_state_dict(resume_state.ema_state)
                ema_active = True
            elif int(start_epoch) > int(ema_warmup_epochs):
                ema_model.load_state_dict(model_in.state_dict())
                ema_active = True
        swa_model = _make_torch_swa_model(model_in, cfg=strategy_cfg)
        swa_n_averaged = int(resume_state.swa_n_averaged)
        if swa_model is not None:
            if resume_state.swa_state is not None:
                swa_model.load_state_dict(resume_state.swa_state)
                swa_n_averaged = max(1, int(resume_state.swa_n_averaged))
            elif int(start_epoch) > int(swa_start_epoch):
                swa_model.load_state_dict(model_in.state_dict())
                swa_n_averaged = 1
        lookahead_model = _make_torch_lookahead_model(model_in, cfg=strategy_cfg)
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
            else _snapshot_torch_training_state(
                optimizer=opt,
                scheduler=sched,
                scaler=None,
                best_state=best_state,
                best_monitor=float(best_loss),
                bad_epochs=int(bad_epochs),
                best_epoch=int(best_epoch),
                base_lrs=base_lrs,
                ema_state=(
                    None if ema_model is None or not ema_active else ema_model.state_dict()
                ),
                swa_state=(
                    None
                    if swa_model is None or int(swa_n_averaged) <= 0
                    else swa_model.state_dict()
                ),
                swa_n_averaged=int(swa_n_averaged),
                lookahead_state=(
                    None if lookahead_model is None else lookahead_model.state_dict()
                ),
                lookahead_step=int(lookahead_step),
                model_state=_maybe_torch_model_state_for_checkpoint(
                    model=model_in,
                    cfg=strategy_cfg,
                    ema_model=ema_model,
                    ema_active=ema_active,
                    swa_model=swa_model,
                    swa_n_averaged=int(swa_n_averaged),
                    lookahead_model=lookahead_model,
                    lookahead_step=int(lookahead_step),
                ),
            )
        )
        last_extra_payload = (
            None
            if last_monitor is None
            else _snapshot_torch_training_state(
                optimizer=opt,
                scheduler=sched,
                scaler=None,
                best_state=best_state,
                best_monitor=float(best_loss),
                bad_epochs=int(bad_epochs),
                best_epoch=int(best_epoch),
                base_lrs=base_lrs,
                ema_state=(
                    None if ema_model is None or not ema_active else ema_model.state_dict()
                ),
                swa_state=(
                    None
                    if swa_model is None or int(swa_n_averaged) <= 0
                    else swa_model.state_dict()
                ),
                swa_n_averaged=int(swa_n_averaged),
                lookahead_state=(
                    None if lookahead_model is None else lookahead_model.state_dict()
                ),
                lookahead_step=int(lookahead_step),
                model_state=_maybe_torch_model_state_for_checkpoint(
                    model=model_in,
                    cfg=strategy_cfg,
                    ema_model=ema_model,
                    ema_active=ema_active,
                    swa_model=swa_model,
                    swa_n_averaged=int(swa_n_averaged),
                    lookahead_model=lookahead_model,
                    lookahead_step=int(lookahead_step),
                ),
            )
        )
        sam_active = _torch_sam_active(cfg=strategy_cfg)

        for ep in range(start_epoch, int(epochs)):
            if tf1 is None or int(epochs) <= 1:
                tf_ratio = tf0
            else:
                frac = float(ep) / float(int(epochs) - 1)
                tf_ratio = tf0 + (float(tf1) - tf0) * frac

            model_in.train()
            total = 0.0
            count = 0
            for xb, idb, yb in train_loader:
                opt.zero_grad(set_to_none=True)
                xb_train = _apply_torch_train_input_dropout(
                    torch,
                    xb,
                    cfg=strategy_cfg,
                )
                xb_train = _apply_torch_train_temporal_dropout(
                    torch,
                    xb_train,
                    cfg=strategy_cfg,
                )
                pred = model_in(
                    xb_train,
                    idb,
                    y_true=yb,
                    teacher_forcing_ratio=float(tf_ratio),
                )
                loss_v = loss_fn(pred, yb)
                loss_v.backward()
                if sam_active:
                    perturbations = _apply_torch_sam_perturbation(
                        torch,
                        model=model_in,
                        cfg=strategy_cfg,
                    )
                    if perturbations:
                        opt.zero_grad(set_to_none=True)
                        pred_sam = model_in(
                            xb_train,
                            idb,
                            y_true=yb,
                            teacher_forcing_ratio=float(tf_ratio),
                        )
                        loss_second = loss_fn(pred_sam, yb)
                        loss_second.backward()
                        _restore_torch_sam_perturbation(
                            torch,
                            perturbations=perturbations,
                        )
                    _apply_torch_gradient_clipping(torch, model_in, cfg=clip_cfg)
                    opt.step()
                else:
                    _apply_torch_gradient_clipping(torch, model_in, cfg=clip_cfg)
                    opt.step()
                if ema_model is not None and _torch_ema_active_for_epoch(
                    cfg=strategy_cfg,
                    epoch_idx=int(ep),
                ):
                    if not ema_active:
                        ema_model.load_state_dict(model_in.state_dict())
                        ema_active = True
                    else:
                        _update_torch_ema_model(
                            torch,
                            ema_model=ema_model,
                            model=model_in,
                            cfg=strategy_cfg,
                        )
                if lookahead_model is not None:
                    lookahead_step = _update_torch_lookahead_model(
                        torch,
                        lookahead_model=lookahead_model,
                        model=model_in,
                        cfg=strategy_cfg,
                        lookahead_step=int(lookahead_step),
                    )
                if swa_model is not None and _torch_swa_active_for_epoch(
                    cfg=strategy_cfg,
                    epoch_idx=int(ep),
                ):
                    swa_n_averaged = _update_torch_swa_model(
                        torch,
                        swa_model=swa_model,
                        model=model_in,
                        n_averaged=int(swa_n_averaged),
                    )

                total += float(loss_v.detach().cpu().item()) * int(xb.shape[0])
                count += int(xb.shape[0])

            train_loss = total / max(1, count)

            eval_model = _select_torch_deploy_model(
                model=model_in,
                cfg=strategy_cfg,
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
                    for xb, idb, yb in val_loader:
                        pred = eval_model(xb, idb, y_true=None, teacher_forcing_ratio=0.0)
                        v_loss = loss_fn(pred, yb)
                        v_total += float(v_loss.detach().cpu().item()) * int(xb.shape[0])
                        v_count += int(xb.shape[0])
                monitor = v_total / max(1, v_count)
            else:
                monitor = train_loss
            last_monitor = float(monitor)
            last_epoch = int(ep) + 1

            stop_training = False
            if float(monitor) + 1e-12 < best_loss:
                best_loss = float(monitor)
                bad_epochs = 0
                best_epoch = int(ep) + 1
                if bool(restore_best) or bool(save_best_checkpoint):
                    best_state = _clone_torch_state_dict_to_cpu(eval_model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= int(patience):
                    stop_training = True

            if sched is not None and not stop_training:
                sched.step()
            if best_state is not None:
                best_extra_payload = _snapshot_torch_training_state(
                    optimizer=opt,
                    scheduler=sched,
                    scaler=None,
                    best_state=best_state,
                    best_monitor=float(best_loss),
                    bad_epochs=int(bad_epochs),
                    best_epoch=int(best_epoch),
                    base_lrs=base_lrs,
                    ema_state=(
                        None if ema_model is None or not ema_active else ema_model.state_dict()
                    ),
                    swa_state=(
                        None
                        if swa_model is None or int(swa_n_averaged) <= 0
                        else swa_model.state_dict()
                    ),
                    swa_n_averaged=int(swa_n_averaged),
                    lookahead_state=(
                        None if lookahead_model is None else lookahead_model.state_dict()
                    ),
                    lookahead_step=int(lookahead_step),
                    model_state=_maybe_torch_model_state_for_checkpoint(
                        model=model_in,
                        cfg=strategy_cfg,
                        ema_model=ema_model,
                        ema_active=ema_active,
                        swa_model=swa_model,
                        swa_n_averaged=int(swa_n_averaged),
                        lookahead_model=lookahead_model,
                        lookahead_step=int(lookahead_step),
                    ),
                )
            last_extra_payload = _snapshot_torch_training_state(
                optimizer=opt,
                scheduler=sched,
                scaler=None,
                best_state=best_state,
                best_monitor=float(best_loss),
                bad_epochs=int(bad_epochs),
                best_epoch=int(best_epoch),
                base_lrs=base_lrs,
                ema_state=(
                    None if ema_model is None or not ema_active else ema_model.state_dict()
                ),
                swa_state=(
                    None
                    if swa_model is None or int(swa_n_averaged) <= 0
                    else swa_model.state_dict()
                ),
                swa_n_averaged=int(swa_n_averaged),
                lookahead_state=(
                    None if lookahead_model is None else lookahead_model.state_dict()
                ),
                lookahead_step=int(lookahead_step),
                model_state=_maybe_torch_model_state_for_checkpoint(
                    model=model_in,
                    cfg=strategy_cfg,
                    ema_model=ema_model,
                    ema_active=ema_active,
                    swa_model=swa_model,
                    swa_n_averaged=int(swa_n_averaged),
                    lookahead_model=lookahead_model,
                    lookahead_step=int(lookahead_step),
                ),
            )
            if stop_training:
                break

        if bool(save_best_checkpoint) and best_state is not None:
            _save_torch_checkpoint(
                torch,
                checkpoint_dir=str(checkpoint_dir),
                filename="best.pt",
                state_dict=best_state,
                monitor=float(best_loss),
                epoch=int(best_epoch),
                extra_payload=best_extra_payload,
            )
        if bool(save_last_checkpoint) and last_monitor is not None:
            deploy_model = _select_torch_deploy_model(
                model=model_in,
                cfg=strategy_cfg,
                ema_model=ema_model,
                ema_active=ema_active,
                swa_model=swa_model,
                swa_n_averaged=int(swa_n_averaged),
                lookahead_model=lookahead_model,
                lookahead_step=int(lookahead_step),
            )
            _save_torch_checkpoint(
                torch,
                checkpoint_dir=str(checkpoint_dir),
                filename="last.pt",
                state_dict=_clone_torch_state_dict_to_cpu(deploy_model.state_dict()),
                monitor=float(last_monitor),
                epoch=int(last_epoch),
                extra_payload=last_extra_payload,
            )

        if bool(restore_best) and best_state is not None:
            model_in.load_state_dict(best_state)

        model_in.eval()
        return model_in

    model = _train_seq2seq(model)

    dev = torch.device(str(device))
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32, device=dev)
    idp = torch.tensor(ids_pred, dtype=torch.long, device=dev)
    with torch.no_grad():
        yhat_scaled = (
            model(x_pred_tensor, idp, y_true=None, teacher_forcing_ratio=0.0).detach().cpu().numpy()
        )

    rows: list[dict[str, Any]] = []
    if qs:
        if yhat_scaled.shape != (int(len(pred_uids)), h, out_dim):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}, {out_dim}), got {yhat_scaled.shape}"
            )
        yhat_all = yhat_scaled * pred_std.reshape(-1, 1, 1) + pred_mean.reshape(-1, 1, 1)
        yhat_point = yhat_all[:, :, point_idx]

        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                row = {"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat_point[i, j])}
                for kk, qv in enumerate(qs):
                    row[_quantile_col(float(qv))] = float(yhat_all[i, j, kk])
                rows.append(row)
    else:
        if yhat_scaled.shape != (int(len(pred_uids)), h):
            raise ValueError(
                f"Expected prediction shape ({len(pred_uids)}, {h}), got {yhat_scaled.shape}"
            )
        yhat = yhat_scaled * pred_std.reshape(-1, 1) + pred_mean.reshape(-1, 1)
        for i, uid in enumerate(pred_uids):
            ds_f = pred_ds_list[i]
            for j in range(h):
                rows.append({"unique_id": uid, "ds": ds_f[j], "yhat": float(yhat[i, j])})

    return pd.DataFrame(rows)


def torch_seq2seq_global_forecaster(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
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
    device: str = "cpu",
    cell: str = "lstm",
    attention: str = "none",
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    id_emb_dim: int = 8,
    teacher_forcing: float = 0.5,
    teacher_forcing_final: float | None = None,
    quantiles: Any = (),
) -> Any:
    """
    Seq2Seq (encoder-decoder RNN) global/panel forecaster (lite).

    Returns a callable: (long_df, cutoff, horizon) -> prediction DataFrame (unique_id, ds, yhat).
    """

    def _f(long_df: pd.DataFrame, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _predict_torch_seq2seq_global(
            long_df,
            cutoff,
            int(horizon),
            context_length=int(context_length),
            x_cols=x_cols,
            static_cols=static_cols,
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
            device=str(device),
            cell=str(cell),
            attention=str(attention),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            id_emb_dim=int(id_emb_dim),
            teacher_forcing=float(teacher_forcing),
            teacher_forcing_final=teacher_forcing_final,
            quantiles=quantiles,
        )

    return _f
