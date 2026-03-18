from __future__ import annotations

from typing import Any

import numpy as np

from .torch_nn import (
    TorchTrainConfig,
    _apply_torch_gradient_clipping,
    _apply_torch_sam_perturbation,
    _apply_torch_train_input_dropout,
    _apply_torch_train_temporal_dropout,
    _apply_torch_warmup,
    _as_1d_float_array,
    _clamp_torch_optimizer_min_lr,
    _clone_torch_state_dict_to_cpu,
    _load_torch_training_state,
    _make_lagged_xy_multi,
    _make_manual_gru,
    _make_manual_lstm,
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
    _normalize_series,
    _require_torch,
    _restore_torch_sam_perturbation,
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


def _validate_seq2seq_training_config(cfg: TorchTrainConfig) -> None:
    _validate_torch_train_config(cfg)


def _validate_seq2seq_teacher_forcing(
    *,
    teacher_forcing_start: float,
    teacher_forcing_final: float,
) -> tuple[float, float]:
    tf0 = float(teacher_forcing_start)
    tf1 = float(teacher_forcing_final)
    if not (0.0 <= tf0 <= 1.0):
        raise ValueError("teacher_forcing_start must be in [0,1]")
    if not (0.0 <= tf1 <= 1.0):
        raise ValueError("teacher_forcing_final must be in [0,1]")
    return tf0, tf1


def _split_seq2seq_train_validation(
    torch: Any,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cfg: TorchTrainConfig,
) -> tuple[Any, Any]:
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(Y, dtype=torch.float32)

    n = int(x_tensor.shape[0])
    val_n = 0
    if float(cfg.val_split) > 0.0 and n >= 5:
        val_n = max(1, int(round(float(cfg.val_split) * n)))
        val_n = min(val_n, n - 1)

    if val_n > 0:
        train_end = n - val_n
        X_train, Y_train = x_tensor[:train_end], y_tensor[:train_end]
        x_val, y_val = x_tensor[train_end:], y_tensor[train_end:]
    else:
        X_train, Y_train = x_tensor, y_tensor
        x_val, y_val = None, None

    train_loader = _make_torch_dataloader(
        torch,
        torch.utils.data.TensorDataset(X_train, Y_train),
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
    return train_loader, val_loader


def _make_seq2seq_optimizer(torch: Any, model: Any, *, cfg: TorchTrainConfig) -> Any:
    opt_name = str(cfg.optimizer).lower().strip()
    if opt_name in {"adam", ""}:
        return torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
    if opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.lr),
            momentum=float(cfg.momentum),
            weight_decay=float(cfg.weight_decay),
        )
    raise ValueError("optimizer must be one of: adam, adamw, sgd")


def _make_seq2seq_scheduler(
    torch: Any,
    opt: Any,
    *,
    cfg: TorchTrainConfig,
    steps_per_epoch: int | None = None,
) -> tuple[Any, str]:
    return _make_torch_scheduler(torch, opt, cfg=cfg, steps_per_epoch=steps_per_epoch)


def _seq2seq_teacher_forcing_ratio(
    epoch: int,
    *,
    total_epochs: int,
    teacher_forcing_start: float,
    teacher_forcing_final: float,
) -> float:
    if int(total_epochs) == 1:
        return float(teacher_forcing_final)
    t = float(epoch) / float(int(total_epochs) - 1)
    return (1.0 - t) * float(teacher_forcing_start) + t * float(teacher_forcing_final)


def _validate_seq2seq_direct_config(
    *,
    horizon: int,
    lags: int,
    cell: str,
    attention: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    teacher_forcing: float,
    teacher_forcing_final: float | None,
) -> tuple[int, int, str, str, int, int, float, float, float]:
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")

    cell_s = str(cell).lower().strip()
    attn_s = str(attention).lower().strip()
    if cell_s not in {"lstm", "gru"}:
        raise ValueError("cell must be one of: lstm, gru")
    if attn_s not in {"none", "bahdanau"}:
        raise ValueError("attention must be one of: none, bahdanau")

    hidden = int(hidden_size)
    layers = int(num_layers)
    if hidden <= 0:
        raise ValueError("hidden_size must be >= 1")
    if layers <= 0:
        raise ValueError("num_layers must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    tf0, tf1 = _validate_seq2seq_teacher_forcing(
        teacher_forcing_start=float(teacher_forcing),
        teacher_forcing_final=(
            float(teacher_forcing) if teacher_forcing_final is None else float(teacher_forcing_final)
        ),
    )
    return h, lag_count, cell_s, attn_s, hidden, layers, drop, tf0, tf1


def _validate_lstnet_direct_config(
    *,
    horizon: int,
    lags: int,
    cnn_channels: int,
    kernel_size: int,
    rnn_hidden: int,
    skip: int,
    highway_window: int,
    dropout: float,
) -> tuple[int, int, int, int, int, int, int, float]:
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")

    channels = int(cnn_channels)
    kernel = int(kernel_size)
    hidden = int(rnn_hidden)
    if channels <= 0:
        raise ValueError("cnn_channels must be >= 1")
    if kernel <= 0:
        raise ValueError("kernel_size must be >= 1")
    if hidden <= 0:
        raise ValueError("rnn_hidden must be >= 1")

    skip_int = int(skip)
    if skip_int < 0:
        raise ValueError("skip must be >= 0")
    highway = int(highway_window)
    if highway < 0:
        raise ValueError("highway_window must be >= 0")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0,1)")

    return h, lag_count, channels, kernel, hidden, skip_int, highway, drop


def _build_torch_train_config(
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


def _prepare_univariate_direct_payload(
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
    x_seq = X.reshape(X.shape[0], int(lags), 1)
    return x_work, x_seq, Y, mean, std


def _predict_direct_torch_model(
    torch: Any,
    model: Any,
    history: np.ndarray,
    *,
    lag_count: int,
    device: str,
    predict_fn: Any | None = None,
) -> np.ndarray:
    feat = history[-int(lag_count) :].astype(float, copy=False).reshape(1, int(lag_count), 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        raw = model(feat_t) if predict_fn is None else predict_fn(model, feat_t)
        return raw.detach().cpu().numpy().reshape(-1)


def _maybe_denormalize_forecast(
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


def _train_seq2seq(
    model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cfg: TorchTrainConfig,
    device: str,
    teacher_forcing_start: float,
    teacher_forcing_final: float,
) -> Any:
    torch = _require_torch()
    nn = torch.nn

    _validate_seq2seq_training_config(cfg)
    tf0, tf1 = _validate_seq2seq_teacher_forcing(
        teacher_forcing_start=teacher_forcing_start,
        teacher_forcing_final=teacher_forcing_final,
    )

    torch.manual_seed(int(cfg.seed))

    dev = torch.device(str(device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available")
    amp_enabled, amp_dtype, scaler = _make_torch_amp_state(torch, cfg=cfg, dev=dev)

    model = model.to(dev)
    train_loader, val_loader = _split_seq2seq_train_validation(
        torch,
        X,
        Y,
        cfg=cfg,
    )
    opt = _make_seq2seq_optimizer(torch, model, cfg=cfg)
    base_lrs = tuple(float(group["lr"]) for group in opt.param_groups)

    loss_fn = _make_torch_loss_fn(torch, nn, cfg=cfg)
    accum_steps = int(cfg.grad_accum_steps)
    sched, sched_name = _make_seq2seq_scheduler(
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
                cfg=cfg,
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
                cfg=cfg,
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
    sam_active = _torch_sam_active(cfg=cfg)

    for epoch in range(start_epoch, int(cfg.epochs)):
        tf = _seq2seq_teacher_forcing_ratio(
            epoch,
            total_epochs=int(cfg.epochs),
            teacher_forcing_start=tf0,
            teacher_forcing_final=tf1,
        )
        _apply_torch_warmup(opt, cfg=cfg, epoch_idx=int(epoch), base_lrs=base_lrs)

        model.train()
        total = 0.0
        count = 0
        opt.zero_grad(set_to_none=True)
        num_batches = len(train_loader)
        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(dev, non_blocking=non_blocking)
            yb = yb.to(dev, non_blocking=non_blocking)
            xb_train = _apply_torch_train_input_dropout(torch, xb, cfg=cfg)
            xb_train = _apply_torch_train_temporal_dropout(torch, xb_train, cfg=cfg)
            with _make_torch_autocast_context(
                torch,
                enabled=bool(amp_enabled),
                dev=dev,
                dtype=amp_dtype,
            ):
                pred = model(xb_train, yb, teacher_forcing_ratio=float(tf))
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
                            pred = model(xb_train, yb, teacher_forcing_ratio=float(tf))
                            loss_second = loss_fn(pred, yb)
                        loss_second.backward()
                        _restore_torch_sam_perturbation(
                            torch,
                            perturbations=perturbations,
                        )
                    _apply_torch_gradient_clipping(torch, model, cfg=cfg)
                    opt.step()
                else:
                    if needs_unscale:
                        scaler.unscale_(opt)
                    _apply_torch_gradient_clipping(torch, model, cfg=cfg)
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
                        cfg=cfg,
                        lookahead_step=int(lookahead_step),
                    )
                if ema_model is not None and _torch_ema_active_for_epoch(
                    cfg=cfg,
                    epoch_idx=int(epoch),
                ):
                    if not ema_active:
                        ema_model.load_state_dict(model.state_dict())
                        ema_active = True
                    else:
                        _update_torch_ema_model(
                            torch,
                            ema_model=ema_model,
                            model=model,
                            cfg=cfg,
                        )
                if swa_model is not None and _torch_swa_active_for_epoch(
                    cfg=cfg,
                    epoch_idx=int(epoch),
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
                for xb, yb in val_loader:
                    xb = xb.to(dev, non_blocking=non_blocking)
                    yb = yb.to(dev, non_blocking=non_blocking)
                    with _make_torch_autocast_context(
                        torch,
                        enabled=bool(amp_enabled),
                        dev=dev,
                        dtype=amp_dtype,
                    ):
                        pred = eval_model(xb, yb, teacher_forcing_ratio=0.0)
                        v_loss = loss_fn(pred, yb)
                    v_total += float(v_loss.detach().cpu().item()) * int(xb.shape[0])
                    v_count += int(xb.shape[0])
            val_loss = v_total / max(1, v_count)

        monitor = _select_torch_monitor_value(
            cfg,
            train_loss=float(train_loss),
            val_loss=val_loss,
        )
        last_monitor = float(monitor)
        last_epoch = int(epoch) + 1

        stop_training = False
        if _torch_monitor_improved(value=float(monitor), best=float(best_monitor), cfg=cfg):
            best_monitor = float(monitor)
            bad_epochs = 0
            best_epoch = int(epoch) + 1
            if bool(cfg.restore_best) or bool(cfg.save_best_checkpoint):
                best_state = _clone_torch_state_dict_to_cpu(eval_model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg.patience) and int(epoch) + 1 >= int(cfg.min_epochs):
                stop_training = True

        if not stop_training and sched is not None and not _torch_scheduler_steps_per_batch(sched_name):
            if int(epoch) + 1 > int(cfg.warmup_epochs):
                if sched_name == "plateau":
                    sched.step(float(monitor))
                else:
                    sched.step()
                _clamp_torch_optimizer_min_lr(opt, cfg=cfg)
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
                    cfg=cfg,
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
                cfg=cfg,
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
            cfg=cfg,
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
    return model


def torch_seq2seq_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 48,
    cell: str = "lstm",
    attention: str = "none",
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    teacher_forcing: float = 0.5,
    teacher_forcing_final: float | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
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
) -> np.ndarray:
    """
    Seq2Seq RNN (LSTM/GRU) with optional Bahdanau attention.

    This is a true encoder-decoder that decodes the horizon autoregressively, with
    scheduled teacher forcing during training.
    """
    torch = _require_torch()
    nn = torch.nn

    h, lag_count, cell_s, attn_s, hidden, layers, drop, tf0, tf1 = _validate_seq2seq_direct_config(
        horizon=horizon,
        lags=lags,
        cell=cell,
        attention=attention,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        teacher_forcing=teacher_forcing,
        teacher_forcing_final=teacher_forcing_final,
    )
    x_work, x_seq, Y, mean, std = _prepare_univariate_direct_payload(
        train,
        h,
        lags=lag_count,
        normalize=bool(normalize),
    )

    class _Bahdanau(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.W1 = nn.Linear(hidden, hidden, bias=False)
            self.W2 = nn.Linear(hidden, hidden, bias=False)
            self.v = nn.Linear(hidden, 1, bias=False)

        def forward(self, enc_out: Any, dec_h: Any) -> tuple[Any, Any]:
            # enc_out: (B,T,H), dec_h: (B,H)
            e = self.v(torch.tanh(self.W1(enc_out) + self.W2(dec_h).unsqueeze(1))).squeeze(-1)
            w = torch.softmax(e, dim=1)
            ctx = torch.sum(enc_out * w.unsqueeze(-1), dim=1)
            return ctx, w

    class _Seq2Seq(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rnn_drop = drop if layers > 1 else 0.0
            if cell_s == "lstm":
                self.enc = _make_manual_lstm(
                    input_size=1,
                    hidden_size=int(hidden),
                    num_layers=int(layers),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
                self.dec = _make_manual_lstm(
                    input_size=1,
                    hidden_size=int(hidden),
                    num_layers=int(layers),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
            else:
                self.enc = _make_manual_gru(
                    input_size=1,
                    hidden_size=int(hidden),
                    num_layers=int(layers),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )
                self.dec = _make_manual_gru(
                    input_size=1,
                    hidden_size=int(hidden),
                    num_layers=int(layers),
                    dropout=float(rnn_drop),
                    bidirectional=False,
                )

            self.attn = _Bahdanau() if attn_s == "bahdanau" else None
            out_in = hidden if self.attn is None else 2 * hidden
            self.out = nn.Linear(out_in, 1)

        def forward(self, xb: Any, yb: Any | None, *, teacher_forcing_ratio: float) -> Any:
            # xb: (B,T,1), yb: (B,h)
            enc_out, enc_state = self.enc(xb)

            # Decoder initial input: last observed y
            dec_in = xb[:, -1:, :]  # (B,1,1)
            state = enc_state
            outs: list[Any] = []

            for t in range(h):
                dec_out, state = self.dec(dec_in, state)
                dec_h = dec_out[:, -1, :]  # (B,H)
                if self.attn is not None:
                    ctx, _w = self.attn(enc_out, dec_h)
                    feat = torch.cat([dec_h, ctx], dim=-1)
                else:
                    feat = dec_h
                pred = self.out(feat).reshape(-1, 1, 1)
                outs.append(pred.squeeze(1).squeeze(-1))

                if yb is not None and self.training and teacher_forcing_ratio > 0.0:
                    # Sample a mask per batch item
                    use_tf = (
                        torch.rand((pred.shape[0],), device=pred.device) < teacher_forcing_ratio
                    ).to(pred.dtype)
                    gt = yb[:, t].reshape(-1, 1, 1)
                    dec_in = use_tf.reshape(-1, 1, 1) * gt + (1.0 - use_tf.reshape(-1, 1, 1)) * pred
                else:
                    dec_in = pred

            return torch.stack(outs, dim=1)

    model = _Seq2Seq()

    cfg = _build_torch_train_config(
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
    model = _train_seq2seq(
        model,
        x_seq,
        Y,
        cfg=cfg,
        device=str(device),
        teacher_forcing_start=tf0,
        teacher_forcing_final=tf1,
    )

    yhat_t = _predict_direct_torch_model(
        torch,
        model,
        x_work,
        lag_count=lag_count,
        device=str(device),
        predict_fn=lambda seq2seq_model, feat_t: seq2seq_model(
            feat_t,
            None,
            teacher_forcing_ratio=0.0,
        ),
    )
    return _maybe_denormalize_forecast(yhat_t, normalize=bool(normalize), mean=mean, std=std)


def torch_lstnet_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    cnn_channels: int = 16,
    kernel_size: int = 6,
    rnn_hidden: int = 32,
    skip: int = 24,
    highway_window: int = 24,
    dropout: float = 0.2,
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
    LSTNet-style CNN + (skip) GRU + highway (lite).

    This is a simplified deterministic direct multi-horizon model.
    """
    from .torch_nn import _train_loop

    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    h, lag_count, c, k, hidden, skip_int, hw, drop = _validate_lstnet_direct_config(
        horizon=horizon,
        lags=lags,
        cnn_channels=cnn_channels,
        kernel_size=kernel_size,
        rnn_hidden=rnn_hidden,
        skip=skip,
        highway_window=highway_window,
        dropout=dropout,
    )
    x_work, x_seq, Y, mean, std = _prepare_univariate_direct_payload(
        train,
        h,
        lags=lag_count,
        normalize=bool(normalize),
    )

    class _LSTNetLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(1, c, kernel_size=k)
            self.drop = nn.Dropout(p=drop)
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
            self.proj = nn.Linear(hidden * (2 if self.skip_gru is not None else 1), h)

            self.hw = int(hw)
            self.highway = None if self.hw <= 0 else nn.Linear(self.hw, h)

        def forward(self, xb: Any) -> Any:
            # xb: (B,T,1)
            xch = xb.transpose(1, 2)  # (B,1,T)
            z = F.relu(self.conv(xch))  # (B,C,T')
            z = self.drop(z)
            zt = z.transpose(1, 2)  # (B,T',C)
            _, h_main = self.gru(zt)
            h_main = h_main[-1]  # (B,H)

            if self.skip_gru is not None:
                T = int(zt.shape[1])
                s = int(self.skip)
                # Take last floor(T/s)*s steps and reshape into (B, n, s, C) then sample every s.
                n = T // s
                if n <= 0:
                    h_skip = torch.zeros_like(h_main)
                else:
                    z2 = zt[:, -n * s :, :].reshape(zt.shape[0], n, s, zt.shape[2])
                    z3 = z2[:, :, -1, :]  # (B,n,C)
                    _, h_s = self.skip_gru(z3)
                    h_skip = h_s[-1]
                feat = torch.cat([h_main, h_skip], dim=-1)
            else:
                feat = h_main

            out = self.proj(feat)
            if self.highway is not None:
                last = xb[:, -self.hw :, 0]
                out = out + self.highway(last)
            return out

    model = _LSTNetLite()

    cfg = _build_torch_train_config(
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

    yhat_t = _predict_direct_torch_model(
        torch,
        model,
        x_work,
        lag_count=lag_count,
        device=str(device),
    )
    return _maybe_denormalize_forecast(yhat_t, normalize=bool(normalize), mean=mean, std=std)
