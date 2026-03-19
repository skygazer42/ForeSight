from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, get_type_hints

from .torch_nn import TorchTrainConfig

_TORCH_TRAIN_CONFIG_FIELD_NAMES = tuple(field.name for field in fields(TorchTrainConfig))
_TORCH_TRAIN_CONFIG_TYPE_HINTS = get_type_hints(TorchTrainConfig)


def _normalize_bool_like(value: Any) -> bool:
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes", "y", "on"}:
            return True
        if lower in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _coerce_torch_train_config_value(*, key: str, value: Any) -> Any:
    annotation = _TORCH_TRAIN_CONFIG_TYPE_HINTS.get(str(key))
    if annotation is bool:
        return _normalize_bool_like(value)
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        return str(value)
    return value


def coerce_torch_train_config_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _TORCH_TRAIN_CONFIG_FIELD_NAMES:
        if key in params:
            out[key] = _coerce_torch_train_config_value(key=key, value=params[key])
    return out


def _normalize_runtime_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_runtime_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_runtime_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_runtime_value(item) for key, item in value.items()}
    return value


def _copy_runtime_fields(source: Mapping[str, Any], field_names: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field_name in field_names:
        if field_name in source:
            out[str(field_name)] = _normalize_runtime_value(source[field_name])
    return out


def _copy_runtime_field_map(
    source: Mapping[str, Any],
    field_map: Mapping[str, str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for out_key, source_key in field_map.items():
        if source_key in source:
            out[str(out_key)] = _normalize_runtime_value(source[source_key])
    return out


def _compact_runtime_section(value: Any) -> Any:
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            compacted_item = _compact_runtime_section(item)
            if compacted_item in (None, {}, []):
                continue
            compacted[str(key)] = compacted_item
        return compacted
    if isinstance(value, list):
        compacted_list = [_compact_runtime_section(item) for item in value]
        return [item for item in compacted_list if item not in (None, {}, [])]
    return value


def _structured_torch_runtime_sections(training: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = _copy_runtime_field_map(
        training,
        {
            "name": "optimizer",
            "lr": "lr",
            "weight_decay": "weight_decay",
            "momentum": "momentum",
            "grad_accum_steps": "grad_accum_steps",
        },
    )
    grad_clip = _copy_runtime_field_map(
        training,
        {
            "norm": "grad_clip_norm",
            "mode": "grad_clip_mode",
            "value": "grad_clip_value",
        },
    )
    if grad_clip:
        optimizer["grad_clip"] = grad_clip

    scheduler = _copy_runtime_field_map(
        training,
        {
            "name": "scheduler",
            "step_size": "scheduler_step_size",
            "gamma": "scheduler_gamma",
            "restart_period": "scheduler_restart_period",
            "restart_mult": "scheduler_restart_mult",
            "pct_start": "scheduler_pct_start",
            "patience": "scheduler_patience",
            "plateau_factor": "scheduler_plateau_factor",
            "plateau_threshold": "scheduler_plateau_threshold",
            "warmup_epochs": "warmup_epochs",
            "min_lr": "min_lr",
        },
    )

    monitor = _copy_runtime_field_map(
        training,
        {
            "loss": "loss",
            "metric": "monitor",
            "mode": "monitor_mode",
            "min_delta": "min_delta",
            "val_split": "val_split",
            "patience": "patience",
            "min_epochs": "min_epochs",
            "restore_best": "restore_best",
        },
    )

    dataloader = _copy_runtime_field_map(
        training,
        {
            "batch_size": "batch_size",
            "num_workers": "num_workers",
            "pin_memory": "pin_memory",
            "persistent_workers": "persistent_workers",
        },
    )

    strategies = {
        "ema": _copy_runtime_field_map(
            training,
            {
                "decay": "ema_decay",
                "warmup_epochs": "ema_warmup_epochs",
            },
        ),
        "swa": _copy_runtime_field_map(
            training,
            {
                "start_epoch": "swa_start_epoch",
            },
        ),
        "lookahead": _copy_runtime_field_map(
            training,
            {
                "steps": "lookahead_steps",
                "alpha": "lookahead_alpha",
            },
        ),
        "sam": _copy_runtime_field_map(
            training,
            {
                "rho": "sam_rho",
                "adaptive": "sam_adaptive",
            },
        ),
        "regularization": _copy_runtime_field_map(
            training,
            {
                "horizon_loss_decay": "horizon_loss_decay",
                "input_dropout": "input_dropout",
                "temporal_dropout": "temporal_dropout",
                "grad_noise_std": "grad_noise_std",
                "gc_mode": "gc_mode",
                "agc_clip_factor": "agc_clip_factor",
                "agc_eps": "agc_eps",
            },
        ),
    }

    checkpoints = _copy_runtime_field_map(
        training,
        {
            "directory": "checkpoint_dir",
            "save_best": "save_best_checkpoint",
            "save_last": "save_last_checkpoint",
            "resume_path": "resume_checkpoint_path",
            "resume_strict": "resume_checkpoint_strict",
        },
    )

    tracking = {
        "tensorboard": _copy_runtime_field_map(
            training,
            {
                "log_dir": "tensorboard_log_dir",
                "run_name": "tensorboard_run_name",
                "flush_secs": "tensorboard_flush_secs",
            },
        ),
        "mlflow": _copy_runtime_field_map(
            training,
            {
                "tracking_uri": "mlflow_tracking_uri",
                "experiment_name": "mlflow_experiment_name",
                "run_name": "mlflow_run_name",
            },
        ),
        "wandb": _copy_runtime_field_map(
            training,
            {
                "project": "wandb_project",
                "entity": "wandb_entity",
                "run_name": "wandb_run_name",
                "directory": "wandb_dir",
                "mode": "wandb_mode",
            },
        ),
    }

    sections = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "monitor": monitor,
        "dataloader": dataloader,
        "strategies": strategies,
        "checkpoints": checkpoints,
        "tracking": tracking,
    }
    compacted = _compact_runtime_section(sections)
    return compacted if isinstance(compacted, dict) else {}


def summarize_model_runtime(
    *,
    model_key: str,
    model_params: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not str(model_key).startswith("torch-"):
        return None

    training = _copy_runtime_fields(model_params, _TORCH_TRAIN_CONFIG_FIELD_NAMES)

    runtime: dict[str, Any] = {
        "family": "torch",
        "training": training,
    }
    runtime.update(_structured_torch_runtime_sections(training))

    if "device" in model_params:
        runtime["device"] = str(model_params["device"])

    quantiles = model_params.get("quantiles")
    if quantiles not in (None, "", (), []):
        runtime["prediction"] = {
            "mode": "quantile",
            "quantiles": _normalize_runtime_value(quantiles),
        }
    else:
        runtime["prediction"] = {"mode": "point"}

    return runtime


__all__ = ["coerce_torch_train_config_params", "summarize_model_runtime"]
