import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models import torch_global
from foresight.models.registry import make_global_forecaster

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _make_long_df() -> tuple[pd.DataFrame, pd.Timestamp, int]:
    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=80, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.2), ("s2", 0.8)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.01 * np.arange(ds.size)
        y = base + 0.05 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.1).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    return pd.DataFrame(rows), ds[-6], 5


def _assert_global_validation_error(key: str, *, params: dict[str, object], message: str) -> None:
    long_df, cutoff, horizon = _make_long_df()
    forecaster = make_global_forecaster(key, **params, seed=0, patience=2, device="cpu")
    with pytest.raises(ValueError) as exc_info:
        forecaster(long_df, cutoff, horizon)
    assert str(exc_info.value) == message


BASE_PARAMS = {
    "torch-timexer-global": {
        "context_length": 32,
        "x_cols": ("promo",),
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dropout": 0.1,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
    },
    "torch-xformer-performer-global": {
        "context_length": 32,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-patchtst-global": {
        "context_length": 32,
        "patch_len": 8,
        "stride": 4,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-crossformer-global": {
        "context_length": 32,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "segment_len": 8,
        "stride": 8,
        "num_scales": 2,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-pyraformer-global": {
        "context_length": 32,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "segment_len": 8,
        "stride": 8,
        "num_levels": 3,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-tsmixer-global": {
        "context_length": 32,
        "d_model": 32,
        "num_blocks": 2,
        "token_mixing_hidden": 64,
        "channel_mixing_hidden": 64,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-fedformer-global": {
        "context_length": 32,
        "d_model": 32,
        "num_layers": 1,
        "ffn_dim": 64,
        "modes": 8,
        "ma_window": 7,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
        "quantiles": "0.1,0.5,0.9",
    },
    "torch-retnet-global": {
        "context_length": 32,
        "d_model": 24,
        "nhead": 4,
        "num_layers": 1,
        "ffn_dim": 48,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-rnn-gru-global": {
        "context_length": 32,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-nonstationary-transformer-global": {
        "context_length": 32,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
        "quantiles": "0.1,0.5,0.9",
    },
    "torch-itransformer-global": {
        "context_length": 32,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
        "quantiles": "0.1,0.5,0.9",
    },
    "torch-timesnet-global": {
        "context_length": 32,
        "d_model": 32,
        "num_layers": 1,
        "top_k": 2,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-tcn-global": {
        "context_length": 32,
        "channels": (32, 32),
        "kernel_size": 3,
        "dilation_base": 2,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-nbeats-global": {
        "context_length": 32,
        "num_blocks": 2,
        "num_layers": 2,
        "layer_width": 64,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-nhits-global": {
        "context_length": 32,
        "pool_sizes": (1, 2),
        "num_blocks": 2,
        "num_layers": 2,
        "layer_width": 64,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-tide-global": {
        "context_length": 32,
        "d_model": 32,
        "hidden_size": 64,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
        "quantiles": "0.1,0.5,0.9",
    },
    "torch-wavenet-global": {
        "context_length": 32,
        "channels": 16,
        "num_layers": 4,
        "kernel_size": 2,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-resnet1d-global": {
        "context_length": 32,
        "channels": 16,
        "num_blocks": 2,
        "kernel_size": 3,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-fnet-global": {
        "context_length": 32,
        "d_model": 32,
        "num_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
}

GLOBAL_VALIDATION_CASES = [
    (
        "timexer min epochs",
        "torch-timexer-global",
        {"min_epochs": 0},
        "min_epochs must be >= 1",
    ),
    (
        "timexer min epochs > epochs",
        "torch-timexer-global",
        {"epochs": 1, "min_epochs": 2},
        "min_epochs must be <= epochs",
    ),
    (
        "timexer warmup epochs",
        "torch-timexer-global",
        {"warmup_epochs": -1},
        "warmup_epochs must be >= 0",
    ),
    (
        "timexer warmup epochs > epochs",
        "torch-timexer-global",
        {"epochs": 1, "warmup_epochs": 2},
        "warmup_epochs must be <= epochs",
    ),
    (
        "timexer min lr",
        "torch-timexer-global",
        {"min_lr": -0.1},
        "min_lr must be >= 0",
    ),
    (
        "timexer amp dtype invalid",
        "torch-timexer-global",
        {"amp_dtype": "fp8"},
        "amp_dtype must be one of: auto, float16, bfloat16",
    ),
    (
        "timexer scheduler invalid",
        "torch-timexer-global",
        {"scheduler": "triangle"},
        "scheduler must be one of: none, cosine, step, plateau, onecycle, cosine_restarts",
    ),
    (
        "timexer restart period",
        "torch-timexer-global",
        {"scheduler": "cosine_restarts", "scheduler_restart_period": 0},
        "scheduler_restart_period must be >= 1",
    ),
    (
        "timexer restart mult",
        "torch-timexer-global",
        {"scheduler": "cosine_restarts", "scheduler_restart_mult": 0},
        "scheduler_restart_mult must be >= 1",
    ),
    (
        "timexer pct start",
        "torch-timexer-global",
        {"scheduler": "onecycle", "scheduler_pct_start": 1.0},
        "scheduler_pct_start must be in (0, 1)",
    ),
    (
        "timexer grad accum",
        "torch-timexer-global",
        {"grad_accum_steps": 0},
        "grad_accum_steps must be >= 1",
    ),
    (
        "timexer monitor invalid",
        "torch-timexer-global",
        {"monitor": "score"},
        "monitor must be one of: auto, train_loss, val_loss",
    ),
    (
        "timexer monitor val without split",
        "torch-timexer-global",
        {"monitor": "val_loss", "val_split": 0.0},
        "monitor='val_loss' requires val_split > 0",
    ),
    (
        "timexer monitor mode invalid",
        "torch-timexer-global",
        {"monitor_mode": "sideways"},
        "monitor_mode must be one of: min, max",
    ),
    (
        "timexer min delta",
        "torch-timexer-global",
        {"min_delta": -0.1},
        "min_delta must be >= 0",
    ),
    (
        "timexer num workers",
        "torch-timexer-global",
        {"num_workers": -1},
        "num_workers must be >= 0",
    ),
    (
        "timexer persistent workers",
        "torch-timexer-global",
        {"persistent_workers": True, "num_workers": 0},
        "persistent_workers requires num_workers >= 1",
    ),
    (
        "timexer plateau patience",
        "torch-timexer-global",
        {"scheduler": "plateau", "scheduler_patience": 0},
        "scheduler_patience must be >= 1",
    ),
    (
        "timexer grad clip mode",
        "torch-timexer-global",
        {"grad_clip_mode": "percentile"},
        "grad_clip_mode must be one of: norm, value",
    ),
    (
        "timexer grad clip value",
        "torch-timexer-global",
        {"grad_clip_value": -0.1},
        "grad_clip_value must be >= 0",
    ),
    (
        "timexer plateau factor",
        "torch-timexer-global",
        {"scheduler": "plateau", "scheduler_plateau_factor": 1.0},
        "scheduler_plateau_factor must be in (0, 1)",
    ),
    (
        "timexer plateau threshold",
        "torch-timexer-global",
        {"scheduler": "plateau", "scheduler_plateau_threshold": -0.1},
        "scheduler_plateau_threshold must be >= 0",
    ),
    (
        "timexer ema decay negative",
        "torch-timexer-global",
        {"ema_decay": -0.1},
        "ema_decay must be in [0, 1)",
    ),
    (
        "timexer ema decay one",
        "torch-timexer-global",
        {"ema_decay": 1.0},
        "ema_decay must be in [0, 1)",
    ),
    (
        "timexer ema warmup negative",
        "torch-timexer-global",
        {"ema_warmup_epochs": -1},
        "ema_warmup_epochs must be >= 0",
    ),
    (
        "timexer ema warmup > epochs",
        "torch-timexer-global",
        {"epochs": 1, "ema_warmup_epochs": 2},
        "ema_warmup_epochs must be <= epochs",
    ),
    (
        "timexer swa start < -1",
        "torch-timexer-global",
        {"swa_start_epoch": -2},
        "swa_start_epoch must be >= -1",
    ),
    (
        "timexer swa start > epochs",
        "torch-timexer-global",
        {"epochs": 1, "swa_start_epoch": 2},
        "swa_start_epoch must be <= epochs",
    ),
    (
        "timexer ema and swa conflict",
        "torch-timexer-global",
        {"ema_decay": 0.9, "swa_start_epoch": 0},
        "ema_decay and swa_start_epoch cannot both be enabled",
    ),
    (
        "timexer lookahead steps negative",
        "torch-timexer-global",
        {"lookahead_steps": -1},
        "lookahead_steps must be >= 0",
    ),
    (
        "timexer lookahead alpha zero",
        "torch-timexer-global",
        {"lookahead_steps": 1, "lookahead_alpha": 0.0},
        "lookahead_alpha must be in (0, 1]",
    ),
    (
        "timexer lookahead alpha too large",
        "torch-timexer-global",
        {"lookahead_steps": 1, "lookahead_alpha": 1.1},
        "lookahead_alpha must be in (0, 1]",
    ),
    (
        "timexer sam rho negative",
        "torch-timexer-global",
        {"sam_rho": -0.1},
        "sam_rho must be >= 0",
    ),
    (
        "timexer horizon loss decay negative",
        "torch-timexer-global",
        {"horizon_loss_decay": -0.1},
        "horizon_loss_decay must be > 0",
    ),
    (
        "timexer horizon loss decay zero",
        "torch-timexer-global",
        {"horizon_loss_decay": 0.0},
        "horizon_loss_decay must be > 0",
    ),
    (
        "timexer input dropout negative",
        "torch-timexer-global",
        {"input_dropout": -0.1},
        "input_dropout must be in [0, 1)",
    ),
    (
        "timexer input dropout one",
        "torch-timexer-global",
        {"input_dropout": 1.0},
        "input_dropout must be in [0, 1)",
    ),
    (
        "timexer temporal dropout negative",
        "torch-timexer-global",
        {"temporal_dropout": -0.1},
        "temporal_dropout must be in [0, 1)",
    ),
    (
        "timexer temporal dropout one",
        "torch-timexer-global",
        {"temporal_dropout": 1.0},
        "temporal_dropout must be in [0, 1)",
    ),
    (
        "timexer grad noise std negative",
        "torch-timexer-global",
        {"grad_noise_std": -0.1},
        "grad_noise_std must be >= 0",
    ),
    (
        "timexer gc mode invalid",
        "torch-timexer-global",
        {"gc_mode": "layerwise"},
        "gc_mode must be one of: off, all, conv_only",
    ),
    (
        "timexer agc clip factor negative",
        "torch-timexer-global",
        {"agc_clip_factor": -0.1},
        "agc_clip_factor must be >= 0",
    ),
    (
        "timexer agc eps zero",
        "torch-timexer-global",
        {"agc_eps": 0.0},
        "agc_eps must be > 0",
    ),
    (
        "timexer amp requires cuda",
        "torch-timexer-global",
        {"amp": True},
        "amp=True requires device='cuda'",
    ),
    (
        "timexer best checkpoint without dir",
        "torch-timexer-global",
        {"save_best_checkpoint": True},
        "checkpoint_dir is required when checkpoint saving is enabled",
    ),
    (
        "timexer last checkpoint without dir",
        "torch-timexer-global",
        {"save_last_checkpoint": True},
        "checkpoint_dir is required when checkpoint saving is enabled",
    ),
    (
        "timexer resume checkpoint missing",
        "torch-timexer-global",
        {"resume_checkpoint_path": "/tmp/foresight-missing-global-resume.pt"},
        "resume_checkpoint_path does not exist",
    ),
    ("timexer d_model", "torch-timexer-global", {"d_model": 0}, "d_model must be >= 1"),
    ("timexer nhead", "torch-timexer-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "timexer divisibility",
        "torch-timexer-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    ("timexer num_layers", "torch-timexer-global", {"num_layers": 0}, "num_layers must be >= 1"),
    ("timexer dropout", "torch-timexer-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("xformer d_model", "torch-xformer-performer-global", {"d_model": 0}, "d_model must be >= 1"),
    ("xformer nhead", "torch-xformer-performer-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "xformer divisibility",
        "torch-xformer-performer-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    (
        "xformer num_layers",
        "torch-xformer-performer-global",
        {"num_layers": 0},
        "num_layers must be >= 1",
    ),
    (
        "xformer dim_feedforward",
        "torch-xformer-performer-global",
        {"dim_feedforward": 0},
        "dim_feedforward must be >= 1",
    ),
    (
        "xformer dropout",
        "torch-xformer-performer-global",
        {"dropout": 1.0},
        "dropout must be in [0,1)",
    ),
    ("patchtst d_model", "torch-patchtst-global", {"d_model": 0}, "d_model must be >= 1"),
    ("patchtst nhead", "torch-patchtst-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "patchtst divisibility",
        "torch-patchtst-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    ("patchtst num_layers", "torch-patchtst-global", {"num_layers": 0}, "num_layers must be >= 1"),
    (
        "patchtst dim_feedforward",
        "torch-patchtst-global",
        {"dim_feedforward": 0},
        "dim_feedforward must be >= 1",
    ),
    ("patchtst dropout", "torch-patchtst-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    (
        "crossformer d_model",
        "torch-crossformer-global",
        {"d_model": 0},
        "d_model must be >= 1",
    ),
    ("crossformer nhead", "torch-crossformer-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "crossformer divisibility",
        "torch-crossformer-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    (
        "crossformer num_layers",
        "torch-crossformer-global",
        {"num_layers": 0},
        "num_layers must be >= 1",
    ),
    (
        "crossformer dim_feedforward",
        "torch-crossformer-global",
        {"dim_feedforward": 0},
        "dim_feedforward must be >= 1",
    ),
    (
        "crossformer dropout",
        "torch-crossformer-global",
        {"dropout": 1.0},
        "dropout must be in [0,1)",
    ),
    (
        "pyraformer d_model",
        "torch-pyraformer-global",
        {"d_model": 0},
        "d_model must be >= 1",
    ),
    ("pyraformer nhead", "torch-pyraformer-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "pyraformer divisibility",
        "torch-pyraformer-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    (
        "pyraformer num_layers",
        "torch-pyraformer-global",
        {"num_layers": 0},
        "num_layers must be >= 1",
    ),
    (
        "pyraformer dim_feedforward",
        "torch-pyraformer-global",
        {"dim_feedforward": 0},
        "dim_feedforward must be >= 1",
    ),
    ("pyraformer dropout", "torch-pyraformer-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("tsmixer d_model", "torch-tsmixer-global", {"d_model": 0}, "d_model must be >= 1"),
    ("tsmixer num_blocks", "torch-tsmixer-global", {"num_blocks": 0}, "num_blocks must be >= 1"),
    (
        "tsmixer token hidden",
        "torch-tsmixer-global",
        {"token_mixing_hidden": 0},
        "token_mixing_hidden must be >= 1",
    ),
    (
        "tsmixer channel hidden",
        "torch-tsmixer-global",
        {"channel_mixing_hidden": 0},
        "channel_mixing_hidden must be >= 1",
    ),
    ("tsmixer dropout", "torch-tsmixer-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("fedformer d_model", "torch-fedformer-global", {"d_model": 0}, "d_model must be >= 1"),
    (
        "fedformer num_layers",
        "torch-fedformer-global",
        {"num_layers": 0},
        "num_layers must be >= 1",
    ),
    ("fedformer ffn_dim", "torch-fedformer-global", {"ffn_dim": 0}, "ffn_dim must be >= 1"),
    ("fedformer modes", "torch-fedformer-global", {"modes": 0}, "modes must be >= 1"),
    ("fedformer ma_window", "torch-fedformer-global", {"ma_window": 0}, "ma_window must be >= 1"),
    ("fedformer dropout", "torch-fedformer-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("retnet d_model", "torch-retnet-global", {"d_model": 0}, "d_model must be >= 1"),
    ("retnet nhead", "torch-retnet-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "retnet divisibility",
        "torch-retnet-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    ("retnet num_layers", "torch-retnet-global", {"num_layers": 0}, "num_layers must be >= 1"),
    ("retnet ffn_dim", "torch-retnet-global", {"ffn_dim": 0}, "ffn_dim must be >= 1"),
    ("retnet dropout", "torch-retnet-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("patchtst stride", "torch-patchtst-global", {"stride": 0}, "stride must be >= 1"),
    ("crossformer stride", "torch-crossformer-global", {"stride": 0}, "stride must be >= 1"),
    ("pyraformer stride", "torch-pyraformer-global", {"stride": 0}, "stride must be >= 1"),
    ("rnn hidden_size", "torch-rnn-gru-global", {"hidden_size": 0}, "hidden_size must be >= 1"),
    ("rnn num_layers", "torch-rnn-gru-global", {"num_layers": 0}, "num_layers must be >= 1"),
    ("rnn dropout", "torch-rnn-gru-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    (
        "nonstationary d_model",
        "torch-nonstationary-transformer-global",
        {"d_model": 0},
        "d_model must be >= 1",
    ),
    (
        "nonstationary nhead",
        "torch-nonstationary-transformer-global",
        {"nhead": 0},
        "nhead must be >= 1",
    ),
    (
        "nonstationary divisibility",
        "torch-nonstationary-transformer-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    (
        "nonstationary num_layers",
        "torch-nonstationary-transformer-global",
        {"num_layers": 0},
        "num_layers must be >= 1",
    ),
    (
        "nonstationary dim_feedforward",
        "torch-nonstationary-transformer-global",
        {"dim_feedforward": 0},
        "dim_feedforward must be >= 1",
    ),
    (
        "nonstationary dropout",
        "torch-nonstationary-transformer-global",
        {"dropout": 1.0},
        "dropout must be in [0,1)",
    ),
    ("itransformer d_model", "torch-itransformer-global", {"d_model": 0}, "d_model must be >= 1"),
    ("itransformer nhead", "torch-itransformer-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "itransformer divisibility",
        "torch-itransformer-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    (
        "itransformer num_layers",
        "torch-itransformer-global",
        {"num_layers": 0},
        "num_layers must be >= 1",
    ),
    (
        "itransformer dim_feedforward",
        "torch-itransformer-global",
        {"dim_feedforward": 0},
        "dim_feedforward must be >= 1",
    ),
    (
        "itransformer dropout",
        "torch-itransformer-global",
        {"dropout": 1.0},
        "dropout must be in [0,1)",
    ),
    ("timesnet d_model", "torch-timesnet-global", {"d_model": 0}, "d_model must be >= 1"),
    ("timesnet num_layers", "torch-timesnet-global", {"num_layers": 0}, "num_layers must be >= 1"),
    ("timesnet dropout", "torch-timesnet-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("tcn kernel_size", "torch-tcn-global", {"kernel_size": 0}, "kernel_size must be >= 1"),
    ("tcn dropout", "torch-tcn-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("nbeats num_blocks", "torch-nbeats-global", {"num_blocks": 0}, "num_blocks must be >= 1"),
    ("nbeats num_layers", "torch-nbeats-global", {"num_layers": 0}, "num_layers must be >= 1"),
    ("nbeats dropout", "torch-nbeats-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("nhits num_blocks", "torch-nhits-global", {"num_blocks": 0}, "num_blocks must be >= 1"),
    ("nhits num_layers", "torch-nhits-global", {"num_layers": 0}, "num_layers must be >= 1"),
    ("nhits dropout", "torch-nhits-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("tide d_model", "torch-tide-global", {"d_model": 0}, "d_model must be >= 1"),
    ("tide hidden_size", "torch-tide-global", {"hidden_size": 0}, "hidden_size must be >= 1"),
    ("tide dropout", "torch-tide-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("wavenet channels", "torch-wavenet-global", {"channels": 0}, "channels must be >= 1"),
    ("wavenet num_layers", "torch-wavenet-global", {"num_layers": 0}, "num_layers must be >= 1"),
    ("wavenet kernel_size", "torch-wavenet-global", {"kernel_size": 0}, "kernel_size must be >= 1"),
    ("wavenet dropout", "torch-wavenet-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("resnet channels", "torch-resnet1d-global", {"channels": 0}, "channels must be >= 1"),
    ("resnet num_blocks", "torch-resnet1d-global", {"num_blocks": 0}, "num_blocks must be >= 1"),
    ("resnet kernel_size", "torch-resnet1d-global", {"kernel_size": 0}, "kernel_size must be >= 1"),
    ("resnet dropout", "torch-resnet1d-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("fnet d_model", "torch-fnet-global", {"d_model": 0}, "d_model must be >= 1"),
    ("fnet num_layers", "torch-fnet-global", {"num_layers": 0}, "num_layers must be >= 1"),
    (
        "fnet dim_feedforward",
        "torch-fnet-global",
        {"dim_feedforward": 0},
        "dim_feedforward must be >= 1",
    ),
    ("fnet dropout", "torch-fnet-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
]


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
@pytest.mark.parametrize(
    ("case_id", "key", "overrides", "message"),
    GLOBAL_VALIDATION_CASES,
    ids=[case[0] for case in GLOBAL_VALIDATION_CASES],
)
def test_torch_global_validation_messages(
    case_id: str,
    key: str,
    overrides: dict[str, object],
    message: str,
) -> None:
    params = dict(BASE_PARAMS[key])
    params.update(overrides)
    _assert_global_validation_error(key, params=params, message=message)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_cosine_restart_scheduler() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        scheduler="cosine_restarts",
        scheduler_restart_period=1,
        scheduler_restart_mult=1,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_value_clipping_and_plateau_controls() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        scheduler="plateau",
        scheduler_plateau_factor=0.5,
        scheduler_plateau_threshold=0.0,
        grad_clip_mode="value",
        grad_clip_value=0.1,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_ema_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        ema_decay=0.9,
        ema_warmup_epochs=0,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_swa_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        swa_start_epoch=0,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_lookahead_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        lookahead_steps=1,
        lookahead_alpha=0.5,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_sam_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        sam_rho=0.05,
        sam_adaptive=True,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_horizon_loss_decay_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        horizon_loss_decay=0.5,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_input_dropout_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        input_dropout=0.1,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_temporal_dropout_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        temporal_dropout=0.1,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_seq2seq_runtime_accepts_input_dropout_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    forecaster = make_global_forecaster(
        "torch-seq2seq-lstm-deep-global",
        context_length=32,
        x_cols=("promo",),
        epochs=2,
        val_split=0.1,
        batch_size=32,
        seed=0,
        patience=3,
        device="cpu",
        input_dropout=0.1,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_seq2seq_runtime_accepts_horizon_loss_decay_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    forecaster = make_global_forecaster(
        "torch-seq2seq-lstm-deep-global",
        context_length=32,
        x_cols=("promo",),
        epochs=2,
        val_split=0.1,
        batch_size=32,
        seed=0,
        patience=3,
        device="cpu",
        horizon_loss_decay=0.5,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_seq2seq_runtime_accepts_temporal_dropout_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    forecaster = make_global_forecaster(
        "torch-seq2seq-lstm-deep-global",
        context_length=32,
        x_cols=("promo",),
        epochs=2,
        val_split=0.1,
        batch_size=32,
        seed=0,
        patience=3,
        device="cpu",
        temporal_dropout=0.1,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_grad_noise_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        grad_noise_std=0.01,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_seq2seq_runtime_accepts_grad_noise_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    forecaster = make_global_forecaster(
        "torch-seq2seq-lstm-deep-global",
        context_length=32,
        x_cols=("promo",),
        epochs=2,
        val_split=0.1,
        batch_size=32,
        seed=0,
        patience=3,
        device="cpu",
        grad_noise_std=0.01,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_seq2seq_runtime_accepts_sam_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    forecaster = make_global_forecaster(
        "torch-seq2seq-lstm-deep-global",
        context_length=32,
        x_cols=("promo",),
        epochs=2,
        val_split=0.1,
        batch_size=32,
        seed=0,
        patience=3,
        device="cpu",
        sam_rho=0.05,
        sam_adaptive=True,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_seq2seq_runtime_accepts_plateau_scheduler() -> None:
    long_df, cutoff, horizon = _make_long_df()
    forecaster = make_global_forecaster(
        "torch-seq2seq-lstm-deep-global",
        context_length=32,
        x_cols=("promo",),
        epochs=2,
        val_split=0.1,
        batch_size=32,
        seed=0,
        patience=3,
        device="cpu",
        scheduler="plateau",
        monitor="val_loss",
        scheduler_patience=1,
        scheduler_plateau_factor=0.5,
        scheduler_plateau_threshold=0.0,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_agc_strategy() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        agc_clip_factor=0.01,
        agc_eps=1e-3,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_accepts_gc_mode() -> None:
    long_df, cutoff, horizon = _make_long_df()
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        gc_mode="all",
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_writes_checkpoint_files(tmp_path) -> None:
    long_df, cutoff, horizon = _make_long_df()
    checkpoint_dir = tmp_path / "global-checkpoints"
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        checkpoint_dir=str(checkpoint_dir),
        save_best_checkpoint=True,
        save_last_checkpoint=True,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon
    assert (checkpoint_dir / "best.pt").is_file()
    assert (checkpoint_dir / "last.pt").is_file()
    torch = torch_global._require_torch()
    try:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    except TypeError:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu")
    assert "optimizer_state" in last_payload
    assert "epoch" in last_payload
    assert "best_monitor" in last_payload
    assert "bad_epochs" in last_payload
    assert "best_epoch" in last_payload
    assert "best_state" in last_payload


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_swa_checkpoints_store_raw_and_swa_state(tmp_path) -> None:
    long_df, cutoff, horizon = _make_long_df()
    checkpoint_dir = tmp_path / "global-swa-checkpoints"
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        checkpoint_dir=str(checkpoint_dir),
        save_last_checkpoint=True,
        swa_start_epoch=0,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon
    torch = torch_global._require_torch()
    try:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    except TypeError:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu")
    assert "swa_state" in last_payload
    assert "model_state" in last_payload


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_global_runtime_lookahead_checkpoints_store_raw_and_slow_state(tmp_path) -> None:
    long_df, cutoff, horizon = _make_long_df()
    checkpoint_dir = tmp_path / "global-lookahead-checkpoints"
    params = dict(BASE_PARAMS["torch-timexer-global"])
    params["epochs"] = 2
    params["val_split"] = 0.1
    forecaster = make_global_forecaster(
        "torch-timexer-global",
        **params,
        seed=0,
        patience=3,
        device="cpu",
        checkpoint_dir=str(checkpoint_dir),
        save_last_checkpoint=True,
        lookahead_steps=3,
        lookahead_alpha=0.5,
    )
    pred = forecaster(long_df, cutoff, horizon)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 3 * horizon
    torch = torch_global._require_torch()
    try:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    except TypeError:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu")
    assert "lookahead_state" in last_payload
    assert "lookahead_step" in last_payload
    assert "model_state" in last_payload
