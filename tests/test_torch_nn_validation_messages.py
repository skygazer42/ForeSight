import importlib.util

import numpy as np
import pytest

from foresight.models import torch_nn
from foresight.models.registry import make_forecaster

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_torch_nn_exposes_shared_validation_messages() -> None:
    assert torch_nn._HIDDEN_SIZE_MIN_MSG == "hidden_size must be >= 1"
    assert torch_nn._NUM_LAYERS_MIN_MSG == "num_layers must be >= 1"
    assert torch_nn._DROPOUT_RANGE_MSG == "dropout must be in [0, 1)"
    assert torch_nn._HORIZON_MIN_MSG == "horizon must be >= 1"
    assert torch_nn._LAGS_MIN_MSG == "lags must be >= 1"
    assert torch_nn._NUM_BLOCKS_MIN_MSG == "num_blocks must be >= 1"
    assert torch_nn._D_MODEL_MIN_MSG == "d_model must be >= 1"
    assert torch_nn._NHEAD_MIN_MSG == "nhead must be >= 1"
    assert torch_nn._D_MODEL_HEAD_DIVISIBILITY_MSG == "d_model must be divisible by nhead"
    assert torch_nn._DIM_FEEDFORWARD_MIN_MSG == "dim_feedforward must be >= 1"
    assert torch_nn._LOW_FREQ_BINS_MIN_MSG == "low_freq_bins must be >= 1"
    assert torch_nn._PATCH_LEN_MIN_MSG == "patch_len must be >= 1"
    assert torch_nn._SEGMENT_LEN_MIN_MSG == "segment_len must be >= 1"
    assert torch_nn._SEGMENT_LEN_MAX_LAGS_MSG == "segment_len must be <= lags"


def test_torch_train_config_exposes_extended_training_fields() -> None:
    annotations = torch_nn.TorchTrainConfig.__annotations__

    for name in (
        "min_epochs",
        "amp",
        "amp_dtype",
        "warmup_epochs",
        "min_lr",
        "scheduler_restart_period",
        "scheduler_restart_mult",
        "scheduler_pct_start",
        "grad_accum_steps",
        "monitor",
        "monitor_mode",
        "min_delta",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "scheduler_patience",
        "grad_clip_mode",
        "grad_clip_value",
        "scheduler_plateau_factor",
        "scheduler_plateau_threshold",
        "ema_decay",
        "ema_warmup_epochs",
        "swa_start_epoch",
        "lookahead_steps",
        "lookahead_alpha",
        "sam_rho",
        "sam_adaptive",
        "horizon_loss_decay",
        "input_dropout",
        "temporal_dropout",
        "grad_noise_std",
        "gc_mode",
        "agc_clip_factor",
        "agc_eps",
        "checkpoint_dir",
        "save_best_checkpoint",
        "save_last_checkpoint",
        "resume_checkpoint_path",
        "resume_checkpoint_strict",
    ):
        assert name in annotations


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_temporal_dropout_helper_masks_whole_timesteps() -> None:
    torch = torch_nn._require_torch()
    cfg = torch_nn.TorchTrainConfig(
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=4,
        seed=0,
        patience=1,
        temporal_dropout=0.5,
    )
    xb = torch.ones((4, 8, 3), dtype=torch.float32)
    torch.manual_seed(0)

    out = torch_nn._apply_torch_train_temporal_dropout(torch, xb, cfg=cfg)

    assert out.shape == xb.shape
    assert torch.allclose(out[:, :, 0], out[:, :, 1])
    assert torch.allclose(out[:, :, 1], out[:, :, 2])
    assert torch.all((out == 0.0) | (out == 2.0))
    assert not torch.allclose(out, xb)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_horizon_loss_decay_helper_frontloads_earlier_steps() -> None:
    torch = torch_nn._require_torch()
    cfg = torch_nn.TorchTrainConfig(
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=2,
        seed=0,
        patience=1,
        horizon_loss_decay=0.5,
    )
    loss = torch.tensor([[1.0, 3.0]], dtype=torch.float32)

    weighted = torch_nn._reduce_torch_horizon_loss(torch, loss, cfg=cfg)

    assert weighted.item() == pytest.approx(5.0 / 3.0)
    assert weighted.item() < float(loss.mean().item())


def test_manual_recurrent_builders_use_shared_validation_messages() -> None:
    try:
        torch_nn._make_manual_gru_cell(input_size=1, hidden_size=0)
    except ValueError as exc:
        assert str(exc) == torch_nn._HIDDEN_SIZE_MIN_MSG
    else:
        raise AssertionError("expected hidden_size validation error")

    try:
        torch_nn._make_manual_gru(input_size=1, hidden_size=1, num_layers=0)
    except ValueError as exc:
        assert str(exc) == torch_nn._NUM_LAYERS_MIN_MSG
    else:
        raise AssertionError("expected num_layers validation error")

    try:
        torch_nn._make_manual_lstm(input_size=1, hidden_size=1, num_layers=1, dropout=1.0)
    except ValueError as exc:
        assert str(exc) == torch_nn._DROPOUT_RANGE_MSG
    else:
        raise AssertionError("expected dropout validation error")


def test_lagged_xy_builder_uses_shared_horizon_validation_message() -> None:
    try:
        torch_nn._make_lagged_xy_multi(torch_nn.np.arange(10.0), lags=3, horizon=0)
    except ValueError as exc:
        assert str(exc) == torch_nn._HORIZON_MIN_MSG
    else:
        raise AssertionError("expected horizon validation error")

    try:
        torch_nn._make_lagged_xy_multi(torch_nn.np.arange(10.0), lags=0, horizon=1)
    except ValueError as exc:
        assert str(exc) == torch_nn._LAGS_MIN_MSG
    else:
        raise AssertionError("expected lags validation error")


def test_torch_nn_direct_forecasters_use_shared_structural_validation_messages() -> None:
    train = torch_nn.np.arange(20.0)

    cases = [
        (
            torch_nn.torch_nbeats_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "num_blocks": 0},
            "_NUM_BLOCKS_MIN_MSG",
        ),
        (
            torch_nn.torch_transformer_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "d_model": 0},
            "_D_MODEL_MIN_MSG",
        ),
        (
            torch_nn.torch_transformer_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "nhead": 0},
            "_NHEAD_MIN_MSG",
        ),
        (
            torch_nn.torch_transformer_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "d_model": 5, "nhead": 2},
            "_D_MODEL_HEAD_DIVISIBILITY_MSG",
        ),
        (
            torch_nn.torch_transformer_direct_forecast,
            {
                "train": train,
                "horizon": 1,
                "lags": 4,
                "d_model": 4,
                "nhead": 2,
                "dim_feedforward": 0,
            },
            "_DIM_FEEDFORWARD_MIN_MSG",
        ),
        (
            torch_nn.torch_patchtst_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "patch_len": 0},
            "_PATCH_LEN_MIN_MSG",
        ),
        (
            torch_nn.torch_tinytimemixer_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "patch_len": 0},
            "_PATCH_LEN_MIN_MSG",
        ),
        (
            torch_nn.torch_fits_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "low_freq_bins": 0},
            "_LOW_FREQ_BINS_MIN_MSG",
        ),
        (
            torch_nn.torch_crossformer_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "segment_len": 0},
            "_SEGMENT_LEN_MIN_MSG",
        ),
        (
            torch_nn.torch_crossformer_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "segment_len": 5},
            "_SEGMENT_LEN_MAX_LAGS_MSG",
        ),
    ]

    for fn, kwargs, expected_attr in cases:
        try:
            fn(**kwargs)
        except ValueError as exc:
            assert str(exc) == getattr(torch_nn, expected_attr)
        else:
            raise AssertionError(f"expected validation error for {fn.__name__}")


LOCAL_TRAINING_CONFIG_CASES = [
    ("min epochs", {"min_epochs": 0}, "min_epochs must be >= 1"),
    ("min epochs > epochs", {"epochs": 1, "min_epochs": 2}, "min_epochs must be <= epochs"),
    ("warmup epochs", {"warmup_epochs": -1}, "warmup_epochs must be >= 0"),
    (
        "warmup epochs > epochs",
        {"epochs": 1, "warmup_epochs": 2},
        "warmup_epochs must be <= epochs",
    ),
    ("min lr", {"min_lr": -0.1}, "min_lr must be >= 0"),
    (
        "amp dtype invalid",
        {"amp_dtype": "fp8"},
        "amp_dtype must be one of: auto, float16, bfloat16",
    ),
    (
        "scheduler invalid",
        {"scheduler": "triangle"},
        "scheduler must be one of: none, cosine, step, plateau, onecycle, cosine_restarts",
    ),
    (
        "restart period",
        {"scheduler": "cosine_restarts", "scheduler_restart_period": 0},
        "scheduler_restart_period must be >= 1",
    ),
    (
        "restart mult",
        {"scheduler": "cosine_restarts", "scheduler_restart_mult": 0},
        "scheduler_restart_mult must be >= 1",
    ),
    (
        "pct start",
        {"scheduler": "onecycle", "scheduler_pct_start": 0.0},
        "scheduler_pct_start must be in (0, 1)",
    ),
    ("grad accum", {"grad_accum_steps": 0}, "grad_accum_steps must be >= 1"),
    ("monitor invalid", {"monitor": "score"}, "monitor must be one of: auto, train_loss, val_loss"),
    (
        "monitor val without split",
        {"monitor": "val_loss", "val_split": 0.0},
        "monitor='val_loss' requires val_split > 0",
    ),
    ("monitor mode invalid", {"monitor_mode": "sideways"}, "monitor_mode must be one of: min, max"),
    ("min delta", {"min_delta": -0.1}, "min_delta must be >= 0"),
    ("num workers", {"num_workers": -1}, "num_workers must be >= 0"),
    (
        "persistent workers",
        {"persistent_workers": True, "num_workers": 0},
        "persistent_workers requires num_workers >= 1",
    ),
    (
        "plateau patience",
        {"scheduler": "plateau", "scheduler_patience": 0},
        "scheduler_patience must be >= 1",
    ),
    (
        "grad clip mode",
        {"grad_clip_mode": "percentile"},
        "grad_clip_mode must be one of: norm, value",
    ),
    ("grad clip value", {"grad_clip_value": -0.1}, "grad_clip_value must be >= 0"),
    (
        "plateau factor",
        {"scheduler": "plateau", "scheduler_plateau_factor": 1.0},
        "scheduler_plateau_factor must be in (0, 1)",
    ),
    (
        "plateau threshold",
        {"scheduler": "plateau", "scheduler_plateau_threshold": -0.1},
        "scheduler_plateau_threshold must be >= 0",
    ),
    ("ema decay negative", {"ema_decay": -0.1}, "ema_decay must be in [0, 1)"),
    ("ema decay one", {"ema_decay": 1.0}, "ema_decay must be in [0, 1)"),
    (
        "ema warmup negative",
        {"ema_warmup_epochs": -1},
        "ema_warmup_epochs must be >= 0",
    ),
    (
        "ema warmup > epochs",
        {"epochs": 1, "ema_warmup_epochs": 2},
        "ema_warmup_epochs must be <= epochs",
    ),
    (
        "swa start < -1",
        {"swa_start_epoch": -2},
        "swa_start_epoch must be >= -1",
    ),
    (
        "swa start > epochs",
        {"epochs": 1, "swa_start_epoch": 2},
        "swa_start_epoch must be <= epochs",
    ),
    (
        "ema and swa conflict",
        {"ema_decay": 0.9, "swa_start_epoch": 0},
        "ema_decay and swa_start_epoch cannot both be enabled",
    ),
    (
        "lookahead steps negative",
        {"lookahead_steps": -1},
        "lookahead_steps must be >= 0",
    ),
    (
        "lookahead alpha zero",
        {"lookahead_steps": 1, "lookahead_alpha": 0.0},
        "lookahead_alpha must be in (0, 1]",
    ),
    (
        "lookahead alpha too large",
        {"lookahead_steps": 1, "lookahead_alpha": 1.1},
        "lookahead_alpha must be in (0, 1]",
    ),
    (
        "sam rho negative",
        {"sam_rho": -0.1},
        "sam_rho must be >= 0",
    ),
    (
        "horizon loss decay negative",
        {"horizon_loss_decay": -0.1},
        "horizon_loss_decay must be > 0",
    ),
    (
        "horizon loss decay zero",
        {"horizon_loss_decay": 0.0},
        "horizon_loss_decay must be > 0",
    ),
    (
        "input dropout negative",
        {"input_dropout": -0.1},
        "input_dropout must be in [0, 1)",
    ),
    (
        "input dropout one",
        {"input_dropout": 1.0},
        "input_dropout must be in [0, 1)",
    ),
    (
        "temporal dropout negative",
        {"temporal_dropout": -0.1},
        "temporal_dropout must be in [0, 1)",
    ),
    (
        "temporal dropout one",
        {"temporal_dropout": 1.0},
        "temporal_dropout must be in [0, 1)",
    ),
    (
        "grad noise std negative",
        {"grad_noise_std": -0.1},
        "grad_noise_std must be >= 0",
    ),
    (
        "gc mode invalid",
        {"gc_mode": "layerwise"},
        "gc_mode must be one of: off, all, conv_only",
    ),
    (
        "agc clip factor negative",
        {"agc_clip_factor": -0.1},
        "agc_clip_factor must be >= 0",
    ),
    (
        "agc eps zero",
        {"agc_eps": 0.0},
        "agc_eps must be > 0",
    ),
    (
        "best checkpoint without dir",
        {"save_best_checkpoint": True},
        "checkpoint_dir is required when checkpoint saving is enabled",
    ),
    (
        "last checkpoint without dir",
        {"save_last_checkpoint": True},
        "checkpoint_dir is required when checkpoint saving is enabled",
    ),
    (
        "resume checkpoint missing",
        {"resume_checkpoint_path": "/tmp/foresight-missing-resume.pt"},
        "resume_checkpoint_path does not exist",
    ),
    ("amp requires cuda", {"amp": True}, "amp=True requires device='cuda'"),
]


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
@pytest.mark.parametrize(
    ("case_id", "overrides", "message"),
    LOCAL_TRAINING_CONFIG_CASES,
    ids=[case[0] for case in LOCAL_TRAINING_CONFIG_CASES],
)
def test_torch_local_runtime_validates_extended_training_config(
    case_id: str,
    overrides: dict[str, object],
    message: str,
) -> None:
    train = np.arange(24.0, dtype=float)
    params: dict[str, object] = {
        "lags": 6,
        "hidden_sizes": (8,),
        "epochs": 1,
        "batch_size": 4,
        "seed": 0,
        "device": "cpu",
        "patience": 2,
        "val_split": 0.25,
    }
    params.update(overrides)
    forecaster = make_forecaster("torch-mlp-direct", **params)
    with pytest.raises(ValueError) as exc_info:
        forecaster(train, 2)
    assert str(exc_info.value) == message


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_seq2seq_direct_runtime_accepts_shared_training_config() -> None:
    train = np.linspace(0.0, 1.0, 24, dtype=float)
    forecaster = make_forecaster(
        "torch-seq2seq-lstm-direct",
        lags=6,
        hidden_size=8,
        num_layers=1,
        epochs=1,
        batch_size=4,
        seed=0,
        patience=2,
        device="cpu",
        grad_accum_steps=1,
        monitor="train_loss",
        monitor_mode="min",
        min_delta=0.0,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        scheduler_patience=5,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_onecycle_scheduler() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        scheduler="onecycle",
        scheduler_pct_start=0.3,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_value_clipping_and_plateau_controls() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        scheduler="plateau",
        scheduler_plateau_factor=0.5,
        scheduler_plateau_threshold=0.0,
        grad_clip_mode="value",
        grad_clip_value=0.1,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_ema_strategy() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        ema_decay=0.9,
        ema_warmup_epochs=0,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_swa_strategy() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        swa_start_epoch=0,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_lookahead_strategy() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        lookahead_steps=1,
        lookahead_alpha=0.5,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_sam_strategy() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        sam_rho=0.05,
        sam_adaptive=True,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_horizon_loss_decay_strategy() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        horizon_loss_decay=0.5,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_input_dropout_strategy() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        input_dropout=0.1,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_temporal_dropout_strategy() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        temporal_dropout=0.1,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_grad_noise_strategy() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        grad_noise_std=0.01,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_agc_strategy() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        agc_clip_factor=0.01,
        agc_eps=1e-3,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_accepts_gc_mode() -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        gc_mode="all",
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_writes_checkpoint_files(tmp_path) -> None:
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    checkpoint_dir = tmp_path / "local-checkpoints"
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        checkpoint_dir=str(checkpoint_dir),
        save_best_checkpoint=True,
        save_last_checkpoint=True,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)
    assert (checkpoint_dir / "best.pt").is_file()
    assert (checkpoint_dir / "last.pt").is_file()
    try:
        last_payload = torch_nn._require_torch().load(
            checkpoint_dir / "last.pt",
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        last_payload = torch_nn._require_torch().load(checkpoint_dir / "last.pt", map_location="cpu")
    assert "optimizer_state" in last_payload
    assert "epoch" in last_payload
    assert "best_monitor" in last_payload
    assert "bad_epochs" in last_payload
    assert "best_epoch" in last_payload
    assert "best_state" in last_payload


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_ema_checkpoints_store_raw_and_ema_state(tmp_path) -> None:
    torch = torch_nn._require_torch()
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    checkpoint_dir = tmp_path / "local-ema-checkpoints"
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        checkpoint_dir=str(checkpoint_dir),
        save_last_checkpoint=True,
        ema_decay=0.9,
        ema_warmup_epochs=0,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)
    try:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    except TypeError:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu")
    assert "ema_state" in last_payload
    assert "model_state" in last_payload


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_swa_checkpoints_store_raw_and_swa_state(tmp_path) -> None:
    torch = torch_nn._require_torch()
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    checkpoint_dir = tmp_path / "local-swa-checkpoints"
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        checkpoint_dir=str(checkpoint_dir),
        save_last_checkpoint=True,
        swa_start_epoch=0,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)
    try:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    except TypeError:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu")
    assert "swa_state" in last_payload
    assert "model_state" in last_payload


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_local_runtime_lookahead_checkpoints_store_raw_and_slow_state(tmp_path) -> None:
    torch = torch_nn._require_torch()
    train = np.linspace(0.0, 1.0, 32, dtype=float)
    checkpoint_dir = tmp_path / "local-lookahead-checkpoints"
    forecaster = make_forecaster(
        "torch-mlp-direct",
        lags=6,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=4,
        seed=0,
        patience=3,
        device="cpu",
        val_split=0.25,
        checkpoint_dir=str(checkpoint_dir),
        save_last_checkpoint=True,
        lookahead_steps=3,
        lookahead_alpha=0.5,
    )
    yhat = forecaster(train, 2)
    assert yhat.shape == (2,)
    try:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    except TypeError:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu")
    assert "lookahead_state" in last_payload
    assert "lookahead_step" in last_payload
    assert "model_state" in last_payload


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
@pytest.mark.parametrize("wrapped", [True, False], ids=["wrapped", "raw"])
def test_torch_resume_checkpoint_loader_restores_model_state(tmp_path, wrapped: bool) -> None:
    torch = torch_nn._require_torch()

    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(1.5)
        source.bias.fill_(0.25)
        target.weight.zero_()
        target.bias.zero_()

    state_dict = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
    payload: object = {"state_dict": state_dict, "epoch": 1} if wrapped else state_dict
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(payload, checkpoint_path)

    cfg = torch_nn.TorchTrainConfig(
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=1,
        seed=0,
        patience=1,
        resume_checkpoint_path=str(checkpoint_path),
        resume_checkpoint_strict=True,
    )

    torch_nn._load_torch_checkpoint_into_model(torch, target, cfg=cfg)

    assert torch.allclose(target.weight.detach().cpu(), source.weight.detach().cpu())
    assert torch.allclose(target.bias.detach().cpu(), source.bias.detach().cpu())


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_resume_checkpoint_loader_restores_training_state(tmp_path) -> None:
    torch = torch_nn._require_torch()

    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    deploy_target = torch.nn.Linear(2, 1)
    opt_source = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    opt_target = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    sched_source = torch.optim.lr_scheduler.StepLR(opt_source, step_size=1, gamma=0.5)
    sched_target = torch.optim.lr_scheduler.StepLR(opt_target, step_size=1, gamma=0.5)

    xb = torch.ones((4, 2), dtype=torch.float32)
    yb = torch.ones((4, 1), dtype=torch.float32)
    loss = torch.nn.functional.mse_loss(source(xb), yb)
    loss.backward()
    opt_source.step()
    opt_source.zero_grad(set_to_none=True)
    sched_source.step()

    raw_state = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
    deploy_state = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
    ema_state = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
    swa_state = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
    lookahead_state = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
    for key, value in deploy_state.items():
        if getattr(value, "dtype", None) is not None and torch.is_floating_point(value):
            deploy_state[key] = value + 1.0
    for key, value in ema_state.items():
        if getattr(value, "dtype", None) is not None and torch.is_floating_point(value):
            ema_state[key] = value + 2.0
    for key, value in swa_state.items():
        if getattr(value, "dtype", None) is not None and torch.is_floating_point(value):
            swa_state[key] = value + 3.0
    for key, value in lookahead_state.items():
        if getattr(value, "dtype", None) is not None and torch.is_floating_point(value):
            lookahead_state[key] = value + 4.0
    checkpoint_path = tmp_path / "resume-training.pt"
    torch.save(
        {
            "state_dict": deploy_state,
            "model_state": raw_state,
            "ema_state": ema_state,
            "swa_state": swa_state,
            "lookahead_state": lookahead_state,
            "lookahead_step": 7,
            "optimizer_state": opt_source.state_dict(),
            "scheduler_state": sched_source.state_dict(),
            "epoch": 2,
            "best_monitor": 0.25,
            "bad_epochs": 1,
            "best_epoch": 1,
            "best_state": deploy_state,
        },
        checkpoint_path,
    )

    cfg = torch_nn.TorchTrainConfig(
        epochs=4,
        lr=0.1,
        weight_decay=0.0,
        batch_size=1,
        seed=0,
        patience=2,
        resume_checkpoint_path=str(checkpoint_path),
        resume_checkpoint_strict=True,
    )

    torch_nn._load_torch_checkpoint_into_model(torch, deploy_target, cfg=cfg)
    resume_state = torch_nn._load_torch_training_state(
        torch,
        target,
        cfg=cfg,
        optimizer=opt_target,
        scheduler=sched_target,
        scaler=None,
    )

    assert torch.allclose(target.weight.detach().cpu(), source.weight.detach().cpu())
    assert torch.allclose(target.bias.detach().cpu(), source.bias.detach().cpu())
    assert torch.allclose(
        deploy_target.weight.detach().cpu(),
        deploy_state["weight"],
    )
    assert torch.allclose(
        deploy_target.bias.detach().cpu(),
        deploy_state["bias"],
    )
    assert opt_target.state_dict()["state"]
    assert sched_target.state_dict()["last_epoch"] == sched_source.state_dict()["last_epoch"]
    assert resume_state.start_epoch == 2
    assert resume_state.best_monitor == pytest.approx(0.25)
    assert resume_state.bad_epochs == 1
    assert resume_state.best_epoch == 1
    assert resume_state.best_state is not None
    assert resume_state.ema_state is not None
    assert resume_state.swa_state is not None
    assert resume_state.lookahead_state is not None
    assert resume_state.lookahead_step == 7
    assert torch.allclose(resume_state.ema_state["weight"], ema_state["weight"])
    assert torch.allclose(resume_state.ema_state["bias"], ema_state["bias"])
    assert torch.allclose(resume_state.swa_state["weight"], swa_state["weight"])
    assert torch.allclose(resume_state.swa_state["bias"], swa_state["bias"])
    assert torch.allclose(resume_state.lookahead_state["weight"], lookahead_state["weight"])
    assert torch.allclose(resume_state.lookahead_state["bias"], lookahead_state["bias"])


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_train_loop_resume_restores_epoch_and_early_stop_state(tmp_path) -> None:
    torch = torch_nn._require_torch()

    class _ConstantRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32))

        def forward(self, xb):
            return self.bias.expand(int(xb.shape[0]), 1)

    X = np.ones((12, 2, 1), dtype=np.float32)
    Y = np.zeros((12, 1), dtype=np.float32)

    source = _ConstantRegressor()
    checkpoint_path = tmp_path / "resume-loop.pt"
    checkpoint_dir = tmp_path / "resume-loop-checkpoints"
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu().clone() for key, value in source.state_dict().items()
            },
            "epoch": 2,
            "monitor": 0.25,
            "best_monitor": 0.0,
            "bad_epochs": 1,
            "best_epoch": 1,
            "best_state": {
                key: value.detach().cpu().clone() for key, value in source.state_dict().items()
            },
        },
        checkpoint_path,
    )

    cfg = torch_nn.TorchTrainConfig(
        epochs=5,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=4,
        seed=0,
        patience=1,
        loss="mse",
        val_split=0.0,
        grad_clip_norm=0.0,
        optimizer="adam",
        monitor="train_loss",
        monitor_mode="min",
        min_delta=1.0,
        restore_best=False,
        checkpoint_dir=str(checkpoint_dir),
        save_last_checkpoint=True,
        resume_checkpoint_path=str(checkpoint_path),
        resume_checkpoint_strict=True,
    )

    torch_nn._train_loop(_ConstantRegressor(), X, Y, cfg=cfg, device="cpu")

    try:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    except TypeError:
        last_payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu")

    assert last_payload["epoch"] == 3
    assert last_payload["best_monitor"] == pytest.approx(0.0)
    assert last_payload["bad_epochs"] == 2
    assert last_payload["best_epoch"] == 1
