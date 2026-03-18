from foresight.models.registry import get_model_spec, list_models

LOCAL_XFORMER_STRATEGY_PRESET_KEYS = (
    "torch-xformer-full-ema-direct",
    "torch-xformer-performer-swa-direct",
    "torch-xformer-linformer-sam-direct",
    "torch-xformer-nystrom-regularized-direct",
    "torch-xformer-bigbird-longhorizon-direct",
    "torch-xformer-longformer-lookahead-direct",
)

GLOBAL_XFORMER_STRATEGY_PRESET_KEYS = (
    "torch-xformer-full-ema-global",
    "torch-xformer-performer-swa-global",
    "torch-xformer-linformer-sam-global",
    "torch-xformer-nystrom-regularized-global",
    "torch-xformer-bigbird-longhorizon-global",
    "torch-xformer-longformer-lookahead-global",
)


def test_wave52_xformer_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in LOCAL_XFORMER_STRATEGY_PRESET_KEYS + GLOBAL_XFORMER_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave52_xformer_strategy_presets_are_torch_local_or_global() -> None:
    for key in LOCAL_XFORMER_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires

    for key in GLOBAL_XFORMER_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "global"
        assert "torch" in spec.requires


def test_wave52_xformer_strategy_preset_defaults() -> None:
    full_ema_direct = get_model_spec("torch-xformer-full-ema-direct")
    performer_swa_direct = get_model_spec("torch-xformer-performer-swa-direct")
    linformer_sam_direct = get_model_spec("torch-xformer-linformer-sam-direct")
    nystrom_regularized_direct = get_model_spec("torch-xformer-nystrom-regularized-direct")
    bigbird_longhorizon_direct = get_model_spec("torch-xformer-bigbird-longhorizon-direct")
    longformer_lookahead_direct = get_model_spec("torch-xformer-longformer-lookahead-direct")

    full_ema_global = get_model_spec("torch-xformer-full-ema-global")
    performer_swa_global = get_model_spec("torch-xformer-performer-swa-global")
    linformer_sam_global = get_model_spec("torch-xformer-linformer-sam-global")
    nystrom_regularized_global = get_model_spec("torch-xformer-nystrom-regularized-global")
    bigbird_longhorizon_global = get_model_spec("torch-xformer-bigbird-longhorizon-global")
    longformer_lookahead_global = get_model_spec("torch-xformer-longformer-lookahead-global")

    assert full_ema_direct.default_params["attn"] == "full"
    assert full_ema_direct.default_params["ema_decay"] == 0.995
    assert performer_swa_direct.default_params["attn"] == "performer"
    assert performer_swa_direct.default_params["swa_start_epoch"] == 18
    assert linformer_sam_direct.default_params["attn"] == "linformer"
    assert linformer_sam_direct.default_params["sam_rho"] == 0.05
    assert linformer_sam_direct.default_params["sam_adaptive"] is True
    assert nystrom_regularized_direct.default_params["attn"] == "nystrom"
    assert nystrom_regularized_direct.default_params["input_dropout"] == 0.1
    assert nystrom_regularized_direct.default_params["temporal_dropout"] == 0.05
    assert bigbird_longhorizon_direct.default_params["attn"] == "bigbird"
    assert bigbird_longhorizon_direct.default_params["horizon_loss_decay"] == 1.05
    assert bigbird_longhorizon_direct.default_params["loss"] == "huber"
    assert longformer_lookahead_direct.default_params["attn"] == "longformer"
    assert longformer_lookahead_direct.default_params["lookahead_steps"] == 5
    assert longformer_lookahead_direct.default_params["lookahead_alpha"] == 0.5

    assert full_ema_global.default_params["attn"] == "full"
    assert full_ema_global.default_params["ema_decay"] == 0.995
    assert performer_swa_global.default_params["attn"] == "performer"
    assert performer_swa_global.default_params["swa_start_epoch"] == 18
    assert linformer_sam_global.default_params["attn"] == "linformer"
    assert linformer_sam_global.default_params["sam_rho"] == 0.05
    assert linformer_sam_global.default_params["sam_adaptive"] is True
    assert nystrom_regularized_global.default_params["attn"] == "nystrom"
    assert nystrom_regularized_global.default_params["input_dropout"] == 0.1
    assert nystrom_regularized_global.default_params["temporal_dropout"] == 0.05
    assert bigbird_longhorizon_global.default_params["attn"] == "bigbird"
    assert bigbird_longhorizon_global.default_params["horizon_loss_decay"] == 1.05
    assert bigbird_longhorizon_global.default_params["loss"] == "huber"
    assert longformer_lookahead_global.default_params["attn"] == "longformer"
    assert longformer_lookahead_global.default_params["lookahead_steps"] == 5
    assert longformer_lookahead_global.default_params["lookahead_alpha"] == 0.5
