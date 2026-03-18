from foresight.models.registry import get_model_spec, list_models

MODERN_MIX_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-tft-ema-direct",
    "torch-tsmixer-swa-direct",
    "torch-timesnet-sam-direct",
    "torch-patchtst-regularized-direct",
    "torch-retnet-lookahead-direct",
    "torch-timexer-longhorizon-direct",
)


def test_wave105_modern_mix_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in MODERN_MIX_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave105_modern_mix_local_strategy_presets_are_local_torch_optional() -> None:
    for key in MODERN_MIX_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave105_modern_mix_local_strategy_preset_defaults() -> None:
    tft_ema = get_model_spec("torch-tft-ema-direct")
    tsmixer_swa = get_model_spec("torch-tsmixer-swa-direct")
    timesnet_sam = get_model_spec("torch-timesnet-sam-direct")
    patchtst_regularized = get_model_spec("torch-patchtst-regularized-direct")
    retnet_lookahead = get_model_spec("torch-retnet-lookahead-direct")
    timexer_longhorizon = get_model_spec("torch-timexer-longhorizon-direct")

    assert tft_ema.default_params["d_model"] == 64
    assert tft_ema.default_params["ema_decay"] == 0.995
    assert tsmixer_swa.default_params["num_blocks"] == 4
    assert tsmixer_swa.default_params["swa_start_epoch"] == 18
    assert timesnet_sam.default_params["top_k"] == 3
    assert timesnet_sam.default_params["sam_rho"] == 0.05
    assert timesnet_sam.default_params["sam_adaptive"] is True
    assert patchtst_regularized.default_params["patch_len"] == 16
    assert patchtst_regularized.default_params["dropout"] == 0.2
    assert retnet_lookahead.default_params["ffn_dim"] == 128
    assert retnet_lookahead.default_params["lookahead_steps"] == 5
    assert retnet_lookahead.default_params["lookahead_alpha"] == 0.5
    assert timexer_longhorizon.default_params["nhead"] == 4
    assert timexer_longhorizon.default_params["loss"] == "huber"
    assert timexer_longhorizon.default_params["horizon_loss_decay"] == 1.05
