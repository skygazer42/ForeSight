from foresight.models.registry import get_model_spec, list_models

ADVANCED_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-retnet-ema-direct",
    "torch-crossformer-swa-direct",
    "torch-pyraformer-sam-direct",
    "torch-lightts-regularized-direct",
    "torch-samformer-lookahead-direct",
    "torch-timesmamba-longhorizon-direct",
)


def test_wave96_advanced_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in ADVANCED_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave96_advanced_local_strategy_presets_are_local_torch_optional() -> None:
    for key in ADVANCED_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave96_advanced_local_strategy_preset_defaults() -> None:
    retnet_ema = get_model_spec("torch-retnet-ema-direct")
    crossformer_swa = get_model_spec("torch-crossformer-swa-direct")
    pyraformer_sam = get_model_spec("torch-pyraformer-sam-direct")
    lightts_regularized = get_model_spec("torch-lightts-regularized-direct")
    samformer_lookahead = get_model_spec("torch-samformer-lookahead-direct")
    timesmamba_longhorizon = get_model_spec("torch-timesmamba-longhorizon-direct")

    assert retnet_ema.default_params["ffn_dim"] == 128
    assert retnet_ema.default_params["ema_decay"] == 0.995
    assert crossformer_swa.default_params["num_scales"] == 3
    assert crossformer_swa.default_params["swa_start_epoch"] == 18
    assert pyraformer_sam.default_params["num_levels"] == 3
    assert pyraformer_sam.default_params["sam_rho"] == 0.05
    assert pyraformer_sam.default_params["sam_adaptive"] is True
    assert lightts_regularized.default_params["chunk_len"] == 12
    assert lightts_regularized.default_params["dropout"] == 0.2
    assert samformer_lookahead.default_params["num_layers"] == 2
    assert samformer_lookahead.default_params["lookahead_steps"] == 5
    assert samformer_lookahead.default_params["lookahead_alpha"] == 0.5
    assert timesmamba_longhorizon.default_params["state_size"] == 64
    assert timesmamba_longhorizon.default_params["loss"] == "huber"
    assert timesmamba_longhorizon.default_params["horizon_loss_decay"] == 1.05
