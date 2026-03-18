from foresight.models.registry import get_model_spec, list_models

TRANSFORMER_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-informer-ema-direct",
    "torch-autoformer-swa-direct",
    "torch-fedformer-sam-direct",
    "torch-crossformer-regularized-direct",
    "torch-timexer-lookahead-direct",
    "torch-itransformer-longhorizon-direct",
)


def test_wave104_transformer_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in TRANSFORMER_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave104_transformer_local_strategy_presets_are_local_torch_optional() -> None:
    for key in TRANSFORMER_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave104_transformer_local_strategy_preset_defaults() -> None:
    informer_ema = get_model_spec("torch-informer-ema-direct")
    autoformer_swa = get_model_spec("torch-autoformer-swa-direct")
    fedformer_sam = get_model_spec("torch-fedformer-sam-direct")
    crossformer_regularized = get_model_spec("torch-crossformer-regularized-direct")
    timexer_lookahead = get_model_spec("torch-timexer-lookahead-direct")
    itransformer_longhorizon = get_model_spec("torch-itransformer-longhorizon-direct")

    assert informer_ema.default_params["d_model"] == 64
    assert informer_ema.default_params["ema_decay"] == 0.995
    assert autoformer_swa.default_params["ma_window"] == 7
    assert autoformer_swa.default_params["swa_start_epoch"] == 18
    assert fedformer_sam.default_params["modes"] == 16
    assert fedformer_sam.default_params["sam_rho"] == 0.05
    assert fedformer_sam.default_params["sam_adaptive"] is True
    assert crossformer_regularized.default_params["segment_len"] == 16
    assert crossformer_regularized.default_params["dropout"] == 0.2
    assert timexer_lookahead.default_params["nhead"] == 4
    assert timexer_lookahead.default_params["lookahead_steps"] == 5
    assert timexer_lookahead.default_params["lookahead_alpha"] == 0.5
    assert itransformer_longhorizon.default_params["d_model"] == 64
    assert itransformer_longhorizon.default_params["loss"] == "huber"
    assert itransformer_longhorizon.default_params["horizon_loss_decay"] == 1.05
