from foresight.models.registry import get_model_spec, list_models

TRANSFORMER_COMPLETION_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-informer-longhorizon-direct",
    "torch-autoformer-ema-direct",
    "torch-fedformer-swa-direct",
    "torch-crossformer-lookahead-direct",
    "torch-itransformer-regularized-direct",
    "torch-timexer-swa-direct",
)


def test_wave108_transformer_completion_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in TRANSFORMER_COMPLETION_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave108_transformer_completion_local_strategy_presets_are_local_torch_optional() -> None:
    for key in TRANSFORMER_COMPLETION_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave108_transformer_completion_local_strategy_preset_defaults() -> None:
    informer_longhorizon = get_model_spec("torch-informer-longhorizon-direct")
    autoformer_ema = get_model_spec("torch-autoformer-ema-direct")
    fedformer_swa = get_model_spec("torch-fedformer-swa-direct")
    crossformer_lookahead = get_model_spec("torch-crossformer-lookahead-direct")
    itransformer_regularized = get_model_spec("torch-itransformer-regularized-direct")
    timexer_swa = get_model_spec("torch-timexer-swa-direct")

    assert informer_longhorizon.default_params["d_model"] == 64
    assert informer_longhorizon.default_params["loss"] == "huber"
    assert informer_longhorizon.default_params["horizon_loss_decay"] == 1.05
    assert informer_longhorizon.default_params["ema_decay"] == 0.99
    assert autoformer_ema.default_params["ma_window"] == 7
    assert autoformer_ema.default_params["ema_decay"] == 0.995
    assert fedformer_swa.default_params["modes"] == 16
    assert fedformer_swa.default_params["swa_start_epoch"] == 18
    assert crossformer_lookahead.default_params["segment_len"] == 16
    assert crossformer_lookahead.default_params["lookahead_steps"] == 5
    assert crossformer_lookahead.default_params["lookahead_alpha"] == 0.5
    assert itransformer_regularized.default_params["d_model"] == 64
    assert itransformer_regularized.default_params["dropout"] == 0.2
    assert timexer_swa.default_params["nhead"] == 4
    assert timexer_swa.default_params["swa_start_epoch"] == 18
