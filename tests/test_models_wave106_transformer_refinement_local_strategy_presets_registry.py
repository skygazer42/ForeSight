from foresight.models.registry import get_model_spec, list_models

TRANSFORMER_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-informer-sam-direct",
    "torch-autoformer-regularized-direct",
    "torch-fedformer-lookahead-direct",
    "torch-crossformer-ema-direct",
    "torch-itransformer-swa-direct",
    "torch-timexer-ema-direct",
)


def test_wave106_transformer_refinement_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in TRANSFORMER_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave106_transformer_refinement_local_strategy_presets_are_local_torch_optional() -> None:
    for key in TRANSFORMER_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave106_transformer_refinement_local_strategy_preset_defaults() -> None:
    informer_sam = get_model_spec("torch-informer-sam-direct")
    autoformer_regularized = get_model_spec("torch-autoformer-regularized-direct")
    fedformer_lookahead = get_model_spec("torch-fedformer-lookahead-direct")
    crossformer_ema = get_model_spec("torch-crossformer-ema-direct")
    itransformer_swa = get_model_spec("torch-itransformer-swa-direct")
    timexer_ema = get_model_spec("torch-timexer-ema-direct")

    assert informer_sam.default_params["d_model"] == 64
    assert informer_sam.default_params["sam_rho"] == 0.05
    assert informer_sam.default_params["sam_adaptive"] is True
    assert autoformer_regularized.default_params["ma_window"] == 7
    assert autoformer_regularized.default_params["dropout"] == 0.2
    assert fedformer_lookahead.default_params["modes"] == 16
    assert fedformer_lookahead.default_params["lookahead_steps"] == 5
    assert fedformer_lookahead.default_params["lookahead_alpha"] == 0.5
    assert crossformer_ema.default_params["segment_len"] == 16
    assert crossformer_ema.default_params["ema_decay"] == 0.995
    assert itransformer_swa.default_params["d_model"] == 64
    assert itransformer_swa.default_params["swa_start_epoch"] == 18
    assert timexer_ema.default_params["nhead"] == 4
    assert timexer_ema.default_params["ema_decay"] == 0.995
