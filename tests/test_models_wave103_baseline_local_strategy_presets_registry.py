from foresight.models.registry import get_model_spec, list_models

BASELINE_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-nhits-ema-direct",
    "torch-nbeats-sam-direct",
    "torch-tide-regularized-direct",
    "torch-dlinear-lookahead-direct",
    "torch-nlinear-longhorizon-direct",
    "torch-timemixer-ema-direct",
)


def test_wave103_baseline_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in BASELINE_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave103_baseline_local_strategy_presets_are_local_torch_optional() -> None:
    for key in BASELINE_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave103_baseline_local_strategy_preset_defaults() -> None:
    nhits_ema = get_model_spec("torch-nhits-ema-direct")
    nbeats_sam = get_model_spec("torch-nbeats-sam-direct")
    tide_regularized = get_model_spec("torch-tide-regularized-direct")
    dlinear_lookahead = get_model_spec("torch-dlinear-lookahead-direct")
    nlinear_longhorizon = get_model_spec("torch-nlinear-longhorizon-direct")
    timemixer_ema = get_model_spec("torch-timemixer-ema-direct")

    assert nhits_ema.default_params["layer_width"] == 128
    assert nhits_ema.default_params["ema_decay"] == 0.995
    assert nbeats_sam.default_params["layer_width"] == 64
    assert nbeats_sam.default_params["sam_rho"] == 0.05
    assert nbeats_sam.default_params["sam_adaptive"] is True
    assert tide_regularized.default_params["hidden_size"] == 128
    assert tide_regularized.default_params["dropout"] == 0.2
    assert dlinear_lookahead.default_params["ma_window"] == 25
    assert dlinear_lookahead.default_params["lookahead_steps"] == 5
    assert dlinear_lookahead.default_params["lookahead_alpha"] == 0.5
    assert nlinear_longhorizon.default_params["lags"] == 48
    assert nlinear_longhorizon.default_params["loss"] == "huber"
    assert nlinear_longhorizon.default_params["horizon_loss_decay"] == 1.05
    assert timemixer_ema.default_params["num_blocks"] == 4
    assert timemixer_ema.default_params["ema_decay"] == 0.995
