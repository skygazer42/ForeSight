from foresight.models.registry import get_model_spec, list_models

STATE_SPACE_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-lmu-ema-direct",
    "torch-ltc-swa-direct",
    "torch-s4-sam-direct",
    "torch-s4d-regularized-direct",
    "torch-s5-lookahead-direct",
    "torch-mamba2-longhorizon-direct",
)


def test_wave98_state_space_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in STATE_SPACE_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave98_state_space_local_strategy_presets_are_local_torch_optional() -> None:
    for key in STATE_SPACE_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave98_state_space_local_strategy_preset_defaults() -> None:
    lmu_ema = get_model_spec("torch-lmu-ema-direct")
    ltc_swa = get_model_spec("torch-ltc-swa-direct")
    s4_sam = get_model_spec("torch-s4-sam-direct")
    s4d_regularized = get_model_spec("torch-s4d-regularized-direct")
    s5_lookahead = get_model_spec("torch-s5-lookahead-direct")
    mamba2_longhorizon = get_model_spec("torch-mamba2-longhorizon-direct")

    assert lmu_ema.default_params["memory_dim"] == 32
    assert lmu_ema.default_params["ema_decay"] == 0.995
    assert ltc_swa.default_params["hidden_size"] == 64
    assert ltc_swa.default_params["swa_start_epoch"] == 18
    assert s4_sam.default_params["state_dim"] == 32
    assert s4_sam.default_params["sam_rho"] == 0.05
    assert s4_sam.default_params["sam_adaptive"] is True
    assert s4d_regularized.default_params["d_model"] == 64
    assert s4d_regularized.default_params["dropout"] == 0.2
    assert s5_lookahead.default_params["heads"] == 2
    assert s5_lookahead.default_params["lookahead_steps"] == 5
    assert s5_lookahead.default_params["lookahead_alpha"] == 0.5
    assert mamba2_longhorizon.default_params["conv_kernel"] == 3
    assert mamba2_longhorizon.default_params["loss"] == "huber"
    assert mamba2_longhorizon.default_params["horizon_loss_decay"] == 1.05
