from foresight.models.registry import get_model_spec, list_models

STATE_SPACE_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-s4-ema-direct",
    "torch-s4d-swa-direct",
    "torch-s5-sam-direct",
    "torch-mamba2-regularized-direct",
    "torch-timesmamba-lookahead-direct",
    "torch-pathformer-swa-direct",
)


def test_wave107_state_space_refinement_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in STATE_SPACE_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave107_state_space_refinement_local_strategy_presets_are_local_torch_optional() -> None:
    for key in STATE_SPACE_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave107_state_space_refinement_local_strategy_preset_defaults() -> None:
    s4_ema = get_model_spec("torch-s4-ema-direct")
    s4d_swa = get_model_spec("torch-s4d-swa-direct")
    s5_sam = get_model_spec("torch-s5-sam-direct")
    mamba2_regularized = get_model_spec("torch-mamba2-regularized-direct")
    timesmamba_lookahead = get_model_spec("torch-timesmamba-lookahead-direct")
    pathformer_swa = get_model_spec("torch-pathformer-swa-direct")

    assert s4_ema.default_params["d_model"] == 64
    assert s4_ema.default_params["ema_decay"] == 0.995
    assert s4d_swa.default_params["d_model"] == 64
    assert s4d_swa.default_params["swa_start_epoch"] == 18
    assert s5_sam.default_params["state_dim"] == 32
    assert s5_sam.default_params["sam_rho"] == 0.05
    assert s5_sam.default_params["sam_adaptive"] is True
    assert mamba2_regularized.default_params["d_model"] == 64
    assert mamba2_regularized.default_params["dropout"] == 0.2
    assert timesmamba_lookahead.default_params["state_size"] == 64
    assert timesmamba_lookahead.default_params["lookahead_steps"] == 5
    assert timesmamba_lookahead.default_params["lookahead_alpha"] == 0.5
    assert pathformer_swa.default_params["d_model"] == 64
    assert pathformer_swa.default_params["swa_start_epoch"] == 18
