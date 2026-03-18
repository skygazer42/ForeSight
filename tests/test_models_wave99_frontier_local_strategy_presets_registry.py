from foresight.models.registry import get_model_spec, list_models

FRONTIER_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-cfc-ema-direct",
    "torch-xlstm-swa-direct",
    "torch-griffin-sam-direct",
    "torch-hawk-regularized-direct",
    "torch-perceiver-lookahead-direct",
    "torch-moderntcn-longhorizon-direct",
)


def test_wave99_frontier_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in FRONTIER_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave99_frontier_local_strategy_presets_are_local_torch_optional() -> None:
    for key in FRONTIER_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave99_frontier_local_strategy_preset_defaults() -> None:
    cfc_ema = get_model_spec("torch-cfc-ema-direct")
    xlstm_swa = get_model_spec("torch-xlstm-swa-direct")
    griffin_sam = get_model_spec("torch-griffin-sam-direct")
    hawk_regularized = get_model_spec("torch-hawk-regularized-direct")
    perceiver_lookahead = get_model_spec("torch-perceiver-lookahead-direct")
    moderntcn_longhorizon = get_model_spec("torch-moderntcn-longhorizon-direct")

    assert cfc_ema.default_params["backbone_hidden"] == 128
    assert cfc_ema.default_params["ema_decay"] == 0.995
    assert xlstm_swa.default_params["proj_factor"] == 2
    assert xlstm_swa.default_params["swa_start_epoch"] == 18
    assert griffin_sam.default_params["conv_kernel"] == 3
    assert griffin_sam.default_params["sam_rho"] == 0.05
    assert griffin_sam.default_params["sam_adaptive"] is True
    assert hawk_regularized.default_params["expansion_factor"] == 2
    assert hawk_regularized.default_params["dropout"] == 0.2
    assert perceiver_lookahead.default_params["latent_len"] == 32
    assert perceiver_lookahead.default_params["lookahead_steps"] == 5
    assert perceiver_lookahead.default_params["lookahead_alpha"] == 0.5
    assert moderntcn_longhorizon.default_params["kernel_size"] == 9
    assert moderntcn_longhorizon.default_params["loss"] == "huber"
    assert moderntcn_longhorizon.default_params["horizon_loss_decay"] == 1.05
