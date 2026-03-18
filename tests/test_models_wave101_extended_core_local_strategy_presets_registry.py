from foresight.models.registry import get_model_spec, list_models

EXTENDED_CORE_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-bigru-ema-direct",
    "torch-bilstm-swa-direct",
    "torch-linear-attn-sam-direct",
    "torch-koopa-regularized-direct",
    "torch-fits-lookahead-direct",
    "torch-film-longhorizon-direct",
)


def test_wave101_extended_core_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in EXTENDED_CORE_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave101_extended_core_local_strategy_presets_are_local_torch_optional() -> None:
    for key in EXTENDED_CORE_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave101_extended_core_local_strategy_preset_defaults() -> None:
    bigru_ema = get_model_spec("torch-bigru-ema-direct")
    bilstm_swa = get_model_spec("torch-bilstm-swa-direct")
    linear_attn_sam = get_model_spec("torch-linear-attn-sam-direct")
    koopa_regularized = get_model_spec("torch-koopa-regularized-direct")
    fits_lookahead = get_model_spec("torch-fits-lookahead-direct")
    film_longhorizon = get_model_spec("torch-film-longhorizon-direct")

    assert bigru_ema.default_params["hidden_size"] == 32
    assert bigru_ema.default_params["ema_decay"] == 0.995
    assert bilstm_swa.default_params["hidden_size"] == 32
    assert bilstm_swa.default_params["swa_start_epoch"] == 18
    assert linear_attn_sam.default_params["d_model"] == 64
    assert linear_attn_sam.default_params["sam_rho"] == 0.05
    assert linear_attn_sam.default_params["sam_adaptive"] is True
    assert koopa_regularized.default_params["latent_dim"] == 32
    assert koopa_regularized.default_params["dropout"] == 0.2
    assert fits_lookahead.default_params["low_freq_bins"] == 12
    assert fits_lookahead.default_params["lookahead_steps"] == 5
    assert fits_lookahead.default_params["lookahead_alpha"] == 0.5
    assert film_longhorizon.default_params["ma_window"] == 7
    assert film_longhorizon.default_params["loss"] == "huber"
    assert film_longhorizon.default_params["horizon_loss_decay"] == 1.05
