from foresight.models.registry import get_model_spec, list_models

CORE_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-lstm-ema-direct",
    "torch-gru-swa-direct",
    "torch-attn-gru-sam-direct",
    "torch-tcn-regularized-direct",
    "torch-cnn-lookahead-direct",
    "torch-mlp-longhorizon-direct",
)


def test_wave100_core_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in CORE_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave100_core_local_strategy_presets_are_local_torch_optional() -> None:
    for key in CORE_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave100_core_local_strategy_preset_defaults() -> None:
    lstm_ema = get_model_spec("torch-lstm-ema-direct")
    gru_swa = get_model_spec("torch-gru-swa-direct")
    attn_gru_sam = get_model_spec("torch-attn-gru-sam-direct")
    tcn_regularized = get_model_spec("torch-tcn-regularized-direct")
    cnn_lookahead = get_model_spec("torch-cnn-lookahead-direct")
    mlp_longhorizon = get_model_spec("torch-mlp-longhorizon-direct")

    assert lstm_ema.default_params["hidden_size"] == 32
    assert lstm_ema.default_params["ema_decay"] == 0.995
    assert gru_swa.default_params["hidden_size"] == 32
    assert gru_swa.default_params["swa_start_epoch"] == 18
    assert attn_gru_sam.default_params["hidden_size"] == 32
    assert attn_gru_sam.default_params["sam_rho"] == 0.05
    assert attn_gru_sam.default_params["sam_adaptive"] is True
    assert tcn_regularized.default_params["kernel_size"] == 3
    assert tcn_regularized.default_params["dropout"] == 0.2
    assert cnn_lookahead.default_params["pool"] == "last"
    assert cnn_lookahead.default_params["lookahead_steps"] == 5
    assert cnn_lookahead.default_params["lookahead_alpha"] == 0.5
    assert mlp_longhorizon.default_params["hidden_sizes"] == (64, 64)
    assert mlp_longhorizon.default_params["loss"] == "huber"
    assert mlp_longhorizon.default_params["horizon_loss_decay"] == 1.05
