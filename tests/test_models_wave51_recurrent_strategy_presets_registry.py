from foresight.models.registry import get_model_spec, list_models

RECURRENT_STRATEGY_PRESET_KEYS = (
    "torch-rnnpaper-lstm-ema-direct",
    "torch-rnnpaper-gru-swa-direct",
    "torch-rnnpaper-qrnn-lookahead-direct",
    "torch-rnnzoo-lstm-sam-direct",
    "torch-rnnzoo-gru-regularized-direct",
    "torch-rnnzoo-qrnn-longhorizon-direct",
)


def test_wave51_recurrent_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in RECURRENT_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave51_recurrent_strategy_presets_are_local_torch_optional() -> None:
    for key in RECURRENT_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave51_recurrent_strategy_preset_defaults() -> None:
    rnnpaper_lstm_ema = get_model_spec("torch-rnnpaper-lstm-ema-direct")
    rnnpaper_gru_swa = get_model_spec("torch-rnnpaper-gru-swa-direct")
    rnnpaper_qrnn_lookahead = get_model_spec("torch-rnnpaper-qrnn-lookahead-direct")
    rnnzoo_lstm_sam = get_model_spec("torch-rnnzoo-lstm-sam-direct")
    rnnzoo_gru_regularized = get_model_spec("torch-rnnzoo-gru-regularized-direct")
    rnnzoo_qrnn_longhorizon = get_model_spec("torch-rnnzoo-qrnn-longhorizon-direct")

    assert rnnpaper_lstm_ema.default_params["ema_decay"] == 0.995
    assert rnnpaper_lstm_ema.default_params["scheduler"] == "cosine"
    assert rnnpaper_gru_swa.default_params["swa_start_epoch"] == 18
    assert rnnpaper_gru_swa.default_params["scheduler"] == "cosine_restarts"
    assert rnnpaper_qrnn_lookahead.default_params["lookahead_steps"] == 5
    assert rnnpaper_qrnn_lookahead.default_params["lookahead_alpha"] == 0.5
    assert rnnzoo_lstm_sam.default_params["sam_rho"] == 0.05
    assert rnnzoo_lstm_sam.default_params["sam_adaptive"] is True
    assert rnnzoo_gru_regularized.default_params["input_dropout"] == 0.1
    assert rnnzoo_gru_regularized.default_params["temporal_dropout"] == 0.05
    assert rnnzoo_qrnn_longhorizon.default_params["horizon_loss_decay"] == 1.05
    assert rnnzoo_qrnn_longhorizon.default_params["loss"] == "huber"
