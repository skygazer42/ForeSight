from foresight.models.registry import get_model_spec, list_models

GLOBAL_RNN_STRATEGY_PRESET_KEYS = (
    "torch-rnn-lstm-ema-global",
    "torch-rnn-gru-swa-global",
    "torch-rnn-lstm-sam-global",
    "torch-rnn-gru-regularized-global",
    "torch-rnn-encoder-longhorizon-global",
    "torch-rnn-encoder-lookahead-global",
)


def test_wave55_global_rnn_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in GLOBAL_RNN_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave55_global_rnn_strategy_presets_are_global_torch_optional() -> None:
    for key in GLOBAL_RNN_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "global"
        assert "torch" in spec.requires


def test_wave55_global_rnn_strategy_preset_defaults() -> None:
    lstm_ema = get_model_spec("torch-rnn-lstm-ema-global")
    gru_swa = get_model_spec("torch-rnn-gru-swa-global")
    lstm_sam = get_model_spec("torch-rnn-lstm-sam-global")
    gru_regularized = get_model_spec("torch-rnn-gru-regularized-global")
    encoder_longhorizon = get_model_spec("torch-rnn-encoder-longhorizon-global")
    encoder_lookahead = get_model_spec("torch-rnn-encoder-lookahead-global")

    assert lstm_ema.default_params["cell"] == "lstm"
    assert lstm_ema.default_params["ema_decay"] == 0.995
    assert gru_swa.default_params["cell"] == "gru"
    assert gru_swa.default_params["swa_start_epoch"] == 18
    assert lstm_sam.default_params["cell"] == "lstm"
    assert lstm_sam.default_params["sam_rho"] == 0.05
    assert lstm_sam.default_params["sam_adaptive"] is True
    assert gru_regularized.default_params["cell"] == "gru"
    assert gru_regularized.default_params["input_dropout"] == 0.1
    assert gru_regularized.default_params["temporal_dropout"] == 0.05
    assert encoder_longhorizon.default_params["hidden_size"] == 32
    assert encoder_longhorizon.default_params["loss"] == "huber"
    assert encoder_longhorizon.default_params["horizon_loss_decay"] == 1.05
    assert encoder_lookahead.default_params["hidden_size"] == 32
    assert encoder_lookahead.default_params["lookahead_steps"] == 5
    assert encoder_lookahead.default_params["lookahead_alpha"] == 0.5
