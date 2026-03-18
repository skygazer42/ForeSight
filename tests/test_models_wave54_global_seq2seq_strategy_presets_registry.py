from foresight.models.registry import get_model_spec, list_models

GLOBAL_SEQ2SEQ_STRATEGY_PRESET_KEYS = (
    "torch-seq2seq-lstm-ema-global",
    "torch-seq2seq-gru-swa-global",
    "torch-seq2seq-attn-lstm-sam-global",
    "torch-seq2seq-attn-gru-regularized-global",
    "torch-seq2seq-lstm-longhorizon-global",
    "torch-seq2seq-attn-lstm-lookahead-global",
)


def test_wave54_global_seq2seq_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in GLOBAL_SEQ2SEQ_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave54_global_seq2seq_strategy_presets_are_global_torch_optional() -> None:
    for key in GLOBAL_SEQ2SEQ_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "global"
        assert "torch" in spec.requires


def test_wave54_global_seq2seq_strategy_preset_defaults() -> None:
    lstm_ema = get_model_spec("torch-seq2seq-lstm-ema-global")
    gru_swa = get_model_spec("torch-seq2seq-gru-swa-global")
    attn_lstm_sam = get_model_spec("torch-seq2seq-attn-lstm-sam-global")
    attn_gru_regularized = get_model_spec("torch-seq2seq-attn-gru-regularized-global")
    lstm_longhorizon = get_model_spec("torch-seq2seq-lstm-longhorizon-global")
    attn_lstm_lookahead = get_model_spec("torch-seq2seq-attn-lstm-lookahead-global")

    assert lstm_ema.default_params["cell"] == "lstm"
    assert lstm_ema.default_params["attention"] == "none"
    assert lstm_ema.default_params["ema_decay"] == 0.995
    assert gru_swa.default_params["cell"] == "gru"
    assert gru_swa.default_params["attention"] == "none"
    assert gru_swa.default_params["swa_start_epoch"] == 18
    assert attn_lstm_sam.default_params["cell"] == "lstm"
    assert attn_lstm_sam.default_params["attention"] == "bahdanau"
    assert attn_lstm_sam.default_params["sam_rho"] == 0.05
    assert attn_lstm_sam.default_params["sam_adaptive"] is True
    assert attn_gru_regularized.default_params["cell"] == "gru"
    assert attn_gru_regularized.default_params["attention"] == "bahdanau"
    assert attn_gru_regularized.default_params["input_dropout"] == 0.1
    assert attn_gru_regularized.default_params["temporal_dropout"] == 0.05
    assert lstm_longhorizon.default_params["cell"] == "lstm"
    assert lstm_longhorizon.default_params["loss"] == "huber"
    assert lstm_longhorizon.default_params["horizon_loss_decay"] == 1.05
    assert attn_lstm_lookahead.default_params["cell"] == "lstm"
    assert attn_lstm_lookahead.default_params["attention"] == "bahdanau"
    assert attn_lstm_lookahead.default_params["lookahead_steps"] == 5
    assert attn_lstm_lookahead.default_params["lookahead_alpha"] == 0.5
