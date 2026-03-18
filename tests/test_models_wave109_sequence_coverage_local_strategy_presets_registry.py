from foresight.models.registry import get_model_spec, list_models

SEQUENCE_COVERAGE_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-patchtst-longhorizon-direct",
    "torch-retnet-regularized-direct",
    "torch-timesnet-ema-direct",
    "torch-seq2seq-attn-gru-sam-direct",
    "torch-seq2seq-attn-lstm-regularized-direct",
    "torch-seq2seq-lstm-lookahead-direct",
)


def test_wave109_sequence_coverage_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in SEQUENCE_COVERAGE_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave109_sequence_coverage_local_strategy_presets_are_local_torch_optional() -> None:
    for key in SEQUENCE_COVERAGE_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave109_sequence_coverage_local_strategy_preset_defaults() -> None:
    patchtst_longhorizon = get_model_spec("torch-patchtst-longhorizon-direct")
    retnet_regularized = get_model_spec("torch-retnet-regularized-direct")
    timesnet_ema = get_model_spec("torch-timesnet-ema-direct")
    seq2seq_attn_gru_sam = get_model_spec("torch-seq2seq-attn-gru-sam-direct")
    seq2seq_attn_lstm_regularized = get_model_spec("torch-seq2seq-attn-lstm-regularized-direct")
    seq2seq_lstm_lookahead = get_model_spec("torch-seq2seq-lstm-lookahead-direct")

    assert patchtst_longhorizon.default_params["d_model"] == 64
    assert patchtst_longhorizon.default_params["loss"] == "huber"
    assert patchtst_longhorizon.default_params["horizon_loss_decay"] == 1.05
    assert patchtst_longhorizon.default_params["ema_decay"] == 0.99
    assert retnet_regularized.default_params["d_model"] == 64
    assert retnet_regularized.default_params["dropout"] == 0.2
    assert timesnet_ema.default_params["d_model"] == 64
    assert timesnet_ema.default_params["ema_decay"] == 0.995
    assert seq2seq_attn_gru_sam.default_params["cell"] == "gru"
    assert seq2seq_attn_gru_sam.default_params["sam_rho"] == 0.05
    assert seq2seq_attn_gru_sam.default_params["sam_adaptive"] is True
    assert seq2seq_attn_lstm_regularized.default_params["cell"] == "lstm"
    assert seq2seq_attn_lstm_regularized.default_params["dropout"] == 0.2
    assert seq2seq_lstm_lookahead.default_params["cell"] == "lstm"
    assert seq2seq_lstm_lookahead.default_params["lookahead_steps"] == 5
    assert seq2seq_lstm_lookahead.default_params["lookahead_alpha"] == 0.5
