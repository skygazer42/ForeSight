from foresight.models.registry import get_model_spec, list_models

SEQUENCE_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-patchtst-lookahead-direct",
    "torch-retnet-swa-direct",
    "torch-timesnet-lookahead-direct",
    "torch-seq2seq-gru-regularized-direct",
    "torch-seq2seq-gru-sam-direct",
    "torch-seq2seq-attn-lstm-ema-direct",
)


def test_wave110_sequence_refinement_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in SEQUENCE_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave110_sequence_refinement_local_strategy_presets_are_local_torch_optional() -> None:
    for key in SEQUENCE_REFINEMENT_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave110_sequence_refinement_local_strategy_preset_defaults() -> None:
    patchtst_lookahead = get_model_spec("torch-patchtst-lookahead-direct")
    retnet_swa = get_model_spec("torch-retnet-swa-direct")
    timesnet_lookahead = get_model_spec("torch-timesnet-lookahead-direct")
    seq2seq_gru_regularized = get_model_spec("torch-seq2seq-gru-regularized-direct")
    seq2seq_gru_sam = get_model_spec("torch-seq2seq-gru-sam-direct")
    seq2seq_attn_lstm_ema = get_model_spec("torch-seq2seq-attn-lstm-ema-direct")

    assert patchtst_lookahead.default_params["d_model"] == 64
    assert patchtst_lookahead.default_params["lookahead_steps"] == 5
    assert patchtst_lookahead.default_params["lookahead_alpha"] == 0.5
    assert retnet_swa.default_params["d_model"] == 64
    assert retnet_swa.default_params["swa_start_epoch"] == 18
    assert timesnet_lookahead.default_params["d_model"] == 64
    assert timesnet_lookahead.default_params["lookahead_steps"] == 5
    assert timesnet_lookahead.default_params["lookahead_alpha"] == 0.5
    assert seq2seq_gru_regularized.default_params["cell"] == "gru"
    assert seq2seq_gru_regularized.default_params["dropout"] == 0.2
    assert seq2seq_gru_sam.default_params["cell"] == "gru"
    assert seq2seq_gru_sam.default_params["sam_rho"] == 0.05
    assert seq2seq_gru_sam.default_params["sam_adaptive"] is True
    assert seq2seq_attn_lstm_ema.default_params["cell"] == "lstm"
    assert seq2seq_attn_lstm_ema.default_params["attention"] == "bahdanau"
    assert seq2seq_attn_lstm_ema.default_params["ema_decay"] == 0.995
