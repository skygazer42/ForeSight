from foresight.models.registry import get_model_spec, list_models

MAINSTREAM_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-patchtst-swa-direct",
    "torch-retnet-sam-direct",
    "torch-timesnet-regularized-direct",
    "torch-seq2seq-attn-gru-ema-direct",
    "torch-seq2seq-attn-lstm-lookahead-direct",
    "torch-tft-regularized-direct",
)


def test_wave102_mainstream_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in MAINSTREAM_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave102_mainstream_local_strategy_presets_are_local_torch_optional() -> None:
    for key in MAINSTREAM_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave102_mainstream_local_strategy_preset_defaults() -> None:
    patchtst_swa = get_model_spec("torch-patchtst-swa-direct")
    retnet_sam = get_model_spec("torch-retnet-sam-direct")
    timesnet_regularized = get_model_spec("torch-timesnet-regularized-direct")
    seq2seq_attn_gru_ema = get_model_spec("torch-seq2seq-attn-gru-ema-direct")
    seq2seq_attn_lstm_lookahead = get_model_spec("torch-seq2seq-attn-lstm-lookahead-direct")
    tft_regularized = get_model_spec("torch-tft-regularized-direct")

    assert patchtst_swa.default_params["patch_len"] == 16
    assert patchtst_swa.default_params["swa_start_epoch"] == 18
    assert retnet_sam.default_params["ffn_dim"] == 128
    assert retnet_sam.default_params["sam_rho"] == 0.05
    assert retnet_sam.default_params["sam_adaptive"] is True
    assert timesnet_regularized.default_params["top_k"] == 3
    assert timesnet_regularized.default_params["dropout"] == 0.2
    assert seq2seq_attn_gru_ema.default_params["cell"] == "gru"
    assert seq2seq_attn_gru_ema.default_params["attention"] == "bahdanau"
    assert seq2seq_attn_gru_ema.default_params["ema_decay"] == 0.995
    assert seq2seq_attn_lstm_lookahead.default_params["cell"] == "lstm"
    assert seq2seq_attn_lstm_lookahead.default_params["attention"] == "bahdanau"
    assert seq2seq_attn_lstm_lookahead.default_params["lookahead_steps"] == 5
    assert seq2seq_attn_lstm_lookahead.default_params["lookahead_alpha"] == 0.5
    assert tft_regularized.default_params["lstm_layers"] == 1
    assert tft_regularized.default_params["weight_decay"] == 5e-4
    assert tft_regularized.default_params["dropout"] == 0.2
