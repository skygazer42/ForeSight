from foresight.models.registry import get_model_spec, list_models

GLOBAL_SEQ2SEQ_ATTN_GRU_STRATEGY_PRESET_KEYS = (
    "torch-seq2seq-attn-gru-ema-global",
    "torch-seq2seq-attn-gru-swa-global",
    "torch-seq2seq-attn-gru-sam-global",
    "torch-seq2seq-attn-gru-regularized-global",
    "torch-seq2seq-attn-gru-longhorizon-global",
    "torch-seq2seq-attn-gru-lookahead-global",
)


def test_wave92_global_seq2seq_attn_gru_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in GLOBAL_SEQ2SEQ_ATTN_GRU_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave92_global_seq2seq_attn_gru_strategy_presets_are_global_torch_optional() -> None:
    for key in GLOBAL_SEQ2SEQ_ATTN_GRU_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "global"
        assert "torch" in spec.requires


def test_wave92_global_seq2seq_attn_gru_strategy_preset_defaults() -> None:
    ema = get_model_spec("torch-seq2seq-attn-gru-ema-global")
    swa = get_model_spec("torch-seq2seq-attn-gru-swa-global")
    sam = get_model_spec("torch-seq2seq-attn-gru-sam-global")
    regularized = get_model_spec("torch-seq2seq-attn-gru-regularized-global")
    longhorizon = get_model_spec("torch-seq2seq-attn-gru-longhorizon-global")
    lookahead = get_model_spec("torch-seq2seq-attn-gru-lookahead-global")

    assert ema.default_params["cell"] == "gru"
    assert ema.default_params["attention"] == "bahdanau"
    assert ema.default_params["ema_decay"] == 0.995
    assert swa.default_params["cell"] == "gru"
    assert swa.default_params["attention"] == "bahdanau"
    assert swa.default_params["swa_start_epoch"] == 18
    assert sam.default_params["cell"] == "gru"
    assert sam.default_params["attention"] == "bahdanau"
    assert sam.default_params["sam_rho"] == 0.05
    assert sam.default_params["sam_adaptive"] is True
    assert regularized.default_params["input_dropout"] == 0.1
    assert regularized.default_params["temporal_dropout"] == 0.05
    assert longhorizon.default_params["loss"] == "huber"
    assert longhorizon.default_params["horizon_loss_decay"] == 1.05
    assert lookahead.default_params["lookahead_steps"] == 5
    assert lookahead.default_params["lookahead_alpha"] == 0.5
