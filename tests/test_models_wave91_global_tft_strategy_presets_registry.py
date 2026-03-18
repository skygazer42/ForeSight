from foresight.models.registry import get_model_spec, list_models

GLOBAL_TFT_STRATEGY_PRESET_KEYS = (
    "torch-tft-ema-global",
    "torch-tft-swa-global",
    "torch-tft-sam-global",
    "torch-tft-regularized-global",
    "torch-tft-longhorizon-global",
    "torch-tft-lookahead-global",
)


def test_wave91_global_tft_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in GLOBAL_TFT_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave91_global_tft_strategy_presets_are_global_torch_optional() -> None:
    for key in GLOBAL_TFT_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "global"
        assert "torch" in spec.requires


def test_wave91_global_tft_strategy_preset_defaults() -> None:
    ema = get_model_spec("torch-tft-ema-global")
    swa = get_model_spec("torch-tft-swa-global")
    sam = get_model_spec("torch-tft-sam-global")
    regularized = get_model_spec("torch-tft-regularized-global")
    longhorizon = get_model_spec("torch-tft-longhorizon-global")
    lookahead = get_model_spec("torch-tft-lookahead-global")

    assert ema.default_params["ema_decay"] == 0.995
    assert ema.default_params["scheduler"] == "cosine"
    assert swa.default_params["swa_start_epoch"] == 18
    assert swa.default_params["scheduler"] == "cosine_restarts"
    assert sam.default_params["sam_rho"] == 0.05
    assert sam.default_params["sam_adaptive"] is True
    assert regularized.default_params["input_dropout"] == 0.1
    assert regularized.default_params["temporal_dropout"] == 0.05
    assert longhorizon.default_params["loss"] == "huber"
    assert longhorizon.default_params["horizon_loss_decay"] == 1.05
    assert lookahead.default_params["lookahead_steps"] == 5
    assert lookahead.default_params["lookahead_alpha"] == 0.5
