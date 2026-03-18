from foresight.models.registry import get_model_spec, list_models

MULTIVARIATE_STRATEGY_PRESET_KEYS = (
    "torch-stid-ema-multivariate",
    "torch-stgcn-swa-multivariate",
    "torch-graphwavenet-sam-multivariate",
    "torch-astgcn-regularized-multivariate",
    "torch-agcrn-longhorizon-multivariate",
    "torch-stemgnn-lookahead-multivariate",
)


def test_wave50_multivariate_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in MULTIVARIATE_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave50_multivariate_strategy_presets_are_torch_multivariate() -> None:
    for key in MULTIVARIATE_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "multivariate"
        assert "torch" in spec.requires


def test_wave50_multivariate_strategy_preset_defaults() -> None:
    stid_ema = get_model_spec("torch-stid-ema-multivariate")
    stgcn_swa = get_model_spec("torch-stgcn-swa-multivariate")
    graphwavenet_sam = get_model_spec("torch-graphwavenet-sam-multivariate")
    astgcn_regularized = get_model_spec("torch-astgcn-regularized-multivariate")
    agcrn_longhorizon = get_model_spec("torch-agcrn-longhorizon-multivariate")
    stemgnn_lookahead = get_model_spec("torch-stemgnn-lookahead-multivariate")

    assert stid_ema.default_params["ema_decay"] == 0.995
    assert stid_ema.default_params["scheduler"] == "cosine"
    assert stgcn_swa.default_params["swa_start_epoch"] == 18
    assert stgcn_swa.default_params["scheduler"] == "cosine_restarts"
    assert graphwavenet_sam.default_params["sam_rho"] == 0.05
    assert graphwavenet_sam.default_params["sam_adaptive"] is True
    assert astgcn_regularized.default_params["input_dropout"] == 0.1
    assert astgcn_regularized.default_params["temporal_dropout"] == 0.05
    assert agcrn_longhorizon.default_params["horizon_loss_decay"] == 1.05
    assert agcrn_longhorizon.default_params["loss"] == "huber"
    assert stemgnn_lookahead.default_params["lookahead_steps"] == 5
    assert stemgnn_lookahead.default_params["lookahead_alpha"] == 0.5
