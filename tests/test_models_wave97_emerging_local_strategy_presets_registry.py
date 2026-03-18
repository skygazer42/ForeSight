from foresight.models.registry import get_model_spec, list_models

EMERGING_LOCAL_STRATEGY_PRESET_KEYS = (
    "torch-pathformer-ema-direct",
    "torch-timemixer-swa-direct",
    "torch-tinytimemixer-sam-direct",
    "torch-basisformer-regularized-direct",
    "torch-witran-lookahead-direct",
    "torch-crossgnn-longhorizon-direct",
)


def test_wave97_emerging_local_strategy_presets_are_registered() -> None:
    keys = set(list_models())
    for key in EMERGING_LOCAL_STRATEGY_PRESET_KEYS:
        assert key in keys


def test_wave97_emerging_local_strategy_presets_are_local_torch_optional() -> None:
    for key in EMERGING_LOCAL_STRATEGY_PRESET_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires


def test_wave97_emerging_local_strategy_preset_defaults() -> None:
    pathformer_ema = get_model_spec("torch-pathformer-ema-direct")
    timemixer_swa = get_model_spec("torch-timemixer-swa-direct")
    tinytimemixer_sam = get_model_spec("torch-tinytimemixer-sam-direct")
    basisformer_regularized = get_model_spec("torch-basisformer-regularized-direct")
    witran_lookahead = get_model_spec("torch-witran-lookahead-direct")
    crossgnn_longhorizon = get_model_spec("torch-crossgnn-longhorizon-direct")

    assert pathformer_ema.default_params["expert_patch_lens"] == (4, 8, 16)
    assert pathformer_ema.default_params["ema_decay"] == 0.995
    assert timemixer_swa.default_params["multiscale_factors"] == (1, 2, 4)
    assert timemixer_swa.default_params["swa_start_epoch"] == 18
    assert tinytimemixer_sam.default_params["patch_len"] == 8
    assert tinytimemixer_sam.default_params["sam_rho"] == 0.05
    assert tinytimemixer_sam.default_params["sam_adaptive"] is True
    assert basisformer_regularized.default_params["num_bases"] == 16
    assert basisformer_regularized.default_params["dropout"] == 0.2
    assert witran_lookahead.default_params["grid_cols"] == 12
    assert witran_lookahead.default_params["lookahead_steps"] == 5
    assert witran_lookahead.default_params["lookahead_alpha"] == 0.5
    assert crossgnn_longhorizon.default_params["top_k"] == 8
    assert crossgnn_longhorizon.default_params["loss"] == "huber"
    assert crossgnn_longhorizon.default_params["horizon_loss_decay"] == 1.05
