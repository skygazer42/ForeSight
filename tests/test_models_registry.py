from pathlib import Path

from foresight.base import (
    BaseForecaster as RuntimeBaseForecaster,
)
from foresight.base import (
    BaseGlobalForecaster as RuntimeBaseGlobalForecaster,
)
from foresight.base import (
    RegistryForecaster as BaseRegistryForecaster,
)
from foresight.base import (
    RegistryGlobalForecaster as BaseRegistryGlobalForecaster,
)
from foresight.models.registry import ModelSpec, get_model_spec, list_models
from foresight.models.factories import build_local_forecaster
from foresight.models.specs import (
    LocalForecasterFn as RuntimeLocalForecasterFn,
)
from foresight.models.specs import (
    ModelFactory as RuntimeModelFactory,
)
from foresight.models.specs import ModelSpec as RuntimeModelSpec

WAVE1A_TORCH_LOCAL_KEYS = (
    "torch-informer-direct",
    "torch-autoformer-direct",
    "torch-nonstationary-transformer-direct",
    "torch-fedformer-direct",
    "torch-itransformer-direct",
    "torch-timesnet-direct",
    "torch-tft-direct",
    "torch-timemixer-direct",
    "torch-timemixer-deep-direct",
    "torch-timemixer-wide-direct",
    "torch-sparsetsf-direct",
)

LIGHTWEIGHT_TORCH_LOCAL_KEYS = (
    "torch-lightts-direct",
    "torch-frets-direct",
    "torch-frets-deep-direct",
    "torch-frets-wide-direct",
)

DECOMP_TORCH_LOCAL_KEYS = (
    "torch-film-direct",
    "torch-film-deep-direct",
    "torch-film-wide-direct",
    "torch-micn-direct",
    "torch-micn-deep-direct",
    "torch-micn-wide-direct",
)

WAVE1B_TORCH_LOCAL_KEYS = (
    "torch-koopa-direct",
    "torch-koopa-deep-direct",
    "torch-koopa-wide-direct",
    "torch-samformer-direct",
    "torch-samformer-deep-direct",
    "torch-samformer-wide-direct",
)

RETENTION_TORCH_LOCAL_KEYS = (
    "torch-retnet-direct",
    "torch-retnet-recursive",
)
RETENTION_TORCH_GLOBAL_KEYS = ("torch-retnet-global",)

TIME_XER_TORCH_KEYS = (
    "torch-timexer-direct",
    "torch-timexer-global",
)

STATE_SPACE_TORCH_LOCAL_KEYS = (
    "torch-lmu-direct",
    "torch-s4d-direct",
)

CT_RNN_TORCH_LOCAL_KEYS = (
    "torch-ltc-direct",
    "torch-cfc-direct",
)

REVIVAL_TORCH_LOCAL_KEYS = (
    "torch-xlstm-direct",
    "torch-mamba2-direct",
)

SSM_TORCH_LOCAL_KEYS = (
    "torch-s4-direct",
    "torch-s5-direct",
)

RECURRENT_REVIVAL_TORCH_LOCAL_KEYS = (
    "torch-griffin-direct",
    "torch-hawk-direct",
)

LATENT_TORCH_LOCAL_KEYS = (
    "torch-perceiver-direct",
    "torch-perceiver-deep-direct",
    "torch-perceiver-wide-direct",
)

SEGMENTED_TORCH_LOCAL_KEYS = (
    "torch-segrnn-direct",
    "torch-segrnn-deep-direct",
    "torch-segrnn-wide-direct",
)

MODERN_CONV_TORCH_LOCAL_KEYS = (
    "torch-moderntcn-direct",
    "torch-moderntcn-deep-direct",
    "torch-moderntcn-wide-direct",
)

BASIS_TORCH_LOCAL_KEYS = (
    "torch-basisformer-direct",
    "torch-basisformer-deep-direct",
    "torch-basisformer-wide-direct",
)

GRID_RECURRENT_TORCH_LOCAL_KEYS = (
    "torch-witran-direct",
    "torch-witran-deep-direct",
    "torch-witran-wide-direct",
)

LAG_GRAPH_TORCH_LOCAL_KEYS = (
    "torch-crossgnn-direct",
    "torch-crossgnn-deep-direct",
    "torch-crossgnn-wide-direct",
)

MULTISCALE_ROUTING_TORCH_LOCAL_KEYS = (
    "torch-pathformer-direct",
    "torch-pathformer-deep-direct",
    "torch-pathformer-wide-direct",
)

PATCH_SSM_TORCH_LOCAL_KEYS = (
    "torch-timesmamba-direct",
    "torch-timesmamba-deep-direct",
    "torch-timesmamba-wide-direct",
)

PATCH_MIXER_TORCH_LOCAL_KEYS = (
    "torch-tinytimemixer-direct",
    "torch-tinytimemixer-deep-direct",
    "torch-tinytimemixer-wide-direct",
)

FREQUENCY_INTERPOLATION_TORCH_LOCAL_KEYS = (
    "torch-fits-direct",
    "torch-fits-deep-direct",
    "torch-fits-wide-direct",
)

SEQUENCE_PRESET_TORCH_LOCAL_KEYS = (
    "torch-fnet-deep-direct",
    "torch-fnet-wide-direct",
    "torch-gmlp-deep-direct",
    "torch-gmlp-wide-direct",
    "torch-linear-attn-deep-direct",
    "torch-linear-attn-wide-direct",
    "torch-inception-deep-direct",
    "torch-inception-wide-direct",
    "torch-mamba-deep-direct",
    "torch-mamba-wide-direct",
    "torch-rwkv-deep-direct",
    "torch-rwkv-wide-direct",
    "torch-hyena-deep-direct",
    "torch-hyena-wide-direct",
    "torch-dilated-rnn-deep-direct",
    "torch-dilated-rnn-wide-direct",
)

CAPACITY_PRESET_TORCH_LOCAL_KEYS = (
    "torch-kan-deep-direct",
    "torch-kan-wide-direct",
    "torch-scinet-deep-direct",
    "torch-scinet-wide-direct",
    "torch-etsformer-deep-direct",
    "torch-etsformer-wide-direct",
    "torch-esrnn-deep-direct",
    "torch-esrnn-wide-direct",
    "torch-patchtst-deep-direct",
    "torch-patchtst-wide-direct",
    "torch-crossformer-deep-direct",
    "torch-crossformer-wide-direct",
    "torch-pyraformer-deep-direct",
    "torch-pyraformer-wide-direct",
    "torch-tsmixer-deep-direct",
    "torch-tsmixer-wide-direct",
    "torch-nhits-deep-direct",
    "torch-nhits-wide-direct",
)

FOUNDATION_PRESET_TORCH_LOCAL_KEYS = (
    "torch-mlp-deep-direct",
    "torch-mlp-wide-direct",
    "torch-lstm-deep-direct",
    "torch-lstm-wide-direct",
    "torch-gru-deep-direct",
    "torch-gru-wide-direct",
    "torch-tcn-deep-direct",
    "torch-tcn-wide-direct",
    "torch-nbeats-deep-direct",
    "torch-nbeats-wide-direct",
    "torch-transformer-deep-direct",
    "torch-transformer-wide-direct",
    "torch-cnn-deep-direct",
    "torch-cnn-wide-direct",
    "torch-resnet1d-deep-direct",
    "torch-resnet1d-wide-direct",
    "torch-wavenet-deep-direct",
    "torch-wavenet-wide-direct",
)

RECURRENT_STATEFUL_PRESET_TORCH_LOCAL_KEYS = (
    "torch-bilstm-deep-direct",
    "torch-bilstm-wide-direct",
    "torch-bigru-deep-direct",
    "torch-bigru-wide-direct",
    "torch-attn-gru-deep-direct",
    "torch-attn-gru-wide-direct",
    "torch-lmu-deep-direct",
    "torch-lmu-wide-direct",
    "torch-ltc-deep-direct",
    "torch-ltc-wide-direct",
    "torch-cfc-deep-direct",
    "torch-cfc-wide-direct",
    "torch-xlstm-deep-direct",
    "torch-xlstm-wide-direct",
    "torch-griffin-deep-direct",
    "torch-griffin-wide-direct",
    "torch-hawk-deep-direct",
    "torch-hawk-wide-direct",
)

TRANSFORMER_SSM_PRESET_TORCH_LOCAL_KEYS = (
    "torch-informer-deep-direct",
    "torch-informer-wide-direct",
    "torch-autoformer-deep-direct",
    "torch-autoformer-wide-direct",
    "torch-nonstationary-transformer-deep-direct",
    "torch-nonstationary-transformer-wide-direct",
    "torch-fedformer-deep-direct",
    "torch-fedformer-wide-direct",
    "torch-itransformer-deep-direct",
    "torch-itransformer-wide-direct",
    "torch-timesnet-deep-direct",
    "torch-timesnet-wide-direct",
    "torch-tft-deep-direct",
    "torch-tft-wide-direct",
    "torch-timexer-deep-direct",
    "torch-timexer-wide-direct",
    "torch-s4d-deep-direct",
    "torch-s4d-wide-direct",
    "torch-s4-deep-direct",
    "torch-s4-wide-direct",
    "torch-s5-deep-direct",
    "torch-s5-wide-direct",
    "torch-mamba2-deep-direct",
    "torch-mamba2-wide-direct",
)

REMAINING_BASELINE_PRESET_TORCH_LOCAL_KEYS = (
    "torch-retnet-deep-direct",
    "torch-retnet-wide-direct",
    "torch-lightts-long-direct",
    "torch-lightts-wide-direct",
    "torch-sparsetsf-long-direct",
    "torch-sparsetsf-wide-direct",
    "torch-tide-long-direct",
    "torch-tide-wide-direct",
    "torch-nlinear-long-direct",
    "torch-dlinear-long-direct",
)

LOCAL_LSTNET_PRESET_TORCH_LOCAL_KEYS = (
    "torch-lstnet-long-direct",
    "torch-lstnet-wide-direct",
)

STRUCTURED_RNN_PRESET_TORCH_LOCAL_KEYS = (
    "torch-multidim-rnn-long-direct",
    "torch-multidim-rnn-wide-direct",
    "torch-grid-lstm-long-direct",
    "torch-grid-lstm-wide-direct",
    "torch-structural-rnn-long-direct",
    "torch-structural-rnn-wide-direct",
)

RECURSIVE_RNN_PRESET_TORCH_LOCAL_KEYS = (
    "torch-deepar-deep-recursive",
    "torch-deepar-wide-recursive",
    "torch-qrnn-deep-recursive",
    "torch-qrnn-wide-recursive",
)

PROBABILISTIC_PRESET_TORCH_LOCAL_KEYS = (
    "torch-timegrad-long-direct",
    "torch-timegrad-wide-direct",
    "torch-tactis-long-direct",
    "torch-tactis-wide-direct",
)

RESERVOIR_PRESET_TORCH_LOCAL_KEYS = (
    "torch-esn-long-direct",
    "torch-esn-wide-direct",
    "torch-deep-esn-long-direct",
    "torch-deep-esn-wide-direct",
    "torch-liquid-state-long-direct",
    "torch-liquid-state-wide-direct",
)

RETNET_RECURSIVE_PRESET_TORCH_LOCAL_KEYS = (
    "torch-retnet-deep-recursive",
    "torch-retnet-wide-recursive",
)

CANONICAL_RNN_ZOO_PRESET_TORCH_LOCAL_KEYS = (
    "torch-rnnpaper-lstm-long-direct",
    "torch-rnnpaper-lstm-wide-direct",
    "torch-rnnpaper-gru-long-direct",
    "torch-rnnpaper-gru-wide-direct",
    "torch-rnnpaper-qrnn-long-direct",
    "torch-rnnpaper-qrnn-wide-direct",
    "torch-rnnzoo-lstm-long-direct",
    "torch-rnnzoo-lstm-wide-direct",
    "torch-rnnzoo-gru-long-direct",
    "torch-rnnzoo-gru-wide-direct",
    "torch-rnnzoo-qrnn-long-direct",
    "torch-rnnzoo-qrnn-wide-direct",
)

GLOBAL_PRESET_TORCH_KEYS = (
    "torch-tft-deep-global",
    "torch-tft-wide-global",
    "torch-timexer-deep-global",
    "torch-timexer-wide-global",
    "torch-retnet-deep-global",
    "torch-retnet-wide-global",
    "torch-informer-deep-global",
    "torch-informer-wide-global",
    "torch-autoformer-deep-global",
    "torch-autoformer-wide-global",
    "torch-fedformer-deep-global",
    "torch-fedformer-wide-global",
    "torch-nonstationary-transformer-deep-global",
    "torch-nonstationary-transformer-wide-global",
    "torch-patchtst-deep-global",
    "torch-patchtst-wide-global",
    "torch-crossformer-deep-global",
    "torch-crossformer-wide-global",
    "torch-pyraformer-deep-global",
    "torch-pyraformer-wide-global",
    "torch-itransformer-deep-global",
    "torch-itransformer-wide-global",
    "torch-timesnet-deep-global",
    "torch-timesnet-wide-global",
    "torch-tsmixer-deep-global",
    "torch-tsmixer-wide-global",
    "torch-nbeats-deep-global",
    "torch-nbeats-wide-global",
    "torch-nhits-deep-global",
    "torch-nhits-wide-global",
    "torch-tcn-deep-global",
    "torch-tcn-wide-global",
    "torch-wavenet-deep-global",
    "torch-wavenet-wide-global",
    "torch-resnet1d-deep-global",
    "torch-resnet1d-wide-global",
    "torch-inception-deep-global",
    "torch-inception-wide-global",
    "torch-kan-deep-global",
    "torch-kan-wide-global",
    "torch-scinet-deep-global",
    "torch-scinet-wide-global",
    "torch-etsformer-deep-global",
    "torch-etsformer-wide-global",
    "torch-esrnn-deep-global",
    "torch-esrnn-wide-global",
    "torch-fnet-deep-global",
    "torch-fnet-wide-global",
    "torch-gmlp-deep-global",
    "torch-gmlp-wide-global",
    "torch-ssm-deep-global",
    "torch-ssm-wide-global",
    "torch-mamba-deep-global",
    "torch-mamba-wide-global",
    "torch-rwkv-deep-global",
    "torch-rwkv-wide-global",
    "torch-hyena-deep-global",
    "torch-hyena-wide-global",
    "torch-dilated-rnn-deep-global",
    "torch-dilated-rnn-wide-global",
    "torch-transformer-encdec-deep-global",
    "torch-transformer-encdec-wide-global",
    "torch-seq2seq-lstm-deep-global",
    "torch-seq2seq-lstm-wide-global",
    "torch-seq2seq-gru-deep-global",
    "torch-seq2seq-gru-wide-global",
    "torch-seq2seq-attn-lstm-deep-global",
    "torch-seq2seq-attn-lstm-wide-global",
    "torch-seq2seq-attn-gru-deep-global",
    "torch-seq2seq-attn-gru-wide-global",
    "torch-deepar-deep-global",
    "torch-deepar-wide-global",
    "torch-tide-long-global",
    "torch-tide-wide-global",
    "torch-nlinear-long-global",
    "torch-dlinear-long-global",
    "torch-rnn-lstm-deep-global",
    "torch-rnn-lstm-wide-global",
    "torch-rnn-gru-deep-global",
    "torch-rnn-gru-wide-global",
    "torch-rnn-encoder-deep-global",
    "torch-rnn-encoder-wide-global",
    "torch-lstnet-long-global",
    "torch-lstnet-wide-global",
)

TORCH_MULTIVARIATE_KEYS = (
    "torch-stid-multivariate",
    "torch-stgcn-multivariate",
    "torch-graphwavenet-multivariate",
)

TRANSFORMERS_LOCAL_KEYS = ("hf-timeseries-transformer-direct",)


def test_resolution_module_preserves_registry_lookup_surface() -> None:
    from foresight.models import resolution as model_resolution

    spec = model_resolution.get_model_spec("naive-last")
    assert spec.key == get_model_spec("naive-last").key
    assert model_resolution.list_models() == list_models()
    forecaster = build_local_forecaster(key=spec.key, spec=spec, params={})
    assert forecaster([1.0, 2.0, 3.0], 1).shape == (1,)


def test_registry_module_re_exports_registry_storage_for_compatibility() -> None:
    from foresight.models import registry as model_registry
    from foresight.models import resolution as model_resolution

    assert model_registry._REGISTRY is model_resolution._REGISTRY


def test_runtime_module_exposes_catalog_factory_context() -> None:
    from foresight.models import runtime as model_runtime

    assert callable(model_runtime._factory_naive_last)


def test_list_models_contains_expected_keys():
    keys = set(list_models())
    assert "naive-last" in keys
    assert "seasonal-naive" in keys
    assert "weighted-moving-average" in keys
    assert "moving-median" in keys
    assert "seasonal-drift" in keys


def test_wave1a_torch_local_models_are_registered():
    keys = set(list_models())
    for key in WAVE1A_TORCH_LOCAL_KEYS:
        assert key in keys


def test_lightweight_torch_local_models_are_registered():
    keys = set(list_models())
    for key in LIGHTWEIGHT_TORCH_LOCAL_KEYS:
        assert key in keys


def test_decomposition_torch_local_models_are_registered():
    keys = set(list_models())
    for key in DECOMP_TORCH_LOCAL_KEYS:
        assert key in keys


def test_wave1b_torch_local_models_are_registered():
    keys = set(list_models())
    for key in WAVE1B_TORCH_LOCAL_KEYS:
        assert key in keys


def test_retnet_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in RETENTION_TORCH_LOCAL_KEYS:
        assert key in keys


def test_retnet_torch_global_models_are_registered() -> None:
    keys = set(list_models())
    for key in RETENTION_TORCH_GLOBAL_KEYS:
        assert key in keys


def test_timexer_torch_models_are_registered() -> None:
    keys = set(list_models())
    for key in TIME_XER_TORCH_KEYS:
        assert key in keys


def test_state_space_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in STATE_SPACE_TORCH_LOCAL_KEYS:
        assert key in keys


def test_continuous_time_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in CT_RNN_TORCH_LOCAL_KEYS:
        assert key in keys


def test_revival_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in REVIVAL_TORCH_LOCAL_KEYS:
        assert key in keys


def test_ssm_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in SSM_TORCH_LOCAL_KEYS:
        assert key in keys


def test_recurrent_revival_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in RECURRENT_REVIVAL_TORCH_LOCAL_KEYS:
        assert key in keys


def test_latent_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in LATENT_TORCH_LOCAL_KEYS:
        assert key in keys


def test_segmented_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in SEGMENTED_TORCH_LOCAL_KEYS:
        assert key in keys


def test_modern_conv_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in MODERN_CONV_TORCH_LOCAL_KEYS:
        assert key in keys


def test_basis_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in BASIS_TORCH_LOCAL_KEYS:
        assert key in keys


def test_grid_recurrent_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in GRID_RECURRENT_TORCH_LOCAL_KEYS:
        assert key in keys


def test_lag_graph_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in LAG_GRAPH_TORCH_LOCAL_KEYS:
        assert key in keys


def test_multiscale_routing_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in MULTISCALE_ROUTING_TORCH_LOCAL_KEYS:
        assert key in keys


def test_patch_ssm_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in PATCH_SSM_TORCH_LOCAL_KEYS:
        assert key in keys


def test_patch_mixer_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in PATCH_MIXER_TORCH_LOCAL_KEYS:
        assert key in keys


def test_frequency_interpolation_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in FREQUENCY_INTERPOLATION_TORCH_LOCAL_KEYS:
        assert key in keys


def test_sequence_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in SEQUENCE_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_capacity_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in CAPACITY_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_foundation_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in FOUNDATION_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_recurrent_stateful_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in RECURRENT_STATEFUL_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_transformer_ssm_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in TRANSFORMER_SSM_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_remaining_baseline_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in REMAINING_BASELINE_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_local_lstnet_preset_torch_models_are_registered() -> None:
    keys = set(list_models())
    for key in LOCAL_LSTNET_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_structured_rnn_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in STRUCTURED_RNN_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_recursive_rnn_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in RECURSIVE_RNN_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_probabilistic_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in PROBABILISTIC_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_reservoir_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in RESERVOIR_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_retnet_recursive_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in RETNET_RECURSIVE_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_canonical_rnn_zoo_preset_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in CANONICAL_RNN_ZOO_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_global_preset_torch_models_are_registered() -> None:
    keys = set(list_models())
    for key in GLOBAL_PRESET_TORCH_KEYS:
        assert key in keys


def test_torch_local_catalog_exposes_deduplicated_param_help_strings() -> None:
    perceiver = get_model_spec("torch-perceiver-direct")
    segrnn = get_model_spec("torch-segrnn-direct")
    moderntcn = get_model_spec("torch-moderntcn-direct")
    basisformer = get_model_spec("torch-basisformer-direct")
    witran = get_model_spec("torch-witran-direct")
    pathformer = get_model_spec("torch-pathformer-direct")
    timesmamba = get_model_spec("torch-timesmamba-direct")
    tinytimemixer = get_model_spec("torch-tinytimemixer-direct")
    fits = get_model_spec("torch-fits-direct")

    assert perceiver.param_help["d_model"] == "Latent and token embedding dimension"
    assert perceiver.param_help["latent_len"] == "Number of learned latent tokens"
    assert segrnn.param_help["segment_len"] == "Segment length used to chunk the lag window"
    assert (
        segrnn.param_help["dropout"]
        == "Dropout probability in [0,1) (only if num_layers>1 inside GRU)"
    )
    assert moderntcn.param_help["num_blocks"] == "Number of ModernTCN mixer blocks"
    assert (
        moderntcn.param_help["kernel_size"]
        == "Odd depthwise convolution kernel size over patch tokens"
    )
    assert basisformer.param_help["lags"] == "Lag window length"
    assert (
        basisformer.param_help["num_bases"] == "Number of learned basis vectors in the dictionary"
    )
    assert basisformer.param_help["dim_feedforward"] == "Feed-forward dimension in the encoder"
    assert witran.param_help["grid_cols"] == "Number of columns in the 2D lag grid"
    assert witran.param_help["d_model"] == "Cell embedding / decoder dimension"
    assert (
        witran.param_help["hidden_size"] == "Hidden size of the coupled row/column recurrent states"
    )
    assert witran.param_help["nhead"] == "Number of decoder attention heads"
    assert witran.param_help["num_layers"] == "Number of stacked grid recurrent blocks"
    assert pathformer.param_help["d_model"] == "Shared expert/context embedding dimension"
    assert pathformer.param_help["expert_patch_lens"] == "Patch lengths for the multi-scale experts"
    assert pathformer.param_help["num_blocks"] == "Number of routing blocks"
    assert pathformer.param_help["top_k"] == "Top-k experts selected by the router per block"
    assert (
        timesmamba.param_help["state_size"]
        == "Latent state size in the recurrent state-space mixer"
    )
    assert timesmamba.param_help["num_blocks"] == "Number of stacked state-space mixer blocks"
    assert tinytimemixer.param_help["patch_len"] == "Number of timesteps per patch token"
    assert tinytimemixer.param_help["d_model"] == "Patch embedding dimension"
    assert tinytimemixer.param_help["num_blocks"] == "Number of stacked patch mixer blocks"
    assert (
        fits.param_help["low_freq_bins"]
        == "Number of low-frequency FFT bins retained before interpolation"
    )
    assert fits.param_help["hidden_size"] == "Hidden width of the frequency interpolation MLP"
    assert fits.param_help["num_layers"] == "Number of frequency interpolation blocks"


def test_torch_catalog_exposes_wave1_trainer_defaults_and_help() -> None:
    local = get_model_spec("torch-mlp-direct")
    global_model = get_model_spec("torch-timexer-global")

    for spec in (local, global_model):
        assert spec.default_params.get("min_epochs") == 1
        assert spec.default_params.get("amp") is False
        assert spec.default_params.get("amp_dtype") == "auto"
        assert spec.default_params.get("warmup_epochs") == 0
        assert spec.default_params.get("min_lr") == 0.0
        assert spec.default_params.get("scheduler_restart_period") == 10
        assert spec.default_params.get("scheduler_restart_mult") == 1
        assert spec.default_params.get("scheduler_pct_start") == 0.3
        assert spec.default_params.get("grad_accum_steps") == 1
        assert spec.default_params.get("monitor") == "auto"
        assert spec.default_params.get("monitor_mode") == "min"
        assert spec.default_params.get("min_delta") == 0.0
        assert spec.default_params.get("num_workers") == 0
        assert spec.default_params.get("pin_memory") is False
        assert spec.default_params.get("persistent_workers") is False
        assert spec.default_params.get("scheduler_patience") == 5
        assert spec.default_params.get("grad_clip_mode") == "norm"
        assert spec.default_params.get("grad_clip_value") == 0.0
        assert spec.default_params.get("scheduler_plateau_factor") == 0.1
        assert spec.default_params.get("scheduler_plateau_threshold") == 1e-4
        assert spec.default_params.get("ema_decay") == 0.0
        assert spec.default_params.get("ema_warmup_epochs") == 0
        assert spec.default_params.get("swa_start_epoch") == -1
        assert spec.default_params.get("lookahead_steps") == 0
        assert spec.default_params.get("lookahead_alpha") == 0.5
        assert spec.default_params.get("sam_rho") == 0.0
        assert spec.default_params.get("sam_adaptive") is False
        assert spec.default_params.get("horizon_loss_decay") == 1.0
        assert spec.default_params.get("input_dropout") == 0.0
        assert spec.default_params.get("temporal_dropout") == 0.0
        assert spec.default_params.get("grad_noise_std") == 0.0
        assert spec.default_params.get("gc_mode") == "off"
        assert spec.default_params.get("agc_clip_factor") == 0.0
        assert spec.default_params.get("agc_eps") == 1e-3
        assert spec.default_params.get("checkpoint_dir") == ""
        assert spec.default_params.get("save_best_checkpoint") is False
        assert spec.default_params.get("save_last_checkpoint") is False
        assert spec.default_params.get("resume_checkpoint_path") == ""
        assert spec.default_params.get("resume_checkpoint_strict") is True
        assert spec.default_params.get("mlflow_tracking_uri") == ""
        assert spec.default_params.get("mlflow_experiment_name") == ""
        assert spec.default_params.get("mlflow_run_name") == ""
        assert spec.default_params.get("wandb_project") == ""
        assert spec.default_params.get("wandb_entity") == ""
        assert spec.default_params.get("wandb_run_name") == ""
        assert spec.default_params.get("wandb_dir") == ""
        assert spec.default_params.get("wandb_mode") == "online"
        assert spec.param_help.get("min_epochs") == "Minimum epochs before early stopping can trigger"
        assert spec.param_help.get("amp") == "Enable CUDA automatic mixed precision (true/false)"
        assert spec.param_help.get("amp_dtype") == "AMP compute dtype: auto, float16, bfloat16"
        assert spec.param_help.get("warmup_epochs") == "Linear LR warmup epochs before the main scheduler"
        assert spec.param_help.get("min_lr") == "Lower bound for learning rate during scheduler updates"
        assert spec.param_help.get("scheduler_restart_period") == (
            "Initial restart period in epochs for scheduler=cosine_restarts"
        )
        assert spec.param_help.get("scheduler_restart_mult") == (
            "Cycle-length multiplier for scheduler=cosine_restarts"
        )
        assert spec.param_help.get("scheduler_pct_start") == (
            "Warmup fraction for scheduler=onecycle, must be in (0,1)"
        )
        assert spec.param_help.get("grad_accum_steps") == "Gradient accumulation steps (>=1)"
        assert spec.param_help.get("monitor") == "Early-stop metric: auto, train_loss, val_loss"
        assert spec.param_help.get("monitor_mode") == "Whether the monitor should be minimized or maximized: min, max"
        assert spec.param_help.get("min_delta") == "Minimum improvement required to reset patience"
        assert spec.param_help.get("num_workers") == "DataLoader worker count (0 uses main process)"
        assert spec.param_help.get("pin_memory") == "Pin DataLoader memory before host-to-device transfer"
        assert (
            spec.param_help.get("persistent_workers")
            == "Keep DataLoader workers alive across epochs (requires num_workers>0)"
        )
        assert spec.param_help.get("scheduler_patience") == (
            "ReduceLROnPlateau patience in epochs (only for scheduler=plateau)"
        )
        assert spec.param_help.get("grad_clip_mode") == (
            "Gradient clipping strategy: norm, value"
        )
        assert spec.param_help.get("grad_clip_value") == (
            "Gradient clipping absolute value threshold (only for grad_clip_mode=value)"
        )
        assert spec.param_help.get("scheduler_plateau_factor") == (
            "ReduceLROnPlateau decay factor in (0,1) (only for scheduler=plateau)"
        )
        assert spec.param_help.get("scheduler_plateau_threshold") == (
            "ReduceLROnPlateau minimum monitored change before a decay step"
        )
        assert spec.param_help.get("ema_decay") == (
            "EMA decay in [0,1); 0 disables exponential moving average weights"
        )
        assert spec.param_help.get("ema_warmup_epochs") == (
            "Warmup epochs before EMA updates start"
        )
        assert spec.param_help.get("swa_start_epoch") == (
            "Epoch index where stochastic weight averaging starts; -1 disables SWA"
        )
        assert spec.param_help.get("lookahead_steps") == (
            "Lookahead sync interval in optimizer steps; 0 disables Lookahead"
        )
        assert spec.param_help.get("lookahead_alpha") == (
            "Lookahead slow-weight interpolation factor in (0,1]"
        )
        assert spec.param_help.get("sam_rho") == (
            "SAM neighborhood size rho; 0 disables Sharpness-Aware Minimization"
        )
        assert spec.param_help.get("sam_adaptive") == (
            "Use adaptive SAM scaling based on parameter magnitudes"
        )
        assert spec.param_help.get("horizon_loss_decay") == (
            "Per-step exponential horizon loss decay (>0); 1 disables weighting"
        )
        assert spec.param_help.get("input_dropout") == (
            "Feature dropout applied to training inputs only; 0 disables"
        )
        assert spec.param_help.get("temporal_dropout") == (
            "Drop whole training timesteps across all features; 0 disables"
        )
        assert spec.param_help.get("grad_noise_std") == (
            "Gradient noise stddev before AGC/clipping; 0 disables"
        )
        assert spec.param_help.get("gc_mode") == (
            "Gradient centralization mode: off, all, conv_only"
        )
        assert spec.param_help.get("agc_clip_factor") == (
            "Adaptive Gradient Clipping factor; 0 disables AGC"
        )
        assert spec.param_help.get("agc_eps") == (
            "Adaptive Gradient Clipping epsilon for parameter-norm stabilization"
        )
        assert spec.param_help.get("checkpoint_dir") == (
            "Directory for trainer checkpoints (writes best.pt/last.pt when enabled)"
        )
        assert spec.param_help.get("save_best_checkpoint") == (
            "Persist the best training checkpoint to checkpoint_dir/best.pt"
        )
        assert spec.param_help.get("save_last_checkpoint") == (
            "Persist the last training checkpoint to checkpoint_dir/last.pt"
        )
        assert spec.param_help.get("resume_checkpoint_path") == (
            "Load initial model weights from this checkpoint path before training"
        )
        assert spec.param_help.get("resume_checkpoint_strict") == (
            "Use strict state_dict loading when resume_checkpoint_path is set"
        )
        assert spec.param_help.get("mlflow_tracking_uri") == (
            "Optional MLflow tracking URI; uses the MLflow default backend when unset"
        )
        assert spec.param_help.get("mlflow_experiment_name") == (
            "Optional MLflow experiment name; enables MLflow tracking when set"
        )
        assert spec.param_help.get("mlflow_run_name") == (
            "Optional MLflow run name (default: timestamped run-* name)"
        )
        assert spec.param_help.get("wandb_project") == (
            "Optional Weights & Biases project; enables W&B tracking when set"
        )
        assert spec.param_help.get("wandb_entity") == (
            "Optional Weights & Biases entity / team for wandb_project"
        )
        assert spec.param_help.get("wandb_run_name") == (
            "Optional Weights & Biases run name"
        )
        assert spec.param_help.get("wandb_dir") == (
            "Optional Weights & Biases local run directory"
        )
        assert spec.param_help.get("wandb_mode") == (
            "Weights & Biases mode: online, offline, disabled"
        )


def test_torch_local_catalog_exposes_wave9_preset_defaults() -> None:
    timemixer_deep = get_model_spec("torch-timemixer-deep-direct")
    timemixer_wide = get_model_spec("torch-timemixer-wide-direct")
    frets_deep = get_model_spec("torch-frets-deep-direct")
    frets_wide = get_model_spec("torch-frets-wide-direct")
    film_deep = get_model_spec("torch-film-deep-direct")
    film_wide = get_model_spec("torch-film-wide-direct")
    micn_deep = get_model_spec("torch-micn-deep-direct")
    micn_wide = get_model_spec("torch-micn-wide-direct")
    koopa_deep = get_model_spec("torch-koopa-deep-direct")
    koopa_wide = get_model_spec("torch-koopa-wide-direct")
    samformer_deep = get_model_spec("torch-samformer-deep-direct")
    samformer_wide = get_model_spec("torch-samformer-wide-direct")

    assert timemixer_deep.default_params["num_blocks"] == 6
    assert timemixer_wide.default_params["d_model"] == 128
    assert timemixer_wide.default_params["token_mixing_hidden"] == 256
    assert frets_deep.default_params["num_layers"] == 4
    assert frets_wide.default_params["d_model"] == 128
    assert film_deep.default_params["num_layers"] == 4
    assert film_wide.default_params["d_model"] == 128
    assert micn_deep.default_params["num_layers"] == 4
    assert micn_wide.default_params["d_model"] == 128
    assert koopa_deep.default_params["num_blocks"] == 4
    assert koopa_wide.default_params["d_model"] == 128
    assert koopa_wide.default_params["latent_dim"] == 64
    assert samformer_deep.default_params["num_layers"] == 4
    assert samformer_wide.default_params["d_model"] == 128
    assert samformer_wide.default_params["nhead"] == 8


def test_torch_local_catalog_exposes_wave10_sequence_preset_defaults() -> None:
    fnet_deep = get_model_spec("torch-fnet-deep-direct")
    fnet_wide = get_model_spec("torch-fnet-wide-direct")
    gmlp_deep = get_model_spec("torch-gmlp-deep-direct")
    gmlp_wide = get_model_spec("torch-gmlp-wide-direct")
    linear_attn_deep = get_model_spec("torch-linear-attn-deep-direct")
    linear_attn_wide = get_model_spec("torch-linear-attn-wide-direct")
    inception_deep = get_model_spec("torch-inception-deep-direct")
    inception_wide = get_model_spec("torch-inception-wide-direct")
    mamba_deep = get_model_spec("torch-mamba-deep-direct")
    mamba_wide = get_model_spec("torch-mamba-wide-direct")
    rwkv_deep = get_model_spec("torch-rwkv-deep-direct")
    rwkv_wide = get_model_spec("torch-rwkv-wide-direct")
    hyena_deep = get_model_spec("torch-hyena-deep-direct")
    hyena_wide = get_model_spec("torch-hyena-wide-direct")
    dilated_rnn_deep = get_model_spec("torch-dilated-rnn-deep-direct")
    dilated_rnn_wide = get_model_spec("torch-dilated-rnn-wide-direct")

    assert fnet_deep.default_params["num_layers"] == 6
    assert fnet_wide.default_params["d_model"] == 128
    assert fnet_wide.default_params["dim_feedforward"] == 512
    assert gmlp_deep.default_params["num_layers"] == 6
    assert gmlp_wide.default_params["d_model"] == 128
    assert gmlp_wide.default_params["ffn_dim"] == 256
    assert linear_attn_deep.default_params["num_layers"] == 4
    assert linear_attn_wide.default_params["d_model"] == 128
    assert linear_attn_wide.default_params["dim_feedforward"] == 512
    assert inception_deep.default_params["num_blocks"] == 5
    assert inception_wide.default_params["channels"] == 64
    assert inception_wide.default_params["bottleneck_channels"] == 32
    assert mamba_deep.default_params["num_layers"] == 4
    assert mamba_wide.default_params["d_model"] == 128
    assert rwkv_deep.default_params["num_layers"] == 4
    assert rwkv_wide.default_params["d_model"] == 128
    assert rwkv_wide.default_params["ffn_dim"] == 256
    assert hyena_deep.default_params["num_layers"] == 4
    assert hyena_wide.default_params["d_model"] == 128
    assert hyena_wide.default_params["ffn_dim"] == 256
    assert dilated_rnn_deep.default_params["num_layers"] == 5
    assert dilated_rnn_wide.default_params["hidden_size"] == 128


def test_torch_local_catalog_exposes_wave11_capacity_preset_defaults() -> None:
    kan_deep = get_model_spec("torch-kan-deep-direct")
    kan_wide = get_model_spec("torch-kan-wide-direct")
    scinet_deep = get_model_spec("torch-scinet-deep-direct")
    scinet_wide = get_model_spec("torch-scinet-wide-direct")
    etsformer_deep = get_model_spec("torch-etsformer-deep-direct")
    etsformer_wide = get_model_spec("torch-etsformer-wide-direct")
    esrnn_deep = get_model_spec("torch-esrnn-deep-direct")
    esrnn_wide = get_model_spec("torch-esrnn-wide-direct")
    patchtst_deep = get_model_spec("torch-patchtst-deep-direct")
    patchtst_wide = get_model_spec("torch-patchtst-wide-direct")
    crossformer_deep = get_model_spec("torch-crossformer-deep-direct")
    crossformer_wide = get_model_spec("torch-crossformer-wide-direct")
    pyraformer_deep = get_model_spec("torch-pyraformer-deep-direct")
    pyraformer_wide = get_model_spec("torch-pyraformer-wide-direct")
    tsmixer_deep = get_model_spec("torch-tsmixer-deep-direct")
    tsmixer_wide = get_model_spec("torch-tsmixer-wide-direct")
    nhits_deep = get_model_spec("torch-nhits-deep-direct")
    nhits_wide = get_model_spec("torch-nhits-wide-direct")

    assert kan_deep.default_params["num_layers"] == 4
    assert kan_wide.default_params["d_model"] == 128
    assert scinet_deep.default_params["num_stages"] == 4
    assert scinet_wide.default_params["d_model"] == 128
    assert scinet_wide.default_params["ffn_dim"] == 256
    assert etsformer_deep.default_params["num_layers"] == 4
    assert etsformer_wide.default_params["d_model"] == 128
    assert etsformer_wide.default_params["nhead"] == 8
    assert etsformer_wide.default_params["dim_feedforward"] == 512
    assert esrnn_deep.default_params["num_layers"] == 4
    assert esrnn_wide.default_params["hidden_size"] == 128
    assert patchtst_deep.default_params["num_layers"] == 4
    assert patchtst_wide.default_params["d_model"] == 128
    assert patchtst_wide.default_params["nhead"] == 8
    assert patchtst_wide.default_params["dim_feedforward"] == 512
    assert crossformer_deep.default_params["num_layers"] == 4
    assert crossformer_wide.default_params["d_model"] == 128
    assert crossformer_wide.default_params["nhead"] == 8
    assert crossformer_wide.default_params["dim_feedforward"] == 512
    assert pyraformer_deep.default_params["num_layers"] == 4
    assert pyraformer_wide.default_params["d_model"] == 128
    assert pyraformer_wide.default_params["nhead"] == 8
    assert pyraformer_wide.default_params["dim_feedforward"] == 512
    assert tsmixer_deep.default_params["num_blocks"] == 6
    assert tsmixer_wide.default_params["d_model"] == 128
    assert tsmixer_wide.default_params["token_mixing_hidden"] == 256
    assert tsmixer_wide.default_params["channel_mixing_hidden"] == 256
    assert nhits_deep.default_params["num_blocks"] == 8
    assert nhits_wide.default_params["layer_width"] == 256


def test_torch_local_catalog_exposes_wave12_foundation_preset_defaults() -> None:
    mlp_deep = get_model_spec("torch-mlp-deep-direct")
    mlp_wide = get_model_spec("torch-mlp-wide-direct")
    lstm_deep = get_model_spec("torch-lstm-deep-direct")
    lstm_wide = get_model_spec("torch-lstm-wide-direct")
    gru_deep = get_model_spec("torch-gru-deep-direct")
    gru_wide = get_model_spec("torch-gru-wide-direct")
    tcn_deep = get_model_spec("torch-tcn-deep-direct")
    tcn_wide = get_model_spec("torch-tcn-wide-direct")
    nbeats_deep = get_model_spec("torch-nbeats-deep-direct")
    nbeats_wide = get_model_spec("torch-nbeats-wide-direct")
    transformer_deep = get_model_spec("torch-transformer-deep-direct")
    transformer_wide = get_model_spec("torch-transformer-wide-direct")
    cnn_deep = get_model_spec("torch-cnn-deep-direct")
    cnn_wide = get_model_spec("torch-cnn-wide-direct")
    resnet1d_deep = get_model_spec("torch-resnet1d-deep-direct")
    resnet1d_wide = get_model_spec("torch-resnet1d-wide-direct")
    wavenet_deep = get_model_spec("torch-wavenet-deep-direct")
    wavenet_wide = get_model_spec("torch-wavenet-wide-direct")

    assert mlp_deep.default_params["hidden_sizes"] == (64, 64, 64)
    assert mlp_wide.default_params["hidden_sizes"] == (128, 128)
    assert lstm_deep.default_params["num_layers"] == 2
    assert lstm_deep.default_params["dropout"] == 0.1
    assert lstm_wide.default_params["hidden_size"] == 128
    assert gru_deep.default_params["num_layers"] == 2
    assert gru_deep.default_params["dropout"] == 0.1
    assert gru_wide.default_params["hidden_size"] == 128
    assert tcn_deep.default_params["channels"] == (16, 16, 16, 16)
    assert tcn_wide.default_params["channels"] == (32, 32, 32)
    assert nbeats_deep.default_params["num_blocks"] == 5
    assert nbeats_wide.default_params["layer_width"] == 128
    assert transformer_deep.default_params["num_layers"] == 4
    assert transformer_wide.default_params["d_model"] == 128
    assert transformer_wide.default_params["nhead"] == 8
    assert transformer_wide.default_params["dim_feedforward"] == 512
    assert cnn_deep.default_params["channels"] == (32, 32, 32, 32)
    assert cnn_wide.default_params["channels"] == (64, 64, 64)
    assert resnet1d_deep.default_params["num_blocks"] == 6
    assert resnet1d_wide.default_params["channels"] == 64
    assert wavenet_deep.default_params["num_layers"] == 8
    assert wavenet_wide.default_params["channels"] == 64


def test_torch_local_catalog_exposes_wave13_recurrent_stateful_preset_defaults() -> None:
    bilstm_deep = get_model_spec("torch-bilstm-deep-direct")
    bilstm_wide = get_model_spec("torch-bilstm-wide-direct")
    bigru_deep = get_model_spec("torch-bigru-deep-direct")
    bigru_wide = get_model_spec("torch-bigru-wide-direct")
    attn_gru_deep = get_model_spec("torch-attn-gru-deep-direct")
    attn_gru_wide = get_model_spec("torch-attn-gru-wide-direct")
    lmu_deep = get_model_spec("torch-lmu-deep-direct")
    lmu_wide = get_model_spec("torch-lmu-wide-direct")
    ltc_deep = get_model_spec("torch-ltc-deep-direct")
    ltc_wide = get_model_spec("torch-ltc-wide-direct")
    cfc_deep = get_model_spec("torch-cfc-deep-direct")
    cfc_wide = get_model_spec("torch-cfc-wide-direct")
    xlstm_deep = get_model_spec("torch-xlstm-deep-direct")
    xlstm_wide = get_model_spec("torch-xlstm-wide-direct")
    griffin_deep = get_model_spec("torch-griffin-deep-direct")
    griffin_wide = get_model_spec("torch-griffin-wide-direct")
    hawk_deep = get_model_spec("torch-hawk-deep-direct")
    hawk_wide = get_model_spec("torch-hawk-wide-direct")

    assert bilstm_deep.default_params["num_layers"] == 2
    assert bilstm_deep.default_params["dropout"] == 0.1
    assert bilstm_wide.default_params["hidden_size"] == 128
    assert bigru_deep.default_params["num_layers"] == 2
    assert bigru_deep.default_params["dropout"] == 0.1
    assert bigru_wide.default_params["hidden_size"] == 128
    assert attn_gru_deep.default_params["num_layers"] == 2
    assert attn_gru_wide.default_params["hidden_size"] == 128
    assert lmu_deep.default_params["num_layers"] == 2
    assert lmu_wide.default_params["d_model"] == 128
    assert lmu_wide.default_params["memory_dim"] == 64
    assert ltc_deep.default_params["num_layers"] == 2
    assert ltc_wide.default_params["hidden_size"] == 128
    assert cfc_deep.default_params["num_layers"] == 2
    assert cfc_wide.default_params["hidden_size"] == 128
    assert cfc_wide.default_params["backbone_hidden"] == 256
    assert xlstm_deep.default_params["num_layers"] == 2
    assert xlstm_wide.default_params["hidden_size"] == 128
    assert griffin_deep.default_params["num_layers"] == 2
    assert griffin_wide.default_params["hidden_size"] == 128
    assert hawk_deep.default_params["num_layers"] == 2
    assert hawk_wide.default_params["hidden_size"] == 128


def test_torch_local_catalog_exposes_wave14_transformer_ssm_preset_defaults() -> None:
    informer_deep = get_model_spec("torch-informer-deep-direct")
    informer_wide = get_model_spec("torch-informer-wide-direct")
    autoformer_deep = get_model_spec("torch-autoformer-deep-direct")
    autoformer_wide = get_model_spec("torch-autoformer-wide-direct")
    nonstationary_deep = get_model_spec("torch-nonstationary-transformer-deep-direct")
    nonstationary_wide = get_model_spec("torch-nonstationary-transformer-wide-direct")
    fedformer_deep = get_model_spec("torch-fedformer-deep-direct")
    fedformer_wide = get_model_spec("torch-fedformer-wide-direct")
    itransformer_deep = get_model_spec("torch-itransformer-deep-direct")
    itransformer_wide = get_model_spec("torch-itransformer-wide-direct")
    timesnet_deep = get_model_spec("torch-timesnet-deep-direct")
    timesnet_wide = get_model_spec("torch-timesnet-wide-direct")
    tft_deep = get_model_spec("torch-tft-deep-direct")
    tft_wide = get_model_spec("torch-tft-wide-direct")
    timexer_deep = get_model_spec("torch-timexer-deep-direct")
    timexer_wide = get_model_spec("torch-timexer-wide-direct")
    s4d_deep = get_model_spec("torch-s4d-deep-direct")
    s4d_wide = get_model_spec("torch-s4d-wide-direct")
    s4_deep = get_model_spec("torch-s4-deep-direct")
    s4_wide = get_model_spec("torch-s4-wide-direct")
    s5_deep = get_model_spec("torch-s5-deep-direct")
    s5_wide = get_model_spec("torch-s5-wide-direct")
    mamba2_deep = get_model_spec("torch-mamba2-deep-direct")
    mamba2_wide = get_model_spec("torch-mamba2-wide-direct")

    assert informer_deep.default_params["num_layers"] == 4
    assert informer_wide.default_params["d_model"] == 128
    assert informer_wide.default_params["nhead"] == 8
    assert informer_wide.default_params["dim_feedforward"] == 512
    assert autoformer_deep.default_params["num_layers"] == 4
    assert autoformer_wide.default_params["d_model"] == 128
    assert autoformer_wide.default_params["nhead"] == 8
    assert autoformer_wide.default_params["dim_feedforward"] == 512
    assert nonstationary_deep.default_params["num_layers"] == 4
    assert nonstationary_wide.default_params["d_model"] == 128
    assert nonstationary_wide.default_params["nhead"] == 8
    assert nonstationary_wide.default_params["dim_feedforward"] == 512
    assert fedformer_deep.default_params["num_layers"] == 4
    assert fedformer_wide.default_params["d_model"] == 128
    assert fedformer_wide.default_params["ffn_dim"] == 512
    assert itransformer_deep.default_params["num_layers"] == 4
    assert itransformer_wide.default_params["d_model"] == 128
    assert itransformer_wide.default_params["nhead"] == 8
    assert itransformer_wide.default_params["dim_feedforward"] == 512
    assert timesnet_deep.default_params["num_layers"] == 4
    assert timesnet_wide.default_params["d_model"] == 128
    assert tft_deep.default_params["lstm_layers"] == 2
    assert tft_wide.default_params["d_model"] == 128
    assert tft_wide.default_params["nhead"] == 8
    assert timexer_deep.default_params["num_layers"] == 4
    assert timexer_wide.default_params["d_model"] == 128
    assert timexer_wide.default_params["nhead"] == 8
    assert s4d_deep.default_params["num_layers"] == 4
    assert s4d_wide.default_params["d_model"] == 128
    assert s4_deep.default_params["num_layers"] == 4
    assert s4_wide.default_params["d_model"] == 128
    assert s5_deep.default_params["num_layers"] == 4
    assert s5_wide.default_params["d_model"] == 128
    assert mamba2_deep.default_params["num_layers"] == 4
    assert mamba2_wide.default_params["d_model"] == 128


def test_torch_local_catalog_exposes_wave15_remaining_baseline_preset_defaults() -> None:
    retnet_deep = get_model_spec("torch-retnet-deep-direct")
    retnet_wide = get_model_spec("torch-retnet-wide-direct")
    lightts_long = get_model_spec("torch-lightts-long-direct")
    lightts_wide = get_model_spec("torch-lightts-wide-direct")
    sparsetsf_long = get_model_spec("torch-sparsetsf-long-direct")
    sparsetsf_wide = get_model_spec("torch-sparsetsf-wide-direct")
    tide_long = get_model_spec("torch-tide-long-direct")
    tide_wide = get_model_spec("torch-tide-wide-direct")
    nlinear_long = get_model_spec("torch-nlinear-long-direct")
    dlinear_long = get_model_spec("torch-dlinear-long-direct")

    assert retnet_deep.default_params["num_layers"] == 4
    assert retnet_wide.default_params["d_model"] == 128
    assert retnet_wide.default_params["nhead"] == 8
    assert retnet_wide.default_params["ffn_dim"] == 256
    assert lightts_long.default_params["lags"] == 192
    assert lightts_long.default_params["chunk_len"] == 24
    assert lightts_wide.default_params["d_model"] == 128
    assert sparsetsf_long.default_params["lags"] == 336
    assert sparsetsf_wide.default_params["d_model"] == 128
    assert tide_long.default_params["lags"] == 192
    assert tide_wide.default_params["d_model"] == 128
    assert tide_wide.default_params["hidden_size"] == 256
    assert nlinear_long.default_params["lags"] == 192
    assert dlinear_long.default_params["lags"] == 192


def test_torch_local_catalog_exposes_wave23_lstnet_preset_defaults() -> None:
    lstnet_long = get_model_spec("torch-lstnet-long-direct")
    lstnet_wide = get_model_spec("torch-lstnet-wide-direct")

    assert lstnet_long.default_params["lags"] == 192
    assert lstnet_wide.default_params["cnn_channels"] == 32
    assert lstnet_wide.default_params["rnn_hidden"] == 64


def test_torch_local_catalog_exposes_wave24_structured_rnn_preset_defaults() -> None:
    multidim_long = get_model_spec("torch-multidim-rnn-long-direct")
    multidim_wide = get_model_spec("torch-multidim-rnn-wide-direct")
    grid_lstm_long = get_model_spec("torch-grid-lstm-long-direct")
    grid_lstm_wide = get_model_spec("torch-grid-lstm-wide-direct")
    structural_long = get_model_spec("torch-structural-rnn-long-direct")
    structural_wide = get_model_spec("torch-structural-rnn-wide-direct")

    assert multidim_long.default_params["lags"] == 48
    assert multidim_wide.default_params["hidden_size"] == 64
    assert grid_lstm_long.default_params["lags"] == 48
    assert grid_lstm_wide.default_params["hidden_size"] == 64
    assert structural_long.default_params["lags"] == 48
    assert structural_wide.default_params["hidden_size"] == 64


def test_torch_local_catalog_exposes_wave25_recursive_rnn_preset_defaults() -> None:
    deepar_deep = get_model_spec("torch-deepar-deep-recursive")
    deepar_wide = get_model_spec("torch-deepar-wide-recursive")
    qrnn_deep = get_model_spec("torch-qrnn-deep-recursive")
    qrnn_wide = get_model_spec("torch-qrnn-wide-recursive")

    assert deepar_deep.default_params["num_layers"] == 2
    assert deepar_deep.default_params["dropout"] == 0.1
    assert deepar_wide.default_params["hidden_size"] == 64
    assert qrnn_deep.default_params["num_layers"] == 2
    assert qrnn_deep.default_params["dropout"] == 0.1
    assert qrnn_wide.default_params["hidden_size"] == 64


def test_torch_local_catalog_exposes_wave26_probabilistic_preset_defaults() -> None:
    timegrad_long = get_model_spec("torch-timegrad-long-direct")
    timegrad_wide = get_model_spec("torch-timegrad-wide-direct")
    tactis_long = get_model_spec("torch-tactis-long-direct")
    tactis_wide = get_model_spec("torch-tactis-wide-direct")

    assert timegrad_long.default_params["lags"] == 48
    assert timegrad_wide.default_params["hidden_size"] == 64
    assert tactis_long.default_params["lags"] == 48
    assert tactis_wide.default_params["hidden_size"] == 64
    assert tactis_wide.default_params["num_heads"] == 4


def test_torch_local_catalog_exposes_wave27_reservoir_preset_defaults() -> None:
    esn_long = get_model_spec("torch-esn-long-direct")
    esn_wide = get_model_spec("torch-esn-wide-direct")
    deep_esn_long = get_model_spec("torch-deep-esn-long-direct")
    deep_esn_wide = get_model_spec("torch-deep-esn-wide-direct")
    liquid_long = get_model_spec("torch-liquid-state-long-direct")
    liquid_wide = get_model_spec("torch-liquid-state-wide-direct")

    assert esn_long.default_params["lags"] == 48
    assert esn_wide.default_params["hidden_size"] == 64
    assert deep_esn_long.default_params["lags"] == 48
    assert deep_esn_wide.default_params["hidden_size"] == 64
    assert liquid_long.default_params["lags"] == 48
    assert liquid_wide.default_params["hidden_size"] == 64


def test_torch_local_catalog_exposes_wave28_retnet_recursive_preset_defaults() -> None:
    retnet_deep = get_model_spec("torch-retnet-deep-recursive")
    retnet_wide = get_model_spec("torch-retnet-wide-recursive")

    assert retnet_deep.default_params["num_layers"] == 4
    assert retnet_wide.default_params["d_model"] == 128
    assert retnet_wide.default_params["nhead"] == 8
    assert retnet_wide.default_params["ffn_dim"] == 256


def test_torch_local_catalog_exposes_wave29_canonical_rnn_zoo_preset_defaults() -> None:
    rnnpaper_lstm_long = get_model_spec("torch-rnnpaper-lstm-long-direct")
    rnnpaper_lstm_wide = get_model_spec("torch-rnnpaper-lstm-wide-direct")
    rnnpaper_gru_long = get_model_spec("torch-rnnpaper-gru-long-direct")
    rnnpaper_gru_wide = get_model_spec("torch-rnnpaper-gru-wide-direct")
    rnnpaper_qrnn_long = get_model_spec("torch-rnnpaper-qrnn-long-direct")
    rnnpaper_qrnn_wide = get_model_spec("torch-rnnpaper-qrnn-wide-direct")
    rnnzoo_lstm_long = get_model_spec("torch-rnnzoo-lstm-long-direct")
    rnnzoo_lstm_wide = get_model_spec("torch-rnnzoo-lstm-wide-direct")
    rnnzoo_gru_long = get_model_spec("torch-rnnzoo-gru-long-direct")
    rnnzoo_gru_wide = get_model_spec("torch-rnnzoo-gru-wide-direct")
    rnnzoo_qrnn_long = get_model_spec("torch-rnnzoo-qrnn-long-direct")
    rnnzoo_qrnn_wide = get_model_spec("torch-rnnzoo-qrnn-wide-direct")

    assert rnnpaper_lstm_long.default_params["lags"] == 48
    assert rnnpaper_lstm_wide.default_params["hidden_size"] == 64
    assert rnnpaper_gru_long.default_params["lags"] == 48
    assert rnnpaper_gru_wide.default_params["hidden_size"] == 64
    assert rnnpaper_qrnn_long.default_params["lags"] == 48
    assert rnnpaper_qrnn_wide.default_params["hidden_size"] == 64
    assert rnnzoo_lstm_long.default_params["lags"] == 48
    assert rnnzoo_lstm_wide.default_params["hidden_size"] == 64
    assert rnnzoo_gru_long.default_params["lags"] == 48
    assert rnnzoo_gru_wide.default_params["hidden_size"] == 64
    assert rnnzoo_qrnn_long.default_params["lags"] == 48
    assert rnnzoo_qrnn_wide.default_params["hidden_size"] == 64


def test_torch_local_catalog_exposes_wave30_training_strategy_preset_defaults() -> None:
    patchtst_ema = get_model_spec("torch-patchtst-ema-direct")
    timesnet_swa = get_model_spec("torch-timesnet-swa-direct")
    timexer_sam = get_model_spec("torch-timexer-sam-direct")
    tsmixer_regularized = get_model_spec("torch-tsmixer-regularized-direct")
    tft_longhorizon = get_model_spec("torch-tft-longhorizon-direct")
    nbeats_lookahead = get_model_spec("torch-nbeats-lookahead-direct")

    assert patchtst_ema.default_params["ema_decay"] == 0.995
    assert patchtst_ema.default_params["scheduler"] == "cosine"
    assert timesnet_swa.default_params["swa_start_epoch"] == 18
    assert timesnet_swa.default_params["scheduler"] == "cosine_restarts"
    assert timexer_sam.default_params["sam_rho"] == 0.05
    assert timexer_sam.default_params["sam_adaptive"] is True
    assert tsmixer_regularized.default_params["input_dropout"] == 0.1
    assert tsmixer_regularized.default_params["temporal_dropout"] == 0.05
    assert tft_longhorizon.default_params["horizon_loss_decay"] == 1.05
    assert tft_longhorizon.default_params["loss"] == "huber"
    assert nbeats_lookahead.default_params["lookahead_steps"] == 5
    assert nbeats_lookahead.default_params["lookahead_alpha"] == 0.5


def test_torch_global_catalog_exposes_wave16_global_preset_defaults() -> None:
    tft_deep = get_model_spec("torch-tft-deep-global")
    tft_wide = get_model_spec("torch-tft-wide-global")
    timexer_deep = get_model_spec("torch-timexer-deep-global")
    timexer_wide = get_model_spec("torch-timexer-wide-global")
    retnet_deep = get_model_spec("torch-retnet-deep-global")
    retnet_wide = get_model_spec("torch-retnet-wide-global")
    informer_deep = get_model_spec("torch-informer-deep-global")
    informer_wide = get_model_spec("torch-informer-wide-global")
    autoformer_deep = get_model_spec("torch-autoformer-deep-global")
    autoformer_wide = get_model_spec("torch-autoformer-wide-global")
    fedformer_deep = get_model_spec("torch-fedformer-deep-global")
    fedformer_wide = get_model_spec("torch-fedformer-wide-global")
    nonstationary_deep = get_model_spec("torch-nonstationary-transformer-deep-global")
    nonstationary_wide = get_model_spec("torch-nonstationary-transformer-wide-global")
    patchtst_deep = get_model_spec("torch-patchtst-deep-global")
    patchtst_wide = get_model_spec("torch-patchtst-wide-global")
    crossformer_deep = get_model_spec("torch-crossformer-deep-global")
    crossformer_wide = get_model_spec("torch-crossformer-wide-global")
    pyraformer_deep = get_model_spec("torch-pyraformer-deep-global")
    pyraformer_wide = get_model_spec("torch-pyraformer-wide-global")
    itransformer_deep = get_model_spec("torch-itransformer-deep-global")
    itransformer_wide = get_model_spec("torch-itransformer-wide-global")
    timesnet_deep = get_model_spec("torch-timesnet-deep-global")
    timesnet_wide = get_model_spec("torch-timesnet-wide-global")

    assert tft_deep.default_params["lstm_layers"] == 2
    assert tft_wide.default_params["d_model"] == 128
    assert tft_wide.default_params["nhead"] == 8
    assert timexer_deep.default_params["num_layers"] == 4
    assert timexer_wide.default_params["d_model"] == 128
    assert timexer_wide.default_params["nhead"] == 8
    assert retnet_deep.default_params["num_layers"] == 4
    assert retnet_wide.default_params["d_model"] == 128
    assert retnet_wide.default_params["nhead"] == 8
    assert retnet_wide.default_params["ffn_dim"] == 512
    assert informer_deep.default_params["num_layers"] == 4
    assert informer_wide.default_params["d_model"] == 128
    assert informer_wide.default_params["nhead"] == 8
    assert informer_wide.default_params["dim_feedforward"] == 512
    assert autoformer_deep.default_params["num_layers"] == 4
    assert autoformer_wide.default_params["d_model"] == 128
    assert autoformer_wide.default_params["nhead"] == 8
    assert autoformer_wide.default_params["dim_feedforward"] == 512
    assert fedformer_deep.default_params["num_layers"] == 4
    assert fedformer_wide.default_params["d_model"] == 128
    assert fedformer_wide.default_params["ffn_dim"] == 512
    assert nonstationary_deep.default_params["num_layers"] == 4
    assert nonstationary_wide.default_params["d_model"] == 128
    assert nonstationary_wide.default_params["nhead"] == 8
    assert nonstationary_wide.default_params["dim_feedforward"] == 512
    assert patchtst_deep.default_params["num_layers"] == 4
    assert patchtst_wide.default_params["d_model"] == 128
    assert patchtst_wide.default_params["nhead"] == 8
    assert patchtst_wide.default_params["dim_feedforward"] == 512
    assert crossformer_deep.default_params["num_layers"] == 4
    assert crossformer_wide.default_params["d_model"] == 128
    assert crossformer_wide.default_params["nhead"] == 8
    assert crossformer_wide.default_params["dim_feedforward"] == 512
    assert pyraformer_deep.default_params["num_layers"] == 4
    assert pyraformer_wide.default_params["d_model"] == 128
    assert pyraformer_wide.default_params["nhead"] == 8
    assert pyraformer_wide.default_params["dim_feedforward"] == 512
    assert itransformer_deep.default_params["num_layers"] == 4
    assert itransformer_wide.default_params["d_model"] == 128
    assert itransformer_wide.default_params["nhead"] == 8
    assert itransformer_wide.default_params["dim_feedforward"] == 512
    assert timesnet_deep.default_params["num_layers"] == 4
    assert timesnet_wide.default_params["d_model"] == 128


def test_torch_global_catalog_exposes_wave17_global_preset_defaults() -> None:
    tsmixer_deep = get_model_spec("torch-tsmixer-deep-global")
    tsmixer_wide = get_model_spec("torch-tsmixer-wide-global")
    nbeats_deep = get_model_spec("torch-nbeats-deep-global")
    nbeats_wide = get_model_spec("torch-nbeats-wide-global")
    nhits_deep = get_model_spec("torch-nhits-deep-global")
    nhits_wide = get_model_spec("torch-nhits-wide-global")
    tcn_deep = get_model_spec("torch-tcn-deep-global")
    tcn_wide = get_model_spec("torch-tcn-wide-global")
    wavenet_deep = get_model_spec("torch-wavenet-deep-global")
    wavenet_wide = get_model_spec("torch-wavenet-wide-global")
    resnet1d_deep = get_model_spec("torch-resnet1d-deep-global")
    resnet1d_wide = get_model_spec("torch-resnet1d-wide-global")
    inception_deep = get_model_spec("torch-inception-deep-global")
    inception_wide = get_model_spec("torch-inception-wide-global")
    kan_deep = get_model_spec("torch-kan-deep-global")
    kan_wide = get_model_spec("torch-kan-wide-global")
    scinet_deep = get_model_spec("torch-scinet-deep-global")
    scinet_wide = get_model_spec("torch-scinet-wide-global")
    etsformer_deep = get_model_spec("torch-etsformer-deep-global")
    etsformer_wide = get_model_spec("torch-etsformer-wide-global")
    esrnn_deep = get_model_spec("torch-esrnn-deep-global")
    esrnn_wide = get_model_spec("torch-esrnn-wide-global")

    assert tsmixer_deep.default_params["num_blocks"] == 6
    assert tsmixer_wide.default_params["d_model"] == 128
    assert tsmixer_wide.default_params["token_mixing_hidden"] == 256
    assert tsmixer_wide.default_params["channel_mixing_hidden"] == 256
    assert nbeats_deep.default_params["num_blocks"] == 5
    assert nbeats_wide.default_params["layer_width"] == 512
    assert nhits_deep.default_params["num_blocks"] == 8
    assert nhits_wide.default_params["layer_width"] == 512
    assert tcn_deep.default_params["channels"] == (64, 64, 64, 64)
    assert tcn_wide.default_params["channels"] == (128, 128, 128)
    assert wavenet_deep.default_params["num_layers"] == 8
    assert wavenet_wide.default_params["channels"] == 64
    assert resnet1d_deep.default_params["num_blocks"] == 6
    assert resnet1d_wide.default_params["channels"] == 64
    assert inception_deep.default_params["num_blocks"] == 5
    assert inception_wide.default_params["channels"] == 64
    assert inception_wide.default_params["bottleneck_channels"] == 32
    assert kan_deep.default_params["num_layers"] == 4
    assert kan_wide.default_params["d_model"] == 128
    assert scinet_deep.default_params["num_stages"] == 4
    assert scinet_wide.default_params["d_model"] == 128
    assert scinet_wide.default_params["ffn_dim"] == 256
    assert etsformer_deep.default_params["num_layers"] == 4
    assert etsformer_wide.default_params["d_model"] == 128
    assert etsformer_wide.default_params["nhead"] == 8
    assert etsformer_wide.default_params["dim_feedforward"] == 512
    assert esrnn_deep.default_params["num_layers"] == 4
    assert esrnn_wide.default_params["hidden_size"] == 128


def test_torch_global_catalog_exposes_wave18_global_preset_defaults() -> None:
    fnet_deep = get_model_spec("torch-fnet-deep-global")
    fnet_wide = get_model_spec("torch-fnet-wide-global")
    gmlp_deep = get_model_spec("torch-gmlp-deep-global")
    gmlp_wide = get_model_spec("torch-gmlp-wide-global")
    ssm_deep = get_model_spec("torch-ssm-deep-global")
    ssm_wide = get_model_spec("torch-ssm-wide-global")
    mamba_deep = get_model_spec("torch-mamba-deep-global")
    mamba_wide = get_model_spec("torch-mamba-wide-global")
    rwkv_deep = get_model_spec("torch-rwkv-deep-global")
    rwkv_wide = get_model_spec("torch-rwkv-wide-global")
    hyena_deep = get_model_spec("torch-hyena-deep-global")
    hyena_wide = get_model_spec("torch-hyena-wide-global")
    dilated_rnn_deep = get_model_spec("torch-dilated-rnn-deep-global")
    dilated_rnn_wide = get_model_spec("torch-dilated-rnn-wide-global")
    transformer_encdec_deep = get_model_spec("torch-transformer-encdec-deep-global")
    transformer_encdec_wide = get_model_spec("torch-transformer-encdec-wide-global")

    assert fnet_deep.default_params["num_layers"] == 6
    assert fnet_wide.default_params["d_model"] == 128
    assert fnet_wide.default_params["dim_feedforward"] == 512
    assert gmlp_deep.default_params["num_layers"] == 6
    assert gmlp_wide.default_params["d_model"] == 128
    assert gmlp_wide.default_params["ffn_dim"] == 256
    assert ssm_deep.default_params["num_layers"] == 6
    assert ssm_wide.default_params["d_model"] == 128
    assert mamba_deep.default_params["num_layers"] == 6
    assert mamba_wide.default_params["d_model"] == 128
    assert rwkv_deep.default_params["num_layers"] == 6
    assert rwkv_wide.default_params["d_model"] == 128
    assert rwkv_wide.default_params["ffn_dim"] == 256
    assert hyena_deep.default_params["num_layers"] == 6
    assert hyena_wide.default_params["d_model"] == 128
    assert hyena_wide.default_params["ffn_dim"] == 256
    assert dilated_rnn_deep.default_params["num_layers"] == 5
    assert dilated_rnn_wide.default_params["d_model"] == 128
    assert transformer_encdec_deep.default_params["num_layers"] == 4
    assert transformer_encdec_wide.default_params["d_model"] == 128
    assert transformer_encdec_wide.default_params["nhead"] == 8
    assert transformer_encdec_wide.default_params["dim_feedforward"] == 512


def test_torch_global_catalog_exposes_wave19_global_preset_defaults() -> None:
    seq2seq_lstm_deep = get_model_spec("torch-seq2seq-lstm-deep-global")
    seq2seq_lstm_wide = get_model_spec("torch-seq2seq-lstm-wide-global")
    seq2seq_gru_deep = get_model_spec("torch-seq2seq-gru-deep-global")
    seq2seq_gru_wide = get_model_spec("torch-seq2seq-gru-wide-global")
    seq2seq_attn_lstm_deep = get_model_spec("torch-seq2seq-attn-lstm-deep-global")
    seq2seq_attn_lstm_wide = get_model_spec("torch-seq2seq-attn-lstm-wide-global")
    seq2seq_attn_gru_deep = get_model_spec("torch-seq2seq-attn-gru-deep-global")
    seq2seq_attn_gru_wide = get_model_spec("torch-seq2seq-attn-gru-wide-global")

    assert seq2seq_lstm_deep.default_params["num_layers"] == 2
    assert seq2seq_lstm_deep.default_params["dropout"] == 0.1
    assert seq2seq_lstm_wide.default_params["hidden_size"] == 128
    assert seq2seq_gru_deep.default_params["num_layers"] == 2
    assert seq2seq_gru_deep.default_params["dropout"] == 0.1
    assert seq2seq_gru_wide.default_params["hidden_size"] == 128
    assert seq2seq_attn_lstm_deep.default_params["num_layers"] == 2
    assert seq2seq_attn_lstm_deep.default_params["dropout"] == 0.1
    assert seq2seq_attn_lstm_wide.default_params["hidden_size"] == 128
    assert seq2seq_attn_gru_deep.default_params["num_layers"] == 2
    assert seq2seq_attn_gru_deep.default_params["dropout"] == 0.1
    assert seq2seq_attn_gru_wide.default_params["hidden_size"] == 128


def test_torch_global_catalog_exposes_wave20_global_preset_defaults() -> None:
    deepar_deep = get_model_spec("torch-deepar-deep-global")
    deepar_wide = get_model_spec("torch-deepar-wide-global")
    tide_long = get_model_spec("torch-tide-long-global")
    tide_wide = get_model_spec("torch-tide-wide-global")
    nlinear_long = get_model_spec("torch-nlinear-long-global")
    dlinear_long = get_model_spec("torch-dlinear-long-global")

    assert deepar_deep.default_params["num_layers"] == 2
    assert deepar_deep.default_params["dropout"] == 0.1
    assert deepar_wide.default_params["hidden_size"] == 128
    assert tide_long.default_params["context_length"] == 192
    assert tide_wide.default_params["d_model"] == 128
    assert tide_wide.default_params["hidden_size"] == 256
    assert nlinear_long.default_params["context_length"] == 192
    assert dlinear_long.default_params["context_length"] == 192


def test_torch_global_catalog_exposes_wave21_global_preset_defaults() -> None:
    rnn_lstm_deep = get_model_spec("torch-rnn-lstm-deep-global")
    rnn_lstm_wide = get_model_spec("torch-rnn-lstm-wide-global")
    rnn_gru_deep = get_model_spec("torch-rnn-gru-deep-global")
    rnn_gru_wide = get_model_spec("torch-rnn-gru-wide-global")
    rnn_encoder_deep = get_model_spec("torch-rnn-encoder-deep-global")
    rnn_encoder_wide = get_model_spec("torch-rnn-encoder-wide-global")

    assert rnn_lstm_deep.default_params["num_layers"] == 2
    assert rnn_lstm_deep.default_params["dropout"] == 0.1
    assert rnn_lstm_wide.default_params["hidden_size"] == 128
    assert rnn_gru_deep.default_params["num_layers"] == 2
    assert rnn_gru_deep.default_params["dropout"] == 0.1
    assert rnn_gru_wide.default_params["hidden_size"] == 128
    assert rnn_encoder_deep.default_params["num_layers"] == 2
    assert rnn_encoder_deep.default_params["dropout"] == 0.1
    assert rnn_encoder_deep.default_params["hidden_size"] == 32
    assert rnn_encoder_wide.default_params["hidden_size"] == 64


def test_torch_global_catalog_exposes_wave22_global_preset_defaults() -> None:
    lstnet_long = get_model_spec("torch-lstnet-long-global")
    lstnet_wide = get_model_spec("torch-lstnet-wide-global")

    assert lstnet_long.default_params["context_length"] == 192
    assert lstnet_wide.default_params["cnn_channels"] == 32
    assert lstnet_wide.default_params["rnn_hidden"] == 64


def test_torch_global_catalog_exposes_wave31_training_strategy_preset_defaults() -> None:
    patchtst_ema = get_model_spec("torch-patchtst-ema-global")
    timesnet_swa = get_model_spec("torch-timesnet-swa-global")
    timexer_sam = get_model_spec("torch-timexer-sam-global")
    tsmixer_regularized = get_model_spec("torch-tsmixer-regularized-global")
    tft_longhorizon = get_model_spec("torch-tft-longhorizon-global")
    seq2seq_lookahead = get_model_spec("torch-seq2seq-attn-gru-lookahead-global")

    assert patchtst_ema.default_params["ema_decay"] == 0.995
    assert patchtst_ema.default_params["scheduler"] == "cosine"
    assert timesnet_swa.default_params["swa_start_epoch"] == 18
    assert timesnet_swa.default_params["scheduler"] == "cosine_restarts"
    assert timexer_sam.default_params["sam_rho"] == 0.05
    assert timexer_sam.default_params["sam_adaptive"] is True
    assert tsmixer_regularized.default_params["input_dropout"] == 0.1
    assert tsmixer_regularized.default_params["temporal_dropout"] == 0.05
    assert tft_longhorizon.default_params["horizon_loss_decay"] == 1.05
    assert tft_longhorizon.default_params["loss"] == "huber"
    assert seq2seq_lookahead.default_params["lookahead_steps"] == 5
    assert seq2seq_lookahead.default_params["lookahead_alpha"] == 0.5


def test_classical_catalog_exposes_shared_param_help_strings() -> None:
    from foresight.models.catalog import classical as classical_catalog

    holt = get_model_spec("holt")
    holt_damped = get_model_spec("holt-damped")
    holt_winters_add = get_model_spec("holt-winters-add")
    holt_winters_mul = get_model_spec("holt-winters-mul")
    theta_auto = get_model_spec("theta-auto")
    croston = get_model_spec("croston")

    assert classical_catalog._LEVEL_SMOOTHING_HELP == "Level smoothing in [0, 1]"
    assert classical_catalog._TREND_SMOOTHING_HELP == "Trend smoothing in [0, 1]"
    assert classical_catalog._SEASON_LENGTH_HELP == "Season length"
    assert classical_catalog._ALPHA_GRID_SIZE_HELP == "Number of alpha values to try (default: 19)"
    assert classical_catalog._SMOOTHING_PARAM_HELP == "Smoothing parameter in [0,1]"
    assert holt.param_help["alpha"] == classical_catalog._LEVEL_SMOOTHING_HELP
    assert holt.param_help["beta"] == classical_catalog._TREND_SMOOTHING_HELP
    assert holt_damped.param_help["alpha"] == classical_catalog._LEVEL_SMOOTHING_HELP
    assert holt_winters_add.param_help["season_length"] == classical_catalog._SEASON_LENGTH_HELP
    assert holt_winters_mul.param_help["season_length"] == classical_catalog._SEASON_LENGTH_HELP
    assert theta_auto.param_help["grid_size"] == classical_catalog._ALPHA_GRID_SIZE_HELP
    assert croston.param_help["alpha"] == classical_catalog._SMOOTHING_PARAM_HELP


def test_torch_multivariate_models_are_registered() -> None:
    keys = set(list_models())
    for key in TORCH_MULTIVARIATE_KEYS:
        assert key in keys


def test_transformers_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in TRANSFORMERS_LOCAL_KEYS:
        assert key in keys


def test_catalog_shards_preserve_cross_family_lookup() -> None:
    keys = ["naive-last", "ridge-lag", "arima", "torch-dlinear-direct", "var"]

    for key in keys:
        assert get_model_spec(key).key == key


def test_model_spec_has_description():
    spec = get_model_spec("naive-last")
    assert isinstance(spec.description, str)
    assert spec.description


def test_registry_returns_modelspec_instances() -> None:
    spec = get_model_spec("naive-last")

    assert isinstance(spec, RuntimeModelSpec)
    assert spec.interface == "local"


def test_registry_supports_historical_compatibility_imports() -> None:
    namespace: dict[str, object] = {}

    exec(
        "from foresight.models.registry import "
        "BaseForecaster, BaseGlobalForecaster, "
        "RegistryForecaster, RegistryGlobalForecaster, LocalForecasterFn, ModelFactory",
        namespace,
    )

    assert namespace["BaseForecaster"] is RuntimeBaseForecaster
    assert namespace["BaseGlobalForecaster"] is RuntimeBaseGlobalForecaster
    assert namespace["RegistryForecaster"] is BaseRegistryForecaster
    assert namespace["RegistryGlobalForecaster"] is BaseRegistryGlobalForecaster
    assert namespace["LocalForecasterFn"] is RuntimeLocalForecasterFn
    assert namespace["ModelFactory"] is RuntimeModelFactory


def test_model_spec_exposes_normalized_capabilities():
    spec = get_model_spec("naive-last")

    assert isinstance(spec.capabilities, dict)
    assert spec.capabilities["supports_x_cols"] is False
    assert spec.capabilities["supports_static_cols"] is False
    assert spec.capabilities["supports_quantiles"] is False
    assert spec.capabilities["supports_interval_forecast"] is True
    assert spec.capabilities["supports_interval_forecast_with_x_cols"] is False
    assert spec.capabilities["supports_artifact_save"] is True
    assert spec.capabilities["requires_future_covariates"] is False


def test_model_spec_capability_overrides_can_require_future_covariates() -> None:
    spec = ModelSpec(
        key="__test__",
        description="test",
        factory=lambda **_params: None,
        param_help={"x_cols": "Required future covariates"},
        capability_overrides={"requires_future_covariates": True},
    )

    assert spec.capabilities["supports_x_cols"] is True
    assert spec.capabilities["requires_future_covariates"] is True


def test_model_spec_capabilities_reflect_model_family_support():
    sarimax = get_model_spec("sarimax")
    assert sarimax.capabilities["supports_x_cols"] is True
    assert sarimax.capabilities["supports_static_cols"] is False
    assert sarimax.capabilities["supports_interval_forecast"] is True
    assert sarimax.capabilities["supports_interval_forecast_with_x_cols"] is True

    auto_arima = get_model_spec("auto-arima")
    assert auto_arima.capabilities["supports_x_cols"] is True
    assert auto_arima.capabilities["supports_static_cols"] is False
    assert auto_arima.capabilities["supports_interval_forecast"] is True
    assert auto_arima.capabilities["supports_interval_forecast_with_x_cols"] is True

    global_xgb = get_model_spec("xgb-step-lag-global")
    assert global_xgb.capabilities["supports_x_cols"] is True
    assert global_xgb.capabilities["supports_static_cols"] is False
    assert global_xgb.capabilities["supports_quantiles"] is True
    assert global_xgb.capabilities["supports_interval_forecast"] is True
    assert global_xgb.capabilities["supports_interval_forecast_with_x_cols"] is True
    assert global_xgb.capabilities["supports_artifact_save"] is True

    var = get_model_spec("var")
    assert var.interface == "multivariate"
    assert var.capabilities["supports_artifact_save"] is False

    timexer_local = get_model_spec("torch-timexer-direct")
    assert timexer_local.capabilities["supports_x_cols"] is True
    assert timexer_local.capabilities["supports_static_cols"] is False
    assert timexer_local.capabilities["requires_future_covariates"] is True

    timexer_global = get_model_spec("torch-timexer-global")
    assert timexer_global.capabilities["supports_x_cols"] is True
    assert timexer_global.capabilities["supports_static_cols"] is True
    assert timexer_global.capabilities["requires_future_covariates"] is True

    tft_global = get_model_spec("torch-tft-global")
    assert tft_global.capabilities["supports_static_cols"] is True

    informer_global = get_model_spec("torch-informer-global")
    assert informer_global.capabilities["supports_static_cols"] is True

    autoformer_global = get_model_spec("torch-autoformer-global")
    assert autoformer_global.capabilities["supports_static_cols"] is True

    tsmixer_global = get_model_spec("torch-tsmixer-global")
    assert tsmixer_global.capabilities["supports_static_cols"] is True

    tcn_global = get_model_spec("torch-tcn-global")
    assert tcn_global.capabilities["supports_static_cols"] is True

    deepar_global = get_model_spec("torch-deepar-global")
    assert deepar_global.capabilities["supports_static_cols"] is True

    patchtst_global = get_model_spec("torch-patchtst-global")
    assert patchtst_global.capabilities["supports_static_cols"] is True

    retnet_global = get_model_spec("torch-retnet-global")
    assert retnet_global.capabilities["supports_static_cols"] is True

    timesnet_global = get_model_spec("torch-timesnet-global")
    assert timesnet_global.capabilities["supports_static_cols"] is True

    seq2seq_global = get_model_spec("torch-seq2seq-lstm-global")
    assert seq2seq_global.capabilities["supports_static_cols"] is True

    fedformer_global = get_model_spec("torch-fedformer-global")
    assert fedformer_global.capabilities["supports_static_cols"] is True

    itransformer_global = get_model_spec("torch-itransformer-global")
    assert itransformer_global.capabilities["supports_static_cols"] is True

    crossformer_global = get_model_spec("torch-crossformer-global")
    assert crossformer_global.capabilities["supports_static_cols"] is True

    pyraformer_global = get_model_spec("torch-pyraformer-global")
    assert pyraformer_global.capabilities["supports_static_cols"] is True

    nonstationary_transformer_global = get_model_spec("torch-nonstationary-transformer-global")
    assert nonstationary_transformer_global.capabilities["supports_static_cols"] is True

    xformer_global = get_model_spec("torch-xformer-probsparse-global")
    assert xformer_global.capabilities["supports_static_cols"] is True

    rnn_lstm_global = get_model_spec("torch-rnn-lstm-global")
    assert rnn_lstm_global.capabilities["supports_static_cols"] is True

    transformer_encdec_global = get_model_spec("torch-transformer-encdec-global")
    assert transformer_encdec_global.capabilities["supports_static_cols"] is True

    nbeats_global = get_model_spec("torch-nbeats-global")
    assert nbeats_global.capabilities["supports_static_cols"] is True

    nhits_global = get_model_spec("torch-nhits-global")
    assert nhits_global.capabilities["supports_static_cols"] is True

    tide_global = get_model_spec("torch-tide-global")
    assert tide_global.capabilities["supports_static_cols"] is True

    nlinear_global = get_model_spec("torch-nlinear-global")
    assert nlinear_global.capabilities["supports_static_cols"] is True

    dlinear_global = get_model_spec("torch-dlinear-global")
    assert dlinear_global.capabilities["supports_static_cols"] is True

    wavenet_global = get_model_spec("torch-wavenet-global")
    assert wavenet_global.capabilities["supports_static_cols"] is True

    resnet1d_global = get_model_spec("torch-resnet1d-global")
    assert resnet1d_global.capabilities["supports_static_cols"] is True

    inception_global = get_model_spec("torch-inception-global")
    assert inception_global.capabilities["supports_static_cols"] is True

    lstnet_global = get_model_spec("torch-lstnet-global")
    assert lstnet_global.capabilities["supports_static_cols"] is True

    fnet_global = get_model_spec("torch-fnet-global")
    assert fnet_global.capabilities["supports_static_cols"] is True

    gmlp_global = get_model_spec("torch-gmlp-global")
    assert gmlp_global.capabilities["supports_static_cols"] is True

    ssm_global = get_model_spec("torch-ssm-global")
    assert ssm_global.capabilities["supports_static_cols"] is True

    mamba_global = get_model_spec("torch-mamba-global")
    assert mamba_global.capabilities["supports_static_cols"] is True

    rwkv_global = get_model_spec("torch-rwkv-global")
    assert rwkv_global.capabilities["supports_static_cols"] is True

    hyena_global = get_model_spec("torch-hyena-global")
    assert hyena_global.capabilities["supports_static_cols"] is True

    dilated_rnn_global = get_model_spec("torch-dilated-rnn-global")
    assert dilated_rnn_global.capabilities["supports_static_cols"] is True

    kan_global = get_model_spec("torch-kan-global")
    assert kan_global.capabilities["supports_static_cols"] is True

    scinet_global = get_model_spec("torch-scinet-global")
    assert scinet_global.capabilities["supports_static_cols"] is True

    etsformer_global = get_model_spec("torch-etsformer-global")
    assert etsformer_global.capabilities["supports_static_cols"] is True

    esrnn_global = get_model_spec("torch-esrnn-global")
    assert esrnn_global.capabilities["supports_static_cols"] is True


def test_readme_documents_all_model_capability_flags() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    documented = sorted(
        {key for model_key in list_models() for key in get_model_spec(model_key).capabilities}
    )
    assert documented == [
        "requires_future_covariates",
        "supports_artifact_save",
        "supports_interval_forecast",
        "supports_interval_forecast_with_x_cols",
        "supports_quantiles",
        "supports_static_cols",
        "supports_x_cols",
    ]
    for key in documented:
        assert f"`{key}`" in readme


def test_new_torch_global_models_are_registered():
    keys = set(list_models())
    assert "torch-nlinear-global" in keys
    assert "torch-dlinear-global" in keys
    assert "torch-deepar-global" in keys
    assert "torch-fedformer-global" in keys
    assert "torch-nonstationary-transformer-global" in keys


def test_new_torch_mamba_rwkv_models_are_registered():
    keys = set(list_models())
    assert "torch-mamba-direct" in keys
    assert "torch-rwkv-direct" in keys
    assert "torch-mamba-global" in keys
    assert "torch-rwkv-global" in keys


def test_new_torch_hyena_models_are_registered():
    keys = set(list_models())
    assert "torch-hyena-direct" in keys
    assert "torch-hyena-global" in keys


def test_new_torch_dilated_rnn_and_kan_models_are_registered():
    keys = set(list_models())
    assert "torch-dilated-rnn-direct" in keys
    assert "torch-dilated-rnn-global" in keys
    assert "torch-kan-direct" in keys
    assert "torch-kan-global" in keys


def test_new_torch_scinet_and_etsformer_models_are_registered():
    keys = set(list_models())
    assert "torch-scinet-direct" in keys
    assert "torch-scinet-global" in keys
    assert "torch-etsformer-direct" in keys
    assert "torch-etsformer-global" in keys


def test_new_torch_esrnn_models_are_registered():
    keys = set(list_models())
    assert "torch-esrnn-direct" in keys
    assert "torch-esrnn-global" in keys


def test_new_torch_crossformer_models_are_registered():
    keys = set(list_models())
    assert "torch-crossformer-direct" in keys
    assert "torch-crossformer-global" in keys


def test_new_torch_pyraformer_models_are_registered():
    keys = set(list_models())
    assert "torch-pyraformer-direct" in keys
    assert "torch-pyraformer-global" in keys


def test_new_torch_xformer_attention_variants_are_registered():
    keys = set(list_models())
    assert "torch-xformer-probsparse-ln-gelu-direct" in keys
    assert "torch-xformer-autocorr-ln-gelu-direct" in keys
    assert "torch-xformer-reformer-ln-gelu-direct" in keys
    assert "torch-xformer-logsparse-ln-gelu-direct" in keys
    assert "torch-xformer-longformer-ln-gelu-direct" in keys
    assert "torch-xformer-bigbird-ln-gelu-direct" in keys
    assert "torch-xformer-probsparse-global" in keys
    assert "torch-xformer-autocorr-global" in keys
    assert "torch-xformer-reformer-global" in keys
    assert "torch-xformer-logsparse-global" in keys
    assert "torch-xformer-longformer-global" in keys
    assert "torch-xformer-bigbird-global" in keys
