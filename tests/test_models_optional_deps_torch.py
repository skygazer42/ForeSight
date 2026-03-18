import importlib.util

import numpy as np
import pytest

from foresight.models.registry import (
    get_model_spec,
    list_models,
    make_forecaster,
    make_global_forecaster,
    make_multivariate_forecaster,
)

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

TRAINING_STRATEGY_TORCH_LOCAL_KEYS = (
    "torch-patchtst-ema-direct",
    "torch-timesnet-swa-direct",
    "torch-timexer-sam-direct",
    "torch-tsmixer-regularized-direct",
    "torch-tft-longhorizon-direct",
    "torch-nbeats-lookahead-direct",
)

TRAINING_STRATEGY_TORCH_GLOBAL_KEYS = (
    "torch-patchtst-ema-global",
    "torch-timesnet-swa-global",
    "torch-timexer-sam-global",
    "torch-tsmixer-regularized-global",
    "torch-tft-longhorizon-global",
    "torch-seq2seq-attn-gru-lookahead-global",
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


def _torch_model_keys() -> list[str]:
    return [k for k in list_models() if "torch" in get_model_spec(k).requires]


def _torch_local_model_keys() -> list[str]:
    return [k for k in _torch_model_keys() if get_model_spec(k).interface == "local"]


def _torch_global_model_keys() -> list[str]:
    return [k for k in _torch_model_keys() if get_model_spec(k).interface == "global"]


def _torch_multivariate_model_keys() -> list[str]:
    return [k for k in _torch_model_keys() if get_model_spec(k).interface == "multivariate"]


def test_torch_models_are_registered_as_optional():
    for key in _torch_model_keys():
        spec = get_model_spec(key)
        assert "torch" in spec.requires


def test_wave1a_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in WAVE1A_TORCH_LOCAL_KEYS:
        assert key in keys


def test_lightweight_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in LIGHTWEIGHT_TORCH_LOCAL_KEYS:
        assert key in keys


def test_decomposition_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in DECOMP_TORCH_LOCAL_KEYS:
        assert key in keys


def test_wave1b_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in WAVE1B_TORCH_LOCAL_KEYS:
        assert key in keys


def test_retnet_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in RETENTION_TORCH_LOCAL_KEYS:
        assert key in keys


def test_retnet_torch_global_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_global_model_keys())
    for key in RETENTION_TORCH_GLOBAL_KEYS:
        assert key in keys


def test_timexer_torch_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_model_keys())
    for key in TIME_XER_TORCH_KEYS:
        assert key in keys


def test_state_space_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in STATE_SPACE_TORCH_LOCAL_KEYS:
        assert key in keys


def test_continuous_time_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in CT_RNN_TORCH_LOCAL_KEYS:
        assert key in keys


def test_revival_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in REVIVAL_TORCH_LOCAL_KEYS:
        assert key in keys


def test_ssm_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in SSM_TORCH_LOCAL_KEYS:
        assert key in keys


def test_recurrent_revival_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in RECURRENT_REVIVAL_TORCH_LOCAL_KEYS:
        assert key in keys


def test_latent_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in LATENT_TORCH_LOCAL_KEYS:
        assert key in keys


def test_segmented_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in SEGMENTED_TORCH_LOCAL_KEYS:
        assert key in keys


def test_modern_conv_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in MODERN_CONV_TORCH_LOCAL_KEYS:
        assert key in keys


def test_basis_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in BASIS_TORCH_LOCAL_KEYS:
        assert key in keys


def test_grid_recurrent_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in GRID_RECURRENT_TORCH_LOCAL_KEYS:
        assert key in keys


def test_lag_graph_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in LAG_GRAPH_TORCH_LOCAL_KEYS:
        assert key in keys


def test_multiscale_routing_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in MULTISCALE_ROUTING_TORCH_LOCAL_KEYS:
        assert key in keys


def test_patch_ssm_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in PATCH_SSM_TORCH_LOCAL_KEYS:
        assert key in keys


def test_patch_mixer_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in PATCH_MIXER_TORCH_LOCAL_KEYS:
        assert key in keys


def test_frequency_interpolation_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in FREQUENCY_INTERPOLATION_TORCH_LOCAL_KEYS:
        assert key in keys


def test_sequence_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in SEQUENCE_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_capacity_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in CAPACITY_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_foundation_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in FOUNDATION_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_recurrent_stateful_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in RECURRENT_STATEFUL_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_transformer_ssm_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in TRANSFORMER_SSM_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_remaining_baseline_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in REMAINING_BASELINE_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_local_lstnet_preset_torch_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in LOCAL_LSTNET_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_structured_rnn_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in STRUCTURED_RNN_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_recursive_rnn_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in RECURSIVE_RNN_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_probabilistic_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in PROBABILISTIC_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_reservoir_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in RESERVOIR_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_retnet_recursive_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in RETNET_RECURSIVE_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_canonical_rnn_zoo_preset_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in CANONICAL_RNN_ZOO_PRESET_TORCH_LOCAL_KEYS:
        assert key in keys


def test_training_strategy_torch_local_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_local_model_keys())
    for key in TRAINING_STRATEGY_TORCH_LOCAL_KEYS:
        assert key in keys


def test_training_strategy_torch_global_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_global_model_keys())
    for key in TRAINING_STRATEGY_TORCH_GLOBAL_KEYS:
        assert key in keys


def test_global_preset_torch_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_global_model_keys())
    for key in GLOBAL_PRESET_TORCH_KEYS:
        assert key in keys


def test_torch_multivariate_models_are_covered_by_optional_dep_paths():
    keys = set(_torch_multivariate_model_keys())
    for key in TORCH_MULTIVARIATE_KEYS:
        assert key in keys


def test_torch_models_raise_importerror_when_torch_missing():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test targets the missing-dep path")

    for key in _torch_local_model_keys():
        f = make_forecaster(key)
        with pytest.raises(ImportError):
            f([1.0, 2.0, 3.0], 2)

    for key in _torch_global_model_keys():
        import pandas as pd

        g = make_global_forecaster(key)
        with pytest.raises(ImportError):
            g(
                pd.DataFrame(
                    {
                        "unique_id": ["s0", "s0", "s0"],
                        "ds": pd.date_range("2020-01-01", periods=3, freq="D"),
                        "y": [1.0, 2.0, 3.0],
                    }
                ),
                pd.Timestamp("2020-01-02"),
                2,
            )

    for key in _torch_multivariate_model_keys():
        mv = make_multivariate_forecaster(key)
        with pytest.raises(ImportError):
            mv(np.ones((8, 2), dtype=float), 2)


def test_torch_models_smoke_when_installed():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    y = np.sin(np.arange(120, dtype=float) / 4.0) + 0.05 * np.arange(120, dtype=float)

    cases = [
        ("torch-mlp-direct", {"lags": 24, "hidden_sizes": (16,), "epochs": 8, "batch_size": 16}),
        ("torch-lstm-direct", {"lags": 24, "hidden_size": 16, "epochs": 8, "batch_size": 16}),
        ("torch-gru-direct", {"lags": 24, "hidden_size": 16, "epochs": 8, "batch_size": 16}),
        (
            "torch-tcn-direct",
            {"lags": 24, "channels": (8, 8), "kernel_size": 3, "epochs": 8, "batch_size": 16},
        ),
        (
            "torch-nbeats-direct",
            {
                "lags": 24,
                "num_blocks": 2,
                "num_layers": 2,
                "layer_width": 32,
                "epochs": 8,
                "batch_size": 16,
            },
        ),
        ("torch-nlinear-direct", {"lags": 24, "epochs": 8, "batch_size": 16}),
        ("torch-dlinear-direct", {"lags": 32, "ma_window": 7, "epochs": 8, "batch_size": 16}),
        (
            "torch-transformer-direct",
            {
                "lags": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "epochs": 6,
                "batch_size": 16,
            },
        ),
        (
            "torch-patchtst-direct",
            {
                "lags": 64,
                "patch_len": 8,
                "stride": 4,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "epochs": 6,
                "batch_size": 16,
            },
        ),
        (
            "torch-tsmixer-direct",
            {
                "lags": 32,
                "d_model": 16,
                "num_blocks": 2,
                "token_mixing_hidden": 32,
                "channel_mixing_hidden": 32,
                "epochs": 6,
                "batch_size": 16,
            },
        ),
        (
            "torch-tinytimemixer-direct",
            {
                "lags": 48,
                "patch_len": 6,
                "d_model": 16,
                "num_blocks": 2,
                "token_mixing_hidden": 32,
                "channel_mixing_hidden": 32,
                "epochs": 6,
                "batch_size": 16,
            },
        ),
        (
            "torch-cnn-direct",
            {
                "lags": 32,
                "channels": (8, 8),
                "kernel_size": 3,
                "epochs": 6,
                "batch_size": 16,
            },
        ),
        (
            "torch-bilstm-direct",
            {"lags": 24, "hidden_size": 8, "num_layers": 1, "epochs": 6, "batch_size": 16},
        ),
        (
            "torch-fnet-direct",
            {
                "lags": 32,
                "d_model": 16,
                "num_layers": 1,
                "dim_feedforward": 32,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
        (
            "torch-fits-direct",
            {
                "lags": 48,
                "low_freq_bins": 8,
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.1,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
        (
            "torch-linear-attn-direct",
            {
                "lags": 32,
                "d_model": 16,
                "num_layers": 1,
                "dim_feedforward": 32,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
        (
            "torch-perceiver-direct",
            {
                "lags": 64,
                "d_model": 32,
                "latent_len": 16,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
        (
            "torch-segrnn-direct",
            {
                "lags": 72,
                "segment_len": 12,
                "d_model": 32,
                "hidden_size": 32,
                "num_layers": 1,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
        (
            "torch-moderntcn-direct",
            {
                "lags": 96,
                "patch_len": 8,
                "d_model": 32,
                "num_blocks": 2,
                "expansion_factor": 2.0,
                "kernel_size": 9,
                "dropout": 0.1,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
        (
            "torch-basisformer-direct",
            {
                "lags": 96,
                "patch_len": 8,
                "d_model": 32,
                "num_bases": 16,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.1,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
        (
            "torch-witran-direct",
            {
                "lags": 96,
                "grid_cols": 12,
                "d_model": 32,
                "hidden_size": 32,
                "nhead": 4,
                "num_layers": 1,
                "dropout": 0.1,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
        (
            "torch-crossgnn-direct",
            {
                "lags": 96,
                "d_model": 32,
                "num_blocks": 2,
                "top_k": 8,
                "dropout": 0.1,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
        (
            "torch-pathformer-direct",
            {
                "lags": 96,
                "d_model": 32,
                "expert_patch_lens": (4, 8, 16),
                "num_blocks": 2,
                "top_k": 2,
                "dropout": 0.1,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
        (
            "torch-timesmamba-direct",
            {
                "lags": 96,
                "patch_len": 8,
                "d_model": 32,
                "state_size": 32,
                "num_blocks": 2,
                "dropout": 0.1,
                "epochs": 4,
                "batch_size": 16,
            },
        ),
    ]

    for key, params in cases:
        f = make_forecaster(key, **params, seed=0, patience=2, device="cpu")
        yhat = f(y, 5)
        assert yhat.shape == (5,)
        assert np.all(np.isfinite(yhat))


def test_torch_global_models_smoke_when_installed():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    import pandas as pd

    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=80, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.5), ("s2", 0.8)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.01 * np.arange(ds.size)
        y = base + 0.05 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.1).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    cutoff = ds[-6]  # leave 5 days for horizon
    horizon = 5

    cases = [
        (
            "torch-tft-global",
            {"context_length": 32, "epochs": 2, "batch_size": 32, "x_cols": ("promo",)},
        ),
        (
            "torch-informer-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-autoformer-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
                "ma_window": 7,
            },
        ),
        (
            "torch-timexer-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-tsmixer-global",
            {
                "context_length": 32,
                "d_model": 32,
                "num_blocks": 2,
                "token_mixing_hidden": 32,
                "channel_mixing_hidden": 32,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-tcn-global",
            {
                "context_length": 32,
                "channels": (16, 16),
                "kernel_size": 3,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-deepar-global",
            {
                "context_length": 32,
                "hidden_size": 32,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-patchtst-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "patch_len": 8,
                "stride": 4,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-retnet-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "ffn_dim": 64,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-timesnet-global",
            {
                "context_length": 32,
                "d_model": 32,
                "num_layers": 1,
                "top_k": 2,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-seq2seq-lstm-global",
            {
                "context_length": 32,
                "hidden_size": 32,
                "num_layers": 1,
                "teacher_forcing": 0.5,
                "teacher_forcing_final": 0.0,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-patchtst-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "patch_len": 8,
                "stride": 4,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-retnet-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "ffn_dim": 64,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-timesnet-global",
            {
                "context_length": 32,
                "d_model": 32,
                "num_layers": 1,
                "top_k": 2,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-seq2seq-lstm-global",
            {
                "context_length": 32,
                "hidden_size": 32,
                "num_layers": 1,
                "teacher_forcing": 0.5,
                "teacher_forcing_final": 0.0,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
    ]

    for key, params in cases:
        g = make_global_forecaster(key, **params, seed=0, patience=2, device="cpu")
        pred = g(long_df, cutoff, horizon)
        assert set(pred.columns) >= {"unique_id", "ds", "yhat"}
        assert len(pred) > 0
        assert np.all(np.isfinite(pred["yhat"].to_numpy(dtype=float)))


def test_torch_global_models_support_static_covariates_when_installed():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; smoke test requires it")

    import pandas as pd

    rng = np.random.default_rng(1)
    ds = pd.date_range("2020-01-01", periods=80, freq="D")
    cutoff = ds[-6]
    horizon = 5

    rows = []
    for uid, amp, store_size in [("s0", 1.0, 10.0), ("s1", 1.5, 20.0), ("s2", 0.8, 30.0)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.01 * np.arange(ds.size)
        y = base + 0.05 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.1).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            is_future = t > cutoff
            rows.append(
                {
                    "unique_id": uid,
                    "ds": t,
                    "y": np.nan if is_future else float(yv),
                    "promo": float(pv),
                    "store_size": np.nan if is_future else store_size,
                }
            )
    long_df = pd.DataFrame(rows)

    cases = [
        (
            "torch-tft-global",
            {"context_length": 32, "epochs": 2, "batch_size": 32, "x_cols": ("promo",)},
        ),
        (
            "torch-informer-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-autoformer-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
                "ma_window": 7,
            },
        ),
        (
            "torch-timexer-global",
            {
                "context_length": 32,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-tsmixer-global",
            {
                "context_length": 32,
                "d_model": 32,
                "num_blocks": 2,
                "token_mixing_hidden": 32,
                "channel_mixing_hidden": 32,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-tcn-global",
            {
                "context_length": 32,
                "channels": (16, 16),
                "kernel_size": 3,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
        (
            "torch-deepar-global",
            {
                "context_length": 32,
                "hidden_size": 32,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 32,
                "x_cols": ("promo",),
            },
        ),
    ]

    for key, params in cases:
        g = make_global_forecaster(
            key,
            **params,
            static_cols=("store_size",),
            seed=0,
            patience=2,
            device="cpu",
        )
        pred = g(long_df, cutoff, horizon)
        assert set(pred.columns) >= {"unique_id", "ds", "yhat"}
        assert len(pred) > 0
        assert np.all(np.isfinite(pred["yhat"].to_numpy(dtype=float)))
