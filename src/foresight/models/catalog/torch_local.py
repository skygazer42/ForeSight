from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..specs import ForecasterFn, ModelSpec


_DROPOUT_HELP = "Dropout probability in [0,1)"
_RNN_DROPOUT_HELP = "Dropout probability in [0,1) (only if num_layers>1)"
_SEG_RNN_DROPOUT_HELP = "Dropout probability in [0,1) (only if num_layers>1 inside GRU)"
_LAG_WINDOW_HELP = "Lag window length"
_GRU_HIDDEN_HELP = "GRU hidden size"
_GRU_LAYERS_HELP = "Number of GRU layers"
_CONV_KERNEL_SIZE_HELP = "Conv kernel size"
_RESIDUAL_BLOCKS_HELP = "Number of residual blocks"
_MA_WINDOW_HELP = "Moving average window size for decomposition"
_ATTENTION_HEADS_HELP = "Number of attention heads"
_ENCODER_LAYERS_HELP = "Number of encoder layers"
_ENCODER_FF_DIM_HELP = "Feed-forward dimension in the encoder"
_TRANSFORMER_EMBED_DIM_HELP = "Transformer embedding dimension"
_TRANSFORMER_ENCODER_LAYERS_HELP = "Number of Transformer encoder layers"
_FEED_FORWARD_DIM_HELP = "Feed-forward dimension"
_MODEL_DIMENSION_HELP = "Model dimension"
_HIDDEN_MODEL_DIM_HELP = "Hidden model dimension"
_RECURRENT_HIDDEN_SIZE_HELP = "Recurrent hidden size"
_STATE_SPACE_MODEL_DIM_HELP = "State-space model dimension"
_RNN_HIDDEN_SIZE_HELP = "RNN hidden size"
_STACKED_RNN_LAYERS_HELP = "Number of stacked RNN layers"
_POOLING_HELP = "Pooling: last, mean, max"
_EMBEDDING_DIM_HELP = "Embedding dimension"
_PATCH_LEN_HELP = "Number of timesteps per patch token"
_PATCH_EMBED_DIM_HELP = "Patch embedding dimension"
_PATCH_OR_LATENT_DIM_HELP = "Patch embedding / latent dimension"
_SEGMENT_LEN_HELP = "Segment length used to chunk the lag window"
_SEGMENT_EMBED_DIM_HELP = "Segment embedding dimension"
_SEGMENT_GRU_HIDDEN_HELP = "GRU hidden size over segment tokens"
_SEGMENT_GRU_LAYERS_HELP = "Number of stacked GRU layers"
_LATENT_TOKEN_DIM_HELP = "Latent and token embedding dimension"
_LATENT_TOKEN_COUNT_HELP = "Number of learned latent tokens"
_LATENT_ATTENTION_HEADS_HELP = "Number of cross-attention / latent attention heads"
_LATENT_ENCODER_LAYERS_HELP = "Number of latent Transformer encoder layers"
_LATENT_FF_DIM_HELP = "Latent Transformer feed-forward dimension"
_MODERNTCN_BLOCKS_HELP = "Number of ModernTCN mixer blocks"
_CHANNEL_MLP_EXPANSION_HELP = "Expansion ratio used by the channel MLP"
_PATCH_KERNEL_SIZE_HELP = "Odd depthwise convolution kernel size over patch tokens"
_BASISFORMER_BASES_HELP = "Number of learned basis vectors in the dictionary"
_WITRAN_GRID_COLS_HELP = "Number of columns in the 2D lag grid"
_WITRAN_CELL_DIM_HELP = "Cell embedding / decoder dimension"
_WITRAN_HIDDEN_HELP = "Hidden size of the coupled row/column recurrent states"
_WITRAN_ATTENTION_HEADS_HELP = "Number of decoder attention heads"
_WITRAN_BLOCKS_HELP = "Number of stacked grid recurrent blocks"
_LAG_GRAPH_EMBED_DIM_HELP = "Lag-node embedding dimension"
_LAG_GRAPH_BLOCKS_HELP = "Number of lag-graph mixer blocks"
_LAG_GRAPH_TOP_K_HELP = "Top-k learned neighbors kept per lag node"
_PATHFORMER_DIM_HELP = "Shared expert/context embedding dimension"
_PATHFORMER_PATCH_LENS_HELP = "Patch lengths for the multi-scale experts"
_ROUTING_BLOCKS_HELP = "Number of routing blocks"
_PATHFORMER_TOP_K_HELP = "Top-k experts selected by the router per block"
_TIMESMAMBA_STATE_SIZE_HELP = "Latent state size in the recurrent state-space mixer"
_TIMESMAMBA_BLOCKS_HELP = "Number of stacked state-space mixer blocks"


def build_torch_local_catalog(context: Any) -> dict[str, Any]:
    model_spec = context.ModelSpec
    _TORCH_COMMON_DEFAULTS = context._TORCH_COMMON_DEFAULTS
    _TORCH_COMMON_PARAM_HELP = context._TORCH_COMMON_PARAM_HELP
    _factory_torch_attn_gru_direct = context._factory_torch_attn_gru_direct
    _factory_torch_autoformer_direct = context._factory_torch_autoformer_direct
    _factory_torch_bigru_direct = context._factory_torch_bigru_direct
    _factory_torch_basisformer_direct = context._factory_torch_basisformer_direct
    _factory_torch_bilstm_direct = context._factory_torch_bilstm_direct
    _factory_torch_cfc_direct = context._factory_torch_cfc_direct
    _factory_torch_cnn_direct = context._factory_torch_cnn_direct
    _factory_torch_crossgnn_direct = context._factory_torch_crossgnn_direct
    _factory_torch_crossformer_direct = context._factory_torch_crossformer_direct
    _factory_torch_deepar_recursive = context._factory_torch_deepar_recursive
    _factory_torch_dilated_rnn_direct = context._factory_torch_dilated_rnn_direct
    _factory_torch_dlinear_direct = context._factory_torch_dlinear_direct
    _factory_torch_esrnn_direct = context._factory_torch_esrnn_direct
    _factory_torch_etsformer_direct = context._factory_torch_etsformer_direct
    _factory_torch_fedformer_direct = context._factory_torch_fedformer_direct
    _factory_torch_film_direct = context._factory_torch_film_direct
    _factory_torch_fnet_direct = context._factory_torch_fnet_direct
    _factory_torch_frets_direct = context._factory_torch_frets_direct
    _factory_torch_gmlp_direct = context._factory_torch_gmlp_direct
    _factory_torch_griffin_direct = context._factory_torch_griffin_direct
    _factory_torch_gru_direct = context._factory_torch_gru_direct
    _factory_torch_hawk_direct = context._factory_torch_hawk_direct
    _factory_torch_hyena_direct = context._factory_torch_hyena_direct
    _factory_torch_inception_direct = context._factory_torch_inception_direct
    _factory_torch_informer_direct = context._factory_torch_informer_direct
    _factory_torch_itransformer_direct = context._factory_torch_itransformer_direct
    _factory_torch_kan_direct = context._factory_torch_kan_direct
    _factory_torch_koopa_direct = context._factory_torch_koopa_direct
    _factory_torch_lightts_direct = context._factory_torch_lightts_direct
    _factory_torch_linear_attn_direct = context._factory_torch_linear_attn_direct
    _factory_torch_lmu_direct = context._factory_torch_lmu_direct
    _factory_torch_lstm_direct = context._factory_torch_lstm_direct
    _factory_torch_ltc_direct = context._factory_torch_ltc_direct
    _factory_torch_mamba2_direct = context._factory_torch_mamba2_direct
    _factory_torch_mamba_direct = context._factory_torch_mamba_direct
    _factory_torch_micn_direct = context._factory_torch_micn_direct
    _factory_torch_moderntcn_direct = context._factory_torch_moderntcn_direct
    _factory_torch_mlp_direct = context._factory_torch_mlp_direct
    _factory_torch_nbeats_direct = context._factory_torch_nbeats_direct
    _factory_torch_nhits_direct = context._factory_torch_nhits_direct
    _factory_torch_nlinear_direct = context._factory_torch_nlinear_direct
    _factory_torch_nonstationary_transformer_direct = (
        context._factory_torch_nonstationary_transformer_direct
    )
    _factory_torch_patchtst_direct = context._factory_torch_patchtst_direct
    _factory_torch_pathformer_direct = context._factory_torch_pathformer_direct
    _factory_torch_perceiver_direct = context._factory_torch_perceiver_direct
    _factory_torch_pyraformer_direct = context._factory_torch_pyraformer_direct
    _factory_torch_qrnn_recursive = context._factory_torch_qrnn_recursive
    _factory_torch_resnet1d_direct = context._factory_torch_resnet1d_direct
    _factory_torch_retnet_direct = context._factory_torch_retnet_direct
    _factory_torch_retnet_recursive = context._factory_torch_retnet_recursive
    _factory_torch_rwkv_direct = context._factory_torch_rwkv_direct
    _factory_torch_s4_direct = context._factory_torch_s4_direct
    _factory_torch_s4d_direct = context._factory_torch_s4d_direct
    _factory_torch_s5_direct = context._factory_torch_s5_direct
    _factory_torch_samformer_direct = context._factory_torch_samformer_direct
    _factory_torch_scinet_direct = context._factory_torch_scinet_direct
    _factory_torch_segrnn_direct = context._factory_torch_segrnn_direct
    _factory_torch_sparsetsf_direct = context._factory_torch_sparsetsf_direct
    _factory_torch_tcn_direct = context._factory_torch_tcn_direct
    _factory_torch_tft_direct = context._factory_torch_tft_direct
    _factory_torch_tide_direct = context._factory_torch_tide_direct
    _factory_torch_timemixer_direct = context._factory_torch_timemixer_direct
    _factory_torch_timesmamba_direct = context._factory_torch_timesmamba_direct
    _factory_torch_timesnet_direct = context._factory_torch_timesnet_direct
    _factory_torch_timexer_direct = context._factory_torch_timexer_direct
    _factory_torch_transformer_direct = context._factory_torch_transformer_direct
    _factory_torch_tsmixer_direct = context._factory_torch_tsmixer_direct
    _factory_torch_witran_direct = context._factory_torch_witran_direct
    _factory_torch_wavenet_direct = context._factory_torch_wavenet_direct
    _factory_torch_xlstm_direct = context._factory_torch_xlstm_direct
    return {
        "torch-mlp-direct": model_spec(
            key="torch-mlp-direct",
            description="Torch MLP on lag features (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_mlp_direct,
            default_params={
                "lags": 24,
                "hidden_sizes": (64, 64),
                "dropout": 0.0,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_sizes": "Hidden layer sizes (e.g. 64,64)",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-lstm-direct": model_spec(
            key="torch-lstm-direct",
            description="Torch LSTM on lag windows (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_lstm_direct,
            default_params={
                "lags": 24,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_size": "LSTM hidden size",
                "num_layers": "Number of LSTM layers",
                "dropout": _RNN_DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-gru-direct": model_spec(
            key="torch-gru-direct",
            description="Torch GRU on lag windows (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_gru_direct,
            default_params={
                "lags": 24,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_size": _GRU_HIDDEN_HELP,
                "num_layers": _GRU_LAYERS_HELP,
                "dropout": _RNN_DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-tcn-direct": model_spec(
            key="torch-tcn-direct",
            description="Torch TCN on lag windows (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_tcn_direct,
            default_params={
                "lags": 24,
                "channels": (16, 16, 16),
                "kernel_size": 3,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "channels": "Conv channel sizes (e.g. 16,16,16)",
                "kernel_size": _CONV_KERNEL_SIZE_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-nbeats-direct": model_spec(
            key="torch-nbeats-direct",
            description="Torch N-BEATS-style model (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_nbeats_direct,
            default_params={
                "lags": 48,
                "num_blocks": 3,
                "num_layers": 2,
                "layer_width": 64,
                "dropout": 0.0,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "num_blocks": _RESIDUAL_BLOCKS_HELP,
                "num_layers": "Hidden layers per block",
                "layer_width": "Hidden width per layer",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-nlinear-direct": model_spec(
            key="torch-nlinear-direct",
            description="Torch NLinear-style baseline (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_nlinear_direct,
            default_params={
                "lags": 48,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-dlinear-direct": model_spec(
            key="torch-dlinear-direct",
            description="Torch DLinear-style baseline (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_dlinear_direct,
            default_params={
                "lags": 48,
                "ma_window": 25,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "ma_window": _MA_WINDOW_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-transformer-direct": model_spec(
            key="torch-transformer-direct",
            description="Torch Transformer encoder (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_transformer_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _TRANSFORMER_EMBED_DIM_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _TRANSFORMER_ENCODER_LAYERS_HELP,
                "dim_feedforward": _FEED_FORWARD_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-informer-direct": model_spec(
            key="torch-informer-direct",
            description="Torch Informer-style (lite) encoder (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_informer_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _TRANSFORMER_EMBED_DIM_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _ENCODER_LAYERS_HELP,
                "dim_feedforward": _FEED_FORWARD_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-autoformer-direct": model_spec(
            key="torch-autoformer-direct",
            description="Torch Autoformer-style decomposition encoder (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_autoformer_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                "ma_window": 7,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _TRANSFORMER_EMBED_DIM_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _ENCODER_LAYERS_HELP,
                "dim_feedforward": _FEED_FORWARD_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                "ma_window": _MA_WINDOW_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-nonstationary-transformer-direct": model_spec(
            key="torch-nonstationary-transformer-direct",
            description="Torch Non-stationary Transformer-style model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_nonstationary_transformer_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _TRANSFORMER_EMBED_DIM_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _ENCODER_LAYERS_HELP,
                "dim_feedforward": _FEED_FORWARD_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-fedformer-direct": model_spec(
            key="torch-fedformer-direct",
            description="Torch FEDformer-style decomposition + frequency mixing model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_fedformer_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "ffn_dim": 256,
                "modes": 16,
                "ma_window": 7,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _MODEL_DIMENSION_HELP,
                "num_layers": "Number of frequency-mixing blocks",
                "ffn_dim": _FEED_FORWARD_DIM_HELP,
                "modes": "Number of retained low-frequency Fourier modes",
                "ma_window": _MA_WINDOW_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-itransformer-direct": model_spec(
            key="torch-itransformer-direct",
            description="Torch iTransformer-style inverted-token encoder (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_itransformer_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _TRANSFORMER_EMBED_DIM_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _ENCODER_LAYERS_HELP,
                "dim_feedforward": _FEED_FORWARD_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-timesnet-direct": model_spec(
            key="torch-timesnet-direct",
            description="Torch TimesNet-style period-mixing Conv2D model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_timesnet_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "top_k": 3,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _MODEL_DIMENSION_HELP,
                "num_layers": "Number of TimesBlocks",
                "top_k": "Number of dominant periods to mix",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-tft-direct": model_spec(
            key="torch-tft-direct",
            description="Torch TFT-style hybrid encoder (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_tft_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "lstm_layers": 1,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _MODEL_DIMENSION_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "lstm_layers": "Number of stacked LSTM layers",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-timemixer-direct": model_spec(
            key="torch-timemixer-direct",
            description="Torch TimeMixer-style multiscale mixer (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_timemixer_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_blocks": 4,
                "multiscale_factors": (1, 2, 4),
                "token_mixing_hidden": 128,
                "channel_mixing_hidden": 128,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": "Mixer embedding dimension",
                "num_blocks": "Number of mixer blocks",
                "multiscale_factors": "Temporal smoothing scales, e.g. 1,2,4",
                "token_mixing_hidden": "Token-mixing MLP hidden size",
                "channel_mixing_hidden": "Channel-mixing MLP hidden size",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-sparsetsf-direct": model_spec(
            key="torch-sparsetsf-direct",
            description="Torch SparseTSF-style sparse seasonal readout (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_sparsetsf_direct,
            default_params={
                "lags": 192,
                "period_len": 24,
                "d_model": 64,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "period_len": "Seasonal stride for sparse lag selection",
                "d_model": "Hidden width of the sparse projection head",
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-timexer-direct": model_spec(
            key="torch-timexer-direct",
            description="Torch TimeXer-style exogenous-aware transformer (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_timexer_direct,
            default_params={
                "x_cols": (),
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "x_cols": "Required future covariate columns for forecast_model_long_df / forecast csv",
                "lags": _LAG_WINDOW_HELP,
                "d_model": _MODEL_DIMENSION_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": "Number of encoder / cross-attention blocks",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
            capability_overrides={"requires_future_covariates": True},
        ),
        "torch-lmu-direct": model_spec(
            key="torch-lmu-direct",
            description="Torch LMU-style recurrent memory model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_lmu_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "memory_dim": 32,
                "num_layers": 1,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _HIDDEN_MODEL_DIM_HELP,
                "memory_dim": "Continuous-time memory state dimension",
                "num_layers": "Number of stacked LMU blocks",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-ltc-direct": model_spec(
            key="torch-ltc-direct",
            description="Torch LTC-style liquid time-constant recurrent model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_ltc_direct,
            default_params={
                "lags": 96,
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_size": _RECURRENT_HIDDEN_SIZE_HELP,
                "num_layers": "Number of stacked LTC blocks",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-cfc-direct": model_spec(
            key="torch-cfc-direct",
            description="Torch CfC-style closed-form continuous-time recurrent model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_cfc_direct,
            default_params={
                "lags": 96,
                "hidden_size": 64,
                "num_layers": 1,
                "backbone_hidden": 128,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_size": _RECURRENT_HIDDEN_SIZE_HELP,
                "num_layers": "Number of stacked CfC blocks",
                "backbone_hidden": "Closed-form backbone hidden size",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-xlstm-direct": model_spec(
            key="torch-xlstm-direct",
            description="Torch xLSTM-style expanded-gate recurrent model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_xlstm_direct,
            default_params={
                "lags": 96,
                "hidden_size": 64,
                "num_layers": 1,
                "proj_factor": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_size": _RECURRENT_HIDDEN_SIZE_HELP,
                "num_layers": "Number of stacked xLSTM blocks",
                "proj_factor": "Expansion factor for the gated inner projection",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-griffin-direct": model_spec(
            key="torch-griffin-direct",
            description="Torch Griffin-style recurrent hybrid model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_griffin_direct,
            default_params={
                "lags": 96,
                "hidden_size": 64,
                "num_layers": 1,
                "conv_kernel": 3,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_size": _RECURRENT_HIDDEN_SIZE_HELP,
                "num_layers": "Number of stacked Griffin blocks",
                "conv_kernel": "Depthwise convolution kernel size (>=1)",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-hawk-direct": model_spec(
            key="torch-hawk-direct",
            description="Torch Hawk-style gated recurrent mixer (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_hawk_direct,
            default_params={
                "lags": 96,
                "hidden_size": 64,
                "num_layers": 1,
                "expansion_factor": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_size": _RECURRENT_HIDDEN_SIZE_HELP,
                "num_layers": "Number of stacked Hawk blocks",
                "expansion_factor": "Expansion factor for the recurrent mixer",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-s4d-direct": model_spec(
            key="torch-s4d-direct",
            description="Torch S4D-style diagonal state-space model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_s4d_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _STATE_SPACE_MODEL_DIM_HELP,
                "num_layers": "Number of stacked diagonal SSM blocks",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-s4-direct": model_spec(
            key="torch-s4-direct",
            description="Torch S4-style structured state-space model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_s4_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "state_dim": 32,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _STATE_SPACE_MODEL_DIM_HELP,
                "num_layers": "Number of stacked structured SSM blocks",
                "state_dim": "Latent state dimension per block",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-s5-direct": model_spec(
            key="torch-s5-direct",
            description="Torch S5-style multi-state-space model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_s5_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "state_dim": 32,
                "heads": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _STATE_SPACE_MODEL_DIM_HELP,
                "num_layers": "Number of stacked multi-state SSM blocks",
                "state_dim": "Latent state dimension per head",
                "heads": "Number of state-space heads",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-mamba2-direct": model_spec(
            key="torch-mamba2-direct",
            description="Torch Mamba-2-style selective state-space refinement (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_mamba2_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "conv_kernel": 3,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _STATE_SPACE_MODEL_DIM_HELP,
                "num_layers": "Number of stacked Mamba-2 refinement blocks",
                "conv_kernel": "Causal depthwise conv kernel size (>=1)",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-lightts-direct": model_spec(
            key="torch-lightts-direct",
            description="Torch LightTS-style dual-sampling MLP (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_lightts_direct,
            default_params={
                "lags": 96,
                "chunk_len": 12,
                "d_model": 64,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "chunk_len": "Chunk size for continuous / interval sampling views",
                "d_model": "Hidden width of the fused sampling head",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-frets-direct": model_spec(
            key="torch-frets-direct",
            description="Torch FreTS-style frequency-domain MLP (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_frets_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "top_k_freqs": 8,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": "Hidden width of the frequency MLP",
                "num_layers": "Number of frequency-domain MLP blocks",
                "top_k_freqs": "Number of non-DC frequencies retained per window",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-film-direct": model_spec(
            key="torch-film-direct",
            description="Torch FiLM-style decomposition + long-filter mixer (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_film_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "ma_window": 7,
                "kernel_size": 7,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _HIDDEN_MODEL_DIM_HELP,
                "num_layers": "Number of long-filter mixer blocks",
                "ma_window": _MA_WINDOW_HELP,
                "kernel_size": "Odd depthwise filter width for the seasonal mixer",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-micn-direct": model_spec(
            key="torch-micn-direct",
            description="Torch MICN-style multiscale convolutional decomposition model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_micn_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "kernel_sizes": (3, 5, 7),
                "ma_window": 7,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _HIDDEN_MODEL_DIM_HELP,
                "num_layers": "Number of multiscale convolution blocks",
                "kernel_sizes": "Odd multiscale Conv1D kernel sizes, e.g. 3,5,7",
                "ma_window": _MA_WINDOW_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-koopa-direct": model_spec(
            key="torch-koopa-direct",
            description="Torch Koopa-style decomposition + latent linear dynamics model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_koopa_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "latent_dim": 32,
                "num_blocks": 2,
                "ma_window": 7,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": "Hidden token dimension",
                "latent_dim": "Latent Koopman state dimension",
                "num_blocks": "Number of seasonal encoder blocks",
                "ma_window": _MA_WINDOW_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-samformer-direct": model_spec(
            key="torch-samformer-direct",
            description="Torch SAMformer-style linear-attention + adaptive mixing model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_samformer_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _MODEL_DIMENSION_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": "Number of linear-attention mixer blocks",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-retnet-direct": model_spec(
            key="torch-retnet-direct",
            description="Torch RetNet-style retention network (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_retnet_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "ffn_dim": 128,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _MODEL_DIMENSION_HELP,
                "nhead": "Number of retention heads",
                "num_layers": "Number of stacked retention blocks",
                "ffn_dim": "Feed-forward hidden width inside each retention block",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-retnet-recursive": model_spec(
            key="torch-retnet-recursive",
            description="Torch RetNet-style retention network (one-step trained, recursive forecast). Requires PyTorch.",
            factory=_factory_torch_retnet_recursive,
            default_params={
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "ffn_dim": 128,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _MODEL_DIMENSION_HELP,
                "nhead": "Number of retention heads",
                "num_layers": "Number of stacked retention blocks",
                "ffn_dim": "Feed-forward hidden width inside each retention block",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-mamba-direct": model_spec(
            key="torch-mamba-direct",
            description="Torch Mamba-style selective SSM (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_mamba_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "dropout": 0.1,
                "conv_kernel": 3,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _MODEL_DIMENSION_HELP,
                "num_layers": "Number of stacked Mamba blocks",
                "dropout": _DROPOUT_HELP,
                "conv_kernel": "Causal depthwise conv kernel size (>=1)",
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-rwkv-direct": model_spec(
            key="torch-rwkv-direct",
            description="Torch RWKV-style time-mix + channel-mix (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_rwkv_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "ffn_dim": 128,
                "dropout": 0.0,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _MODEL_DIMENSION_HELP,
                "num_layers": "Number of stacked RWKV blocks",
                "ffn_dim": "Channel-mix hidden size",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-hyena-direct": model_spec(
            key="torch-hyena-direct",
            description="Torch Hyena-style long convolution model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_hyena_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "ffn_dim": 128,
                "kernel_size": 64,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _MODEL_DIMENSION_HELP,
                "num_layers": "Number of Hyena blocks",
                "ffn_dim": "Channel-mixing FFN hidden size",
                "kernel_size": "Depthwise causal conv kernel size (>=1)",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-dilated-rnn-direct": model_spec(
            key="torch-dilated-rnn-direct",
            description="Torch Dilated RNN (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_dilated_rnn_direct,
            default_params={
                "lags": 96,
                "cell": "gru",
                "hidden_size": 64,
                "num_layers": 3,
                "dilation_base": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "cell": "Recurrent cell: gru or lstm",
                "hidden_size": "Hidden size / model dimension",
                "num_layers": "Number of dilated recurrent layers",
                "dilation_base": "Dilation base (>=2); dilations are base^layer_index",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-kan-direct": model_spec(
            key="torch-kan-direct",
            description="Torch KAN-style spline MLP (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_kan_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "grid_size": 16,
                "grid_range": 2.0,
                "dropout": 0.1,
                "linear_skip": True,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": "Hidden width of the KAN network",
                "num_layers": "Number of KAN spline layers",
                "grid_size": "Number of spline knots (>=4)",
                "grid_range": "Spline grid range (+/- range) in normalized y units",
                "dropout": _DROPOUT_HELP,
                "linear_skip": "Add a linear skip connection per layer (true/false)",
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-scinet-direct": model_spec(
            key="torch-scinet-direct",
            description="Torch SCINet-style sample-convolution interaction network (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_scinet_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_stages": 3,
                "conv_kernel": 5,
                "ffn_dim": 128,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _MODEL_DIMENSION_HELP,
                "num_stages": "Number of SCINet interaction stages",
                "conv_kernel": "Conv1D kernel size (>=1) inside interaction blocks",
                "ffn_dim": "FFN hidden size inside blocks",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-etsformer-direct": model_spec(
            key="torch-etsformer-direct",
            description="Torch ETSformer-style exponential smoothing + Transformer residual model (lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_etsformer_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                "alpha_init": 0.3,
                "beta_init": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _TRANSFORMER_EMBED_DIM_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _TRANSFORMER_ENCODER_LAYERS_HELP,
                "dim_feedforward": "Transformer FFN dimension",
                "dropout": _DROPOUT_HELP,
                "alpha_init": "Initial smoothing alpha in (0,1) (learned during training)",
                "beta_init": "Initial smoothing beta in (0,1) (learned during training)",
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-esrnn-direct": model_spec(
            key="torch-esrnn-direct",
            description="Torch ESRNN-style hybrid (Holt smoothing + RNN residual, lite) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_esrnn_direct,
            default_params={
                "lags": 96,
                "cell": "gru",
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.1,
                "alpha_init": 0.3,
                "beta_init": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "cell": "RNN cell: gru or lstm",
                "hidden_size": _RNN_HIDDEN_SIZE_HELP,
                "num_layers": _STACKED_RNN_LAYERS_HELP,
                "dropout": _RNN_DROPOUT_HELP,
                "alpha_init": "Initial smoothing alpha in (0,1) (learned during training)",
                "beta_init": "Initial smoothing beta in (0,1) (learned during training)",
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-patchtst-direct": model_spec(
            key="torch-patchtst-direct",
            description="Torch PatchTST-style model (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_patchtst_direct,
            default_params={
                "lags": 192,
                "patch_len": 16,
                "stride": 8,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "patch_len": "Patch length",
                "stride": "Patch stride",
                "d_model": _TRANSFORMER_EMBED_DIM_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _TRANSFORMER_ENCODER_LAYERS_HELP,
                "dim_feedforward": _FEED_FORWARD_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-crossformer-direct": model_spec(
            key="torch-crossformer-direct",
            description="Torch Crossformer-style (lite) multi-scale segmented Transformer encoder (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_crossformer_direct,
            default_params={
                "lags": 192,
                "segment_len": 16,
                "stride": 16,
                "num_scales": 3,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "segment_len": "Base segment length (scale i uses segment_len * 2^i)",
                "stride": "Base segment stride (scale i uses stride * 2^i)",
                "num_scales": "Number of scales (>=1)",
                "d_model": _TRANSFORMER_EMBED_DIM_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _TRANSFORMER_ENCODER_LAYERS_HELP,
                "dim_feedforward": _FEED_FORWARD_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-pyraformer-direct": model_spec(
            key="torch-pyraformer-direct",
            description="Torch Pyraformer-style (lite) pyramid-pooled segmented Transformer encoder (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_pyraformer_direct,
            default_params={
                "lags": 192,
                "segment_len": 16,
                "stride": 16,
                "num_levels": 3,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "segment_len": "Segment length at level-0",
                "stride": "Segment stride at level-0 (>=1)",
                "num_levels": "Number of pyramid levels (>=1)",
                "d_model": _TRANSFORMER_EMBED_DIM_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _TRANSFORMER_ENCODER_LAYERS_HELP,
                "dim_feedforward": _FEED_FORWARD_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-perceiver-direct": model_spec(
            key="torch-perceiver-direct",
            description="Torch Perceiver-style latent cross-attention model (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_perceiver_direct,
            default_params={
                "lags": 192,
                "d_model": 64,
                "latent_len": 32,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _LATENT_TOKEN_DIM_HELP,
                "latent_len": _LATENT_TOKEN_COUNT_HELP,
                "nhead": _LATENT_ATTENTION_HEADS_HELP,
                "num_layers": _LATENT_ENCODER_LAYERS_HELP,
                "dim_feedforward": _LATENT_FF_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-perceiver-deep-direct": model_spec(
            key="torch-perceiver-deep-direct",
            description="Torch Perceiver-style latent cross-attention model, deeper config (4 latent layers) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_perceiver_direct,
            default_params={
                "lags": 192,
                "d_model": 64,
                "latent_len": 32,
                "nhead": 4,
                "num_layers": 4,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _LATENT_TOKEN_DIM_HELP,
                "latent_len": _LATENT_TOKEN_COUNT_HELP,
                "nhead": _LATENT_ATTENTION_HEADS_HELP,
                "num_layers": _LATENT_ENCODER_LAYERS_HELP,
                "dim_feedforward": _LATENT_FF_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-perceiver-wide-direct": model_spec(
            key="torch-perceiver-wide-direct",
            description="Torch Perceiver-style latent cross-attention model, wider config (d_model=128) (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_perceiver_direct,
            default_params={
                "lags": 192,
                "d_model": 128,
                "latent_len": 48,
                "nhead": 8,
                "num_layers": 2,
                "dim_feedforward": 512,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _LATENT_TOKEN_DIM_HELP,
                "latent_len": _LATENT_TOKEN_COUNT_HELP,
                "nhead": _LATENT_ATTENTION_HEADS_HELP,
                "num_layers": _LATENT_ENCODER_LAYERS_HELP,
                "dim_feedforward": _LATENT_FF_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-tsmixer-direct": model_spec(
            key="torch-tsmixer-direct",
            description="Torch TSMixer-style model (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_tsmixer_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_blocks": 4,
                "token_mixing_hidden": 128,
                "channel_mixing_hidden": 128,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": "Mixer embedding dimension",
                "num_blocks": "Number of mixer blocks",
                "token_mixing_hidden": "Token-mixing MLP hidden size",
                "channel_mixing_hidden": "Channel-mixing MLP hidden size",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-cnn-direct": model_spec(
            key="torch-cnn-direct",
            description="Torch Conv1D stack (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_cnn_direct,
            default_params={
                "lags": 48,
                "channels": (32, 32, 32),
                "kernel_size": 3,
                "dropout": 0.1,
                "pool": "last",
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "channels": "Conv channel sizes (e.g. 32,32,32)",
                "kernel_size": _CONV_KERNEL_SIZE_HELP,
                "dropout": _DROPOUT_HELP,
                "pool": _POOLING_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-resnet1d-direct": model_spec(
            key="torch-resnet1d-direct",
            description="Torch ResNet-1D (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_resnet1d_direct,
            default_params={
                "lags": 96,
                "channels": 32,
                "num_blocks": 4,
                "kernel_size": 3,
                "dropout": 0.1,
                "pool": "last",
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "channels": "ResNet hidden channels",
                "num_blocks": _RESIDUAL_BLOCKS_HELP,
                "kernel_size": _CONV_KERNEL_SIZE_HELP,
                "dropout": _DROPOUT_HELP,
                "pool": _POOLING_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-wavenet-direct": model_spec(
            key="torch-wavenet-direct",
            description="Torch WaveNet-style gated dilated CNN (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_wavenet_direct,
            default_params={
                "lags": 96,
                "channels": 32,
                "num_layers": 6,
                "kernel_size": 2,
                "dropout": 0.0,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "channels": "Hidden channels",
                "num_layers": "Number of dilated gated layers",
                "kernel_size": _CONV_KERNEL_SIZE_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-bilstm-direct": model_spec(
            key="torch-bilstm-direct",
            description="Torch bidirectional LSTM on lag windows (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_bilstm_direct,
            default_params={
                "lags": 24,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_size": "BiLSTM hidden size (per direction)",
                "num_layers": "Number of stacked BiLSTM layers",
                "dropout": _RNN_DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-bigru-direct": model_spec(
            key="torch-bigru-direct",
            description="Torch bidirectional GRU on lag windows (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_bigru_direct,
            default_params={
                "lags": 24,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_size": "BiGRU hidden size (per direction)",
                "num_layers": "Number of stacked BiGRU layers",
                "dropout": _RNN_DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-attn-gru-direct": model_spec(
            key="torch-attn-gru-direct",
            description="Torch GRU + attention pooling (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_attn_gru_direct,
            default_params={
                "lags": 48,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_size": _GRU_HIDDEN_HELP,
                "num_layers": _GRU_LAYERS_HELP,
                "dropout": _RNN_DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-segrnn-direct": model_spec(
            key="torch-segrnn-direct",
            description="Torch SegRNN-style segmented recurrent model (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_segrnn_direct,
            default_params={
                "lags": 96,
                "segment_len": 12,
                "d_model": 64,
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "segment_len": _SEGMENT_LEN_HELP,
                "d_model": _SEGMENT_EMBED_DIM_HELP,
                "hidden_size": _SEGMENT_GRU_HIDDEN_HELP,
                "num_layers": _SEGMENT_GRU_LAYERS_HELP,
                "dropout": _SEG_RNN_DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-segrnn-deep-direct": model_spec(
            key="torch-segrnn-deep-direct",
            description="Torch SegRNN-style segmented recurrent model, deeper config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_segrnn_direct,
            default_params={
                "lags": 96,
                "segment_len": 12,
                "d_model": 64,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "segment_len": _SEGMENT_LEN_HELP,
                "d_model": _SEGMENT_EMBED_DIM_HELP,
                "hidden_size": _SEGMENT_GRU_HIDDEN_HELP,
                "num_layers": _SEGMENT_GRU_LAYERS_HELP,
                "dropout": _SEG_RNN_DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-segrnn-wide-direct": model_spec(
            key="torch-segrnn-wide-direct",
            description="Torch SegRNN-style segmented recurrent model, wider config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_segrnn_direct,
            default_params={
                "lags": 96,
                "segment_len": 12,
                "d_model": 128,
                "hidden_size": 128,
                "num_layers": 1,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "segment_len": _SEGMENT_LEN_HELP,
                "d_model": _SEGMENT_EMBED_DIM_HELP,
                "hidden_size": _SEGMENT_GRU_HIDDEN_HELP,
                "num_layers": _SEGMENT_GRU_LAYERS_HELP,
                "dropout": _SEG_RNN_DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-moderntcn-direct": model_spec(
            key="torch-moderntcn-direct",
            description="Torch ModernTCN-style patchified convolutional mixer (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_moderntcn_direct,
            default_params={
                "lags": 192,
                "patch_len": 8,
                "d_model": 64,
                "num_blocks": 3,
                "expansion_factor": 2.0,
                "kernel_size": 9,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "patch_len": _PATCH_LEN_HELP,
                "d_model": _PATCH_EMBED_DIM_HELP,
                "num_blocks": _MODERNTCN_BLOCKS_HELP,
                "expansion_factor": _CHANNEL_MLP_EXPANSION_HELP,
                "kernel_size": _PATCH_KERNEL_SIZE_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-moderntcn-deep-direct": model_spec(
            key="torch-moderntcn-deep-direct",
            description="Torch ModernTCN-style patchified convolutional mixer, deeper config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_moderntcn_direct,
            default_params={
                "lags": 192,
                "patch_len": 8,
                "d_model": 64,
                "num_blocks": 5,
                "expansion_factor": 2.0,
                "kernel_size": 9,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "patch_len": _PATCH_LEN_HELP,
                "d_model": _PATCH_EMBED_DIM_HELP,
                "num_blocks": _MODERNTCN_BLOCKS_HELP,
                "expansion_factor": _CHANNEL_MLP_EXPANSION_HELP,
                "kernel_size": _PATCH_KERNEL_SIZE_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-moderntcn-wide-direct": model_spec(
            key="torch-moderntcn-wide-direct",
            description="Torch ModernTCN-style patchified convolutional mixer, wider config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_moderntcn_direct,
            default_params={
                "lags": 192,
                "patch_len": 8,
                "d_model": 128,
                "num_blocks": 3,
                "expansion_factor": 2.5,
                "kernel_size": 11,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "patch_len": _PATCH_LEN_HELP,
                "d_model": _PATCH_EMBED_DIM_HELP,
                "num_blocks": _MODERNTCN_BLOCKS_HELP,
                "expansion_factor": _CHANNEL_MLP_EXPANSION_HELP,
                "kernel_size": _PATCH_KERNEL_SIZE_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-basisformer-direct": model_spec(
            key="torch-basisformer-direct",
            description="Torch Basisformer-style learned basis routing model (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_basisformer_direct,
            default_params={
                "lags": 192,
                "patch_len": 8,
                "d_model": 64,
                "num_bases": 16,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "patch_len": _PATCH_LEN_HELP,
                "d_model": _PATCH_OR_LATENT_DIM_HELP,
                "num_bases": _BASISFORMER_BASES_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _ENCODER_LAYERS_HELP,
                "dim_feedforward": _ENCODER_FF_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-basisformer-deep-direct": model_spec(
            key="torch-basisformer-deep-direct",
            description="Torch Basisformer-style learned basis routing model, deeper config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_basisformer_direct,
            default_params={
                "lags": 192,
                "patch_len": 8,
                "d_model": 64,
                "num_bases": 16,
                "nhead": 4,
                "num_layers": 4,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "patch_len": _PATCH_LEN_HELP,
                "d_model": _PATCH_OR_LATENT_DIM_HELP,
                "num_bases": _BASISFORMER_BASES_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _ENCODER_LAYERS_HELP,
                "dim_feedforward": _ENCODER_FF_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-basisformer-wide-direct": model_spec(
            key="torch-basisformer-wide-direct",
            description="Torch Basisformer-style learned basis routing model, wider config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_basisformer_direct,
            default_params={
                "lags": 192,
                "patch_len": 8,
                "d_model": 128,
                "num_bases": 24,
                "nhead": 8,
                "num_layers": 2,
                "dim_feedforward": 512,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "patch_len": _PATCH_LEN_HELP,
                "d_model": _PATCH_OR_LATENT_DIM_HELP,
                "num_bases": _BASISFORMER_BASES_HELP,
                "nhead": _ATTENTION_HEADS_HELP,
                "num_layers": _ENCODER_LAYERS_HELP,
                "dim_feedforward": _ENCODER_FF_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-witran-direct": model_spec(
            key="torch-witran-direct",
            description="Torch WITRAN-style 2D grid recurrent mixer (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_witran_direct,
            default_params={
                "lags": 192,
                "grid_cols": 12,
                "d_model": 64,
                "hidden_size": 64,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "grid_cols": _WITRAN_GRID_COLS_HELP,
                "d_model": _WITRAN_CELL_DIM_HELP,
                "hidden_size": _WITRAN_HIDDEN_HELP,
                "nhead": _WITRAN_ATTENTION_HEADS_HELP,
                "num_layers": _WITRAN_BLOCKS_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-witran-deep-direct": model_spec(
            key="torch-witran-deep-direct",
            description="Torch WITRAN-style 2D grid recurrent mixer, deeper config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_witran_direct,
            default_params={
                "lags": 192,
                "grid_cols": 12,
                "d_model": 64,
                "hidden_size": 64,
                "nhead": 4,
                "num_layers": 4,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "grid_cols": _WITRAN_GRID_COLS_HELP,
                "d_model": _WITRAN_CELL_DIM_HELP,
                "hidden_size": _WITRAN_HIDDEN_HELP,
                "nhead": _WITRAN_ATTENTION_HEADS_HELP,
                "num_layers": _WITRAN_BLOCKS_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-witran-wide-direct": model_spec(
            key="torch-witran-wide-direct",
            description="Torch WITRAN-style 2D grid recurrent mixer, wider config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_witran_direct,
            default_params={
                "lags": 192,
                "grid_cols": 12,
                "d_model": 128,
                "hidden_size": 128,
                "nhead": 8,
                "num_layers": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "grid_cols": _WITRAN_GRID_COLS_HELP,
                "d_model": _WITRAN_CELL_DIM_HELP,
                "hidden_size": _WITRAN_HIDDEN_HELP,
                "nhead": _WITRAN_ATTENTION_HEADS_HELP,
                "num_layers": _WITRAN_BLOCKS_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-crossgnn-direct": model_spec(
            key="torch-crossgnn-direct",
            description="Torch CrossGNN-style lag-graph mixer (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_crossgnn_direct,
            default_params={
                "lags": 192,
                "d_model": 64,
                "num_blocks": 3,
                "top_k": 8,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _LAG_GRAPH_EMBED_DIM_HELP,
                "num_blocks": _LAG_GRAPH_BLOCKS_HELP,
                "top_k": _LAG_GRAPH_TOP_K_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-crossgnn-deep-direct": model_spec(
            key="torch-crossgnn-deep-direct",
            description="Torch CrossGNN-style lag-graph mixer, deeper config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_crossgnn_direct,
            default_params={
                "lags": 192,
                "d_model": 64,
                "num_blocks": 5,
                "top_k": 8,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _LAG_GRAPH_EMBED_DIM_HELP,
                "num_blocks": _LAG_GRAPH_BLOCKS_HELP,
                "top_k": _LAG_GRAPH_TOP_K_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-crossgnn-wide-direct": model_spec(
            key="torch-crossgnn-wide-direct",
            description="Torch CrossGNN-style lag-graph mixer, wider config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_crossgnn_direct,
            default_params={
                "lags": 192,
                "d_model": 128,
                "num_blocks": 3,
                "top_k": 12,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _LAG_GRAPH_EMBED_DIM_HELP,
                "num_blocks": _LAG_GRAPH_BLOCKS_HELP,
                "top_k": _LAG_GRAPH_TOP_K_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-pathformer-direct": model_spec(
            key="torch-pathformer-direct",
            description="Torch Pathformer-style multi-scale expert routing model (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_pathformer_direct,
            default_params={
                "lags": 192,
                "d_model": 64,
                "expert_patch_lens": (4, 8, 16),
                "num_blocks": 3,
                "top_k": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _PATHFORMER_DIM_HELP,
                "expert_patch_lens": _PATHFORMER_PATCH_LENS_HELP,
                "num_blocks": _ROUTING_BLOCKS_HELP,
                "top_k": _PATHFORMER_TOP_K_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-pathformer-deep-direct": model_spec(
            key="torch-pathformer-deep-direct",
            description="Torch Pathformer-style multi-scale expert routing model, deeper config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_pathformer_direct,
            default_params={
                "lags": 192,
                "d_model": 64,
                "expert_patch_lens": (4, 8, 16),
                "num_blocks": 5,
                "top_k": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _PATHFORMER_DIM_HELP,
                "expert_patch_lens": _PATHFORMER_PATCH_LENS_HELP,
                "num_blocks": _ROUTING_BLOCKS_HELP,
                "top_k": _PATHFORMER_TOP_K_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-pathformer-wide-direct": model_spec(
            key="torch-pathformer-wide-direct",
            description="Torch Pathformer-style multi-scale expert routing model, wider config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_pathformer_direct,
            default_params={
                "lags": 192,
                "d_model": 128,
                "expert_patch_lens": (4, 8, 16, 32),
                "num_blocks": 3,
                "top_k": 2,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _PATHFORMER_DIM_HELP,
                "expert_patch_lens": _PATHFORMER_PATCH_LENS_HELP,
                "num_blocks": _ROUTING_BLOCKS_HELP,
                "top_k": _PATHFORMER_TOP_K_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-timesmamba-direct": model_spec(
            key="torch-timesmamba-direct",
            description="Torch TimesMamba-style patch state-space mixer (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_timesmamba_direct,
            default_params={
                "lags": 192,
                "patch_len": 8,
                "d_model": 64,
                "state_size": 64,
                "num_blocks": 3,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "patch_len": _PATCH_LEN_HELP,
                "d_model": _PATCH_EMBED_DIM_HELP,
                "state_size": _TIMESMAMBA_STATE_SIZE_HELP,
                "num_blocks": _TIMESMAMBA_BLOCKS_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-timesmamba-deep-direct": model_spec(
            key="torch-timesmamba-deep-direct",
            description="Torch TimesMamba-style patch state-space mixer, deeper config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_timesmamba_direct,
            default_params={
                "lags": 192,
                "patch_len": 8,
                "d_model": 64,
                "state_size": 64,
                "num_blocks": 5,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "patch_len": _PATCH_LEN_HELP,
                "d_model": _PATCH_EMBED_DIM_HELP,
                "state_size": _TIMESMAMBA_STATE_SIZE_HELP,
                "num_blocks": _TIMESMAMBA_BLOCKS_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-timesmamba-wide-direct": model_spec(
            key="torch-timesmamba-wide-direct",
            description="Torch TimesMamba-style patch state-space mixer, wider config (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_timesmamba_direct,
            default_params={
                "lags": 192,
                "patch_len": 8,
                "d_model": 128,
                "state_size": 128,
                "num_blocks": 3,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "patch_len": _PATCH_LEN_HELP,
                "d_model": _PATCH_EMBED_DIM_HELP,
                "state_size": _TIMESMAMBA_STATE_SIZE_HELP,
                "num_blocks": _TIMESMAMBA_BLOCKS_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-fnet-direct": model_spec(
            key="torch-fnet-direct",
            description="Torch FNet-style (Fourier mixing) model (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_fnet_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 4,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _EMBEDDING_DIM_HELP,
                "num_layers": "Number of FNet layers",
                "dim_feedforward": _FEED_FORWARD_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-linear-attn-direct": model_spec(
            key="torch-linear-attn-direct",
            description="Torch linear-attention encoder (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_linear_attn_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _EMBEDDING_DIM_HELP,
                "num_layers": _ENCODER_LAYERS_HELP,
                "dim_feedforward": _FEED_FORWARD_DIM_HELP,
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-inception-direct": model_spec(
            key="torch-inception-direct",
            description="Torch InceptionTime-style Conv1D model (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_inception_direct,
            default_params={
                "lags": 96,
                "channels": 32,
                "num_blocks": 3,
                "kernel_sizes": (3, 5, 7),
                "bottleneck_channels": 16,
                "dropout": 0.1,
                "pool": "last",
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "channels": "Hidden channels",
                "num_blocks": "Number of inception blocks",
                "kernel_sizes": "Comma-separated kernel sizes (e.g. 3,5,7)",
                "bottleneck_channels": "Bottleneck (1x1) conv channels",
                "dropout": _DROPOUT_HELP,
                "pool": _POOLING_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-gmlp-direct": model_spec(
            key="torch-gmlp-direct",
            description="Torch gMLP-style model (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_gmlp_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "num_layers": 4,
                "ffn_dim": 128,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": _EMBEDDING_DIM_HELP,
                "num_layers": "Number of gMLP layers",
                "ffn_dim": "gMLP feed-forward dimension",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-nhits-direct": model_spec(
            key="torch-nhits-direct",
            description="Torch N-HiTS-style multi-rate residual MLP (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_nhits_direct,
            default_params={
                "lags": 192,
                "pool_sizes": (1, 2, 4),
                "num_blocks": 6,
                "num_layers": 2,
                "layer_width": 128,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "pool_sizes": "Comma-separated pooling sizes (e.g. 1,2,4)",
                "num_blocks": _RESIDUAL_BLOCKS_HELP,
                "num_layers": "Hidden layers per block",
                "layer_width": "Hidden width per layer",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-tide-direct": model_spec(
            key="torch-tide-direct",
            description="Torch TiDE-style encoder/decoder MLP (direct multi-horizon). Requires PyTorch.",
            factory=_factory_torch_tide_direct,
            default_params={
                "lags": 96,
                "d_model": 64,
                "hidden_size": 128,
                "dropout": 0.1,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "d_model": "Context embedding dimension",
                "hidden_size": "Hidden size for encoder/decoder MLP",
                "dropout": _DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
            },
            requires=("torch",),
        ),
        "torch-deepar-recursive": model_spec(
            key="torch-deepar-recursive",
            description="Torch DeepAR-style Gaussian RNN (one-step trained, recursive forecast). Requires PyTorch.",
            factory=_factory_torch_deepar_recursive,
            default_params={
                "lags": 48,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "hidden_size": _RNN_HIDDEN_SIZE_HELP,
                "num_layers": _GRU_LAYERS_HELP,
                "dropout": _RNN_DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
                "loss": "Ignored (DeepAR uses Gaussian NLL)",
            },
            requires=("torch",),
        ),
        "torch-qrnn-recursive": model_spec(
            key="torch-qrnn-recursive",
            description="Torch quantile-regression RNN (one-step trained, recursive forecast). Requires PyTorch.",
            factory=_factory_torch_qrnn_recursive,
            default_params={
                "lags": 48,
                "q": 0.5,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                **_TORCH_COMMON_DEFAULTS,
            },
            param_help={
                "lags": _LAG_WINDOW_HELP,
                "q": "Quantile in (0,1) for pinball loss",
                "hidden_size": _RNN_HIDDEN_SIZE_HELP,
                "num_layers": _GRU_LAYERS_HELP,
                "dropout": _RNN_DROPOUT_HELP,
                **_TORCH_COMMON_PARAM_HELP,
                "loss": "Ignored (QRNN uses pinball loss at quantile q)",
            },
            requires=("torch",),
        ),
    }


def _register_local_xformer_specs(add_local_xformer: Any) -> None:
    # Local xFormer variants: (attn) x (norm) x (ffn)
    for attn_s, attn_label in [
        ("full", "full"),
        ("local", "local-window"),
        ("logsparse", "log-sparse"),
        ("longformer", "longformer-windowed+global"),
        ("bigbird", "bigbird-random+local+global"),
        ("performer", "performer"),
        ("linformer", "linformer"),
        ("nystrom", "nystrom"),
        ("probsparse", "prob-sparse"),
        ("autocorr", "auto-correlation"),
        ("reformer", "reformer-lsh"),
    ]:
        for norm_s, norm_label in [("layer", "LayerNorm"), ("rms", "RMSNorm")]:
            for ffn_s, ffn_label in [("gelu", "GELU"), ("swiglu", "SwiGLU")]:
                norm_short = "ln" if norm_s == "layer" else "rms"
                key = f"torch-xformer-{attn_s}-{norm_short}-{ffn_s}-direct"
                add_local_xformer(
                    key,
                    f"Torch xFormer ({attn_label} attention) with {norm_label}+{ffn_label} "
                    "(direct multi-horizon). Requires PyTorch.",
                    attn=attn_s,
                    norm=norm_s,
                    ffn=ffn_s,
                )

    # 41-44: RoPE positional variants (LN+GELU)
    for attn_s in ["full", "performer", "linformer", "nystrom"]:
        add_local_xformer(
            f"torch-xformer-{attn_s}-rope-ln-gelu-direct",
            f"Torch xFormer ({attn_s} attention) with RoPE positional encoding (LN+GELU). "
            "Requires PyTorch.",
            attn=attn_s,
            pos_emb="rope",
            norm="layer",
            ffn="gelu",
        )

    # 45-48: sincos pos variants (LN+GELU)
    for attn_s in ["full", "performer", "linformer", "nystrom"]:
        add_local_xformer(
            f"torch-xformer-{attn_s}-sincos-ln-gelu-direct",
            f"Torch xFormer ({attn_s} attention) with sinusoidal positional encoding (LN+GELU). "
            "Requires PyTorch.",
            attn=attn_s,
            pos_emb="sincos",
            norm="layer",
            ffn="gelu",
        )

    # 49-52: Time2Vec pos variants (LN+GELU)
    for attn_s in ["full", "performer", "linformer", "nystrom"]:
        add_local_xformer(
            f"torch-xformer-{attn_s}-time2vec-ln-gelu-direct",
            f"Torch xFormer ({attn_s} attention) with Time2Vec positional features (LN+GELU). "
            "Requires PyTorch.",
            attn=attn_s,
            pos_emb="time2vec",
            norm="layer",
            ffn="gelu",
        )

    # 53-56: RevIN variants
    for attn_s in ["full", "performer", "linformer", "nystrom"]:
        add_local_xformer(
            f"torch-xformer-{attn_s}-revin-direct",
            f"Torch xFormer ({attn_s} attention) with RevIN (direct multi-horizon). "
            "Requires PyTorch.",
            attn=attn_s,
            revin=True,
        )

    # 57-60: deeper/wider configs
    add_local_xformer(
        "torch-xformer-full-deep-direct",
        "Torch xFormer (full attention) deeper config (4 layers). Requires PyTorch.",
        attn="full",
        num_layers=4,
    )
    add_local_xformer(
        "torch-xformer-performer-deep-direct",
        "Torch xFormer (performer attention) deeper config (4 layers). Requires PyTorch.",
        attn="performer",
        num_layers=4,
    )
    add_local_xformer(
        "torch-xformer-full-wide-direct",
        "Torch xFormer (full attention) wider config (d_model=128). Requires PyTorch.",
        attn="full",
        d_model=128,
        nhead=8,
        dim_feedforward=512,
    )
    add_local_xformer(
        "torch-xformer-performer-wide-direct",
        "Torch xFormer (performer attention) wider config (d_model=128). Requires PyTorch.",
        attn="performer",
        d_model=128,
        nhead=8,
        dim_feedforward=512,
    )


def _register_global_xformer_specs(add_global_xformer: Any) -> None:
    # 61-65: baseline attention variants
    for attn_s in [
        "full",
        "local",
        "logsparse",
        "longformer",
        "bigbird",
        "performer",
        "linformer",
        "nystrom",
        "probsparse",
        "autocorr",
        "reformer",
    ]:
        add_global_xformer(
            f"torch-xformer-{attn_s}-global",
            f"Torch global xFormer ({attn_s} attention) baseline. Requires PyTorch.",
            attn=attn_s,
        )

    # 66-70: RMSNorm variants
    for attn_s in [
        "full",
        "local",
        "logsparse",
        "longformer",
        "bigbird",
        "performer",
        "linformer",
        "nystrom",
        "probsparse",
        "autocorr",
        "reformer",
    ]:
        add_global_xformer(
            f"torch-xformer-{attn_s}-rms-global",
            f"Torch global xFormer ({attn_s} attention) with RMSNorm. Requires PyTorch.",
            attn=attn_s,
            norm="rms",
        )

    # 71-74: SwiGLU variants (subset)
    for attn_s in ["full", "performer", "linformer", "nystrom"]:
        add_global_xformer(
            f"torch-xformer-{attn_s}-swiglu-global",
            f"Torch global xFormer ({attn_s} attention) with SwiGLU FFN. Requires PyTorch.",
            attn=attn_s,
            ffn="swiglu",
        )

    # 75-78: positional variants
    add_global_xformer(
        "torch-xformer-full-rope-global",
        "Torch global xFormer (full attention) with RoPE positional encoding. Requires PyTorch.",
        attn="full",
        pos_emb="rope",
    )
    add_global_xformer(
        "torch-xformer-performer-rope-global",
        "Torch global xFormer (performer attention) with RoPE positional encoding. Requires PyTorch.",
        attn="performer",
        pos_emb="rope",
    )
    add_global_xformer(
        "torch-xformer-full-sincos-global",
        "Torch global xFormer (full attention) with sinusoidal positional encoding. Requires PyTorch.",
        attn="full",
        pos_emb="sincos",
    )
    add_global_xformer(
        "torch-xformer-full-time2vec-global",
        "Torch global xFormer (full attention) with Time2Vec positional features. Requires PyTorch.",
        attn="full",
        pos_emb="time2vec",
    )

    # 79-80: deeper/wider configs
    add_global_xformer(
        "torch-xformer-full-deep-global",
        "Torch global xFormer (full attention) deeper config (4 layers). Requires PyTorch.",
        attn="full",
        num_layers=4,
    )
    add_global_xformer(
        "torch-xformer-full-wide-global",
        "Torch global xFormer (full attention) wider config (d_model=128). Requires PyTorch.",
        attn="full",
        d_model=128,
        nhead=8,
        dim_feedforward=512,
    )


def _make_torch_dl_variant_specs(context: Any) -> dict[str, ModelSpec]:
    model_spec = context.ModelSpec
    _TORCH_COMMON_DEFAULTS = context._TORCH_COMMON_DEFAULTS
    _TORCH_COMMON_PARAM_HELP = context._TORCH_COMMON_PARAM_HELP
    _factory_torch_lstnet_direct = context._factory_torch_lstnet_direct
    _factory_torch_seq2seq_direct = context._factory_torch_seq2seq_direct
    _factory_torch_xformer_direct = context._factory_torch_xformer_direct
    torch_rnn_global_forecaster = context.torch_rnn_global_forecaster
    torch_xformer_global_forecaster = context.torch_xformer_global_forecaster
    extra: dict[str, ModelSpec] = {}

    xformer_help = {
        "lags": _LAG_WINDOW_HELP,
        "d_model": "Transformer model dimension",
        "nhead": "Attention heads",
        "num_layers": _ENCODER_LAYERS_HELP,
        "dim_feedforward": "FFN hidden dimension",
        "dropout": _DROPOUT_HELP,
        "attn": "Attention type: full, local, logsparse, longformer, bigbird, performer, linformer, nystrom, probsparse, autocorr, reformer",
        "pos_emb": "Positional embedding: learned, sincos, rope, time2vec, none",
        "norm": "Normalization: layer, rms",
        "ffn": "FFN: gelu, swiglu",
        "local_window": "Local attention window radius (attn=local/logsparse/longformer/bigbird)",
        "bigbird_random_k": "Random key connections per token (attn=bigbird)",
        "performer_features": "Performer random feature count (attn=performer)",
        "linformer_k": "Linformer projection length (attn=linformer)",
        "nystrom_landmarks": "Nystrom landmarks (attn=nystrom)",
        "reformer_bucket_size": "Reformer LSH bucket size (attn=reformer)",
        "reformer_n_hashes": "Reformer LSH hash rounds (attn=reformer)",
        "probsparse_top_u": "Top-u queries for ProbSparse attention (attn=probsparse)",
        "autocorr_top_k": "Top-k delays for AutoCorrelation attention (attn=autocorr)",
        "horizon_tokens": "Future token placeholders: zeros, learned",
        "revin": "RevIN per-window normalization (true/false)",
        "residual_gating": "Residual gating (true/false)",
        "drop_path": "Stochastic depth drop probability in [0,1)",
        **_TORCH_COMMON_PARAM_HELP,
    }

    xformer_base_defaults = {
        "lags": 96,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "attn": "full",
        "pos_emb": "learned",
        "norm": "layer",
        "ffn": "gelu",
        "local_window": 16,
        "bigbird_random_k": 8,
        "performer_features": 64,
        "linformer_k": 32,
        "nystrom_landmarks": 16,
        "reformer_bucket_size": 8,
        "reformer_n_hashes": 1,
        "probsparse_top_u": 32,
        "autocorr_top_k": 4,
        "horizon_tokens": "zeros",
        "revin": False,
        "residual_gating": False,
        "drop_path": 0.0,
        **_TORCH_COMMON_DEFAULTS,
    }

    def _add_local_xformer(
        key: str,
        description: str,
        **overrides: Any,
    ) -> None:
        extra[key] = model_spec(
            key=key,
            description=description,
            factory=_factory_torch_xformer_direct,
            default_params={**xformer_base_defaults, **overrides},
            param_help=dict(xformer_help),
            requires=("torch",),
        )

    _register_local_xformer_specs(_add_local_xformer)

    # ---- Local RNN family (Seq2Seq + LSTNet) ----
    seq2seq_help = {
        "lags": "Lag window length (encoder length)",
        "cell": "RNN cell: lstm, gru",
        "attention": "Attention: none, bahdanau",
        "hidden_size": _RNN_HIDDEN_SIZE_HELP,
        "num_layers": _STACKED_RNN_LAYERS_HELP,
        "dropout": _RNN_DROPOUT_HELP,
        "teacher_forcing": "Teacher forcing ratio at the start of training",
        "teacher_forcing_final": "Teacher forcing ratio at the end of training (None keeps it constant)",
        **_TORCH_COMMON_PARAM_HELP,
    }
    seq2seq_base_defaults = {
        "lags": 48,
        "cell": "lstm",
        "attention": "none",
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "teacher_forcing": 0.5,
        "teacher_forcing_final": None,
        **_TORCH_COMMON_DEFAULTS,
        "val_split": 0.1,
    }

    def _add_local_seq2seq(key: str, description: str, **overrides: Any) -> None:
        extra[key] = model_spec(
            key=key,
            description=description,
            factory=_factory_torch_seq2seq_direct,
            default_params={**seq2seq_base_defaults, **overrides},
            param_help=dict(seq2seq_help),
            requires=("torch",),
        )

    _add_local_seq2seq(
        "torch-seq2seq-lstm-direct",
        "Torch Seq2Seq LSTM (encoder-decoder) direct multi-horizon. Requires PyTorch.",
        cell="lstm",
        attention="none",
    )
    _add_local_seq2seq(
        "torch-seq2seq-gru-direct",
        "Torch Seq2Seq GRU (encoder-decoder) direct multi-horizon. Requires PyTorch.",
        cell="gru",
        attention="none",
    )
    _add_local_seq2seq(
        "torch-seq2seq-attn-lstm-direct",
        "Torch Seq2Seq LSTM with Bahdanau attention (direct multi-horizon). Requires PyTorch.",
        cell="lstm",
        attention="bahdanau",
    )
    _add_local_seq2seq(
        "torch-seq2seq-attn-gru-direct",
        "Torch Seq2Seq GRU with Bahdanau attention (direct multi-horizon). Requires PyTorch.",
        cell="gru",
        attention="bahdanau",
    )
    _add_local_seq2seq(
        "torch-seq2seq-lstm-deep-direct",
        "Torch Seq2Seq LSTM deeper config (2 layers). Requires PyTorch.",
        cell="lstm",
        attention="none",
        num_layers=2,
        dropout=0.1,
    )
    _add_local_seq2seq(
        "torch-seq2seq-gru-deep-direct",
        "Torch Seq2Seq GRU deeper config (2 layers). Requires PyTorch.",
        cell="gru",
        attention="none",
        num_layers=2,
        dropout=0.1,
    )
    _add_local_seq2seq(
        "torch-seq2seq-lstm-wide-direct",
        "Torch Seq2Seq LSTM wider config (hidden_size=128). Requires PyTorch.",
        cell="lstm",
        attention="none",
        hidden_size=128,
    )
    _add_local_seq2seq(
        "torch-seq2seq-gru-wide-direct",
        "Torch Seq2Seq GRU wider config (hidden_size=128). Requires PyTorch.",
        cell="gru",
        attention="none",
        hidden_size=128,
    )

    lstnet_help = {
        "lags": _LAG_WINDOW_HELP,
        "cnn_channels": "CNN output channels",
        "kernel_size": "CNN kernel size",
        "rnn_hidden": _GRU_HIDDEN_HELP,
        "skip": "Skip period (0 disables)",
        "highway_window": "Highway window length (0 disables)",
        "dropout": _DROPOUT_HELP,
        **_TORCH_COMMON_PARAM_HELP,
    }
    extra["torch-lstnet-direct"] = model_spec(
        key="torch-lstnet-direct",
        description="Torch LSTNet-style CNN+GRU(+skip)+highway (lite) direct multi-horizon. Requires PyTorch.",
        factory=_factory_torch_lstnet_direct,
        default_params={
            "lags": 96,
            "cnn_channels": 16,
            "kernel_size": 6,
            "rnn_hidden": 32,
            "skip": 24,
            "highway_window": 24,
            "dropout": 0.2,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help=dict(lstnet_help),
        requires=("torch",),
    )

    # ---- Global Transformer-family variants ----
    xformer_global_help = {
        "context_length": "Context window length (encoder length)",
        "x_cols": "Optional covariate columns from long_df (comma-separated)",
        "add_time_features": "Add built-in time features from ds (true/false)",
        "normalize": "Z-score normalize per-series inside each cutoff window (true/false)",
        "max_train_size": "Optional per-series rolling training window length (None for expanding)",
        "sample_step": "Stride when generating training windows (>=1)",
        "d_model": "Transformer model dimension",
        "nhead": "Attention heads",
        "num_layers": _ENCODER_LAYERS_HELP,
        "dim_feedforward": "FFN hidden dimension",
        "id_emb_dim": "Series-id embedding dim (panel/global models)",
        "dropout": _DROPOUT_HELP,
        "attn": "Attention type: full, local, logsparse, longformer, bigbird, performer, linformer, nystrom, probsparse, autocorr, reformer",
        "pos_emb": "Positional embedding: learned, sincos, rope, time2vec, none",
        "norm": "Normalization: layer, rms",
        "ffn": "FFN: gelu, swiglu",
        "local_window": "Local attention window radius (attn=local/logsparse/longformer/bigbird)",
        "bigbird_random_k": "Random key connections per token (attn=bigbird)",
        "performer_features": "Performer random feature count (attn=performer)",
        "linformer_k": "Linformer projection length (attn=linformer)",
        "nystrom_landmarks": "Nystrom landmarks (attn=nystrom)",
        "reformer_bucket_size": "Reformer LSH bucket size (attn=reformer)",
        "reformer_n_hashes": "Reformer LSH hash rounds (attn=reformer)",
        "probsparse_top_u": "Top-u queries for ProbSparse attention (attn=probsparse)",
        "autocorr_top_k": "Top-k delays for AutoCorrelation attention (attn=autocorr)",
        "residual_gating": "Residual gating (true/false)",
        "drop_path": "Stochastic depth drop probability in [0,1)",
        "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
        **_TORCH_COMMON_PARAM_HELP,
    }
    xformer_global_base_defaults = {
        "context_length": 96,
        "x_cols": (),
        "add_time_features": True,
        "normalize": True,
        "max_train_size": None,
        "sample_step": 1,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "id_emb_dim": 8,
        "dropout": 0.1,
        "attn": "full",
        "pos_emb": "learned",
        "norm": "layer",
        "ffn": "gelu",
        "local_window": 16,
        "bigbird_random_k": 8,
        "performer_features": 64,
        "linformer_k": 32,
        "nystrom_landmarks": 16,
        "reformer_bucket_size": 8,
        "reformer_n_hashes": 1,
        "probsparse_top_u": 32,
        "autocorr_top_k": 4,
        "residual_gating": False,
        "drop_path": 0.0,
        "quantiles": (),
        **_TORCH_COMMON_DEFAULTS,
        "epochs": 30,
        "batch_size": 64,
        "val_split": 0.1,
    }

    def _add_global_xformer(key: str, description: str, **overrides: Any) -> None:
        extra[key] = model_spec(
            key=key,
            description=description,
            factory=torch_xformer_global_forecaster,
            default_params={**xformer_global_base_defaults, **overrides},
            param_help=dict(xformer_global_help),
            requires=("torch",),
            interface="global",
        )

    _register_global_xformer_specs(_add_global_xformer)

    # ---- Global RNN variants ----
    rnn_global_help = {
        "context_length": "Context window length (encoder length)",
        "x_cols": "Optional covariate columns from long_df (comma-separated)",
        "add_time_features": "Add built-in time features from ds (true/false)",
        "normalize": "Z-score normalize per-series inside each cutoff window (true/false)",
        "max_train_size": "Optional per-series rolling training window length (None for expanding)",
        "sample_step": "Stride when generating training windows (>=1)",
        "cell": "RNN cell: lstm, gru",
        "hidden_size": _RNN_HIDDEN_SIZE_HELP,
        "num_layers": _STACKED_RNN_LAYERS_HELP,
        "dropout": _RNN_DROPOUT_HELP,
        "id_emb_dim": "Series-id embedding dim (panel/global models)",
        "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
        **_TORCH_COMMON_PARAM_HELP,
    }
    rnn_global_base_defaults = {
        "context_length": 96,
        "x_cols": (),
        "add_time_features": True,
        "normalize": True,
        "max_train_size": None,
        "sample_step": 1,
        "cell": "lstm",
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "id_emb_dim": 8,
        "quantiles": (),
        **_TORCH_COMMON_DEFAULTS,
        "epochs": 30,
        "batch_size": 64,
        "val_split": 0.1,
    }

    def _add_global_rnn(key: str, description: str, **overrides: Any) -> None:
        extra[key] = model_spec(
            key=key,
            description=description,
            factory=torch_rnn_global_forecaster,
            default_params={**rnn_global_base_defaults, **overrides},
            param_help=dict(rnn_global_help),
            requires=("torch",),
            interface="global",
        )

    _add_global_rnn(
        "torch-rnn-lstm-global",
        "Torch global RNN backbone (LSTM) with token-wise horizon head. Requires PyTorch.",
        cell="lstm",
    )
    _add_global_rnn(
        "torch-rnn-gru-global",
        "Torch global RNN backbone (GRU) with token-wise horizon head. Requires PyTorch.",
        cell="gru",
    )
    _add_global_rnn(
        "torch-rnn-encoder-global",
        "Torch global encoder-only RNN horizon head (seq2seq-lite). Requires PyTorch.",
        cell="lstm",
        hidden_size=32,
    )

    return extra


def _make_torch_rnnpaper_specs(context: Any) -> dict[str, ModelSpec]:
    model_spec = context.ModelSpec
    _TORCH_COMMON_DEFAULTS = context._TORCH_COMMON_DEFAULTS
    _TORCH_COMMON_PARAM_HELP = context._TORCH_COMMON_PARAM_HELP
    _factory_torch_rnnpaper_direct = context._factory_torch_rnnpaper_direct
    list_rnnpaper_specs = context.list_rnnpaper_specs
    extra: dict[str, ModelSpec] = {}

    help_map = {
        "paper": "Paper-named RNN architecture (fixed per key)",
        "lags": _LAG_WINDOW_HELP,
        "hidden_size": "Hidden size",
        "num_layers": "Stacked layers (only for some built-in torch RNN bases)",
        "dropout": "Dropout probability in [0,1) (only if num_layers>1 for torch bases)",
        "attn_hidden": "Attention MLP hidden size (for attention/memory variants)",
        "kernel_size": "Conv1d kernel size (for QRNN / Conv* variants)",
        "hops": "Attention hops (memory networks)",
        "memory_slots": "External memory slots (NTM/DNC variants)",
        "memory_dim": "External memory slot size (NTM/DNC variants)",
        "spectral_radius": "Reservoir spectral radius (ESN/LSM/Conceptor variants)",
        "leak": "Reservoir leaking rate in (0,1] (ESN/LSM/Conceptor variants)",
        **_TORCH_COMMON_PARAM_HELP,
    }

    base_defaults = {
        "lags": 24,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "attn_hidden": 32,
        "kernel_size": 3,
        "hops": 2,
        "memory_slots": 16,
        "memory_dim": 32,
        "spectral_radius": 0.9,
        "leak": 1.0,
        **_TORCH_COMMON_DEFAULTS,
    }

    for spec in list_rnnpaper_specs():
        extra[spec.key] = model_spec(
            key=spec.key,
            description=spec.description + ". Requires PyTorch.",
            factory=_factory_torch_rnnpaper_direct,
            default_params={**base_defaults, "paper": spec.paper_id},
            param_help=dict(help_map),
            requires=("torch",),
            interface="local",
        )

    return extra


def _make_torch_rnnzoo_specs(context: Any) -> dict[str, ModelSpec]:
    model_spec = context.ModelSpec
    _TORCH_COMMON_DEFAULTS = context._TORCH_COMMON_DEFAULTS
    _TORCH_COMMON_PARAM_HELP = context._TORCH_COMMON_PARAM_HELP
    _factory_torch_rnnzoo_direct = context._factory_torch_rnnzoo_direct
    list_rnnzoo_specs = context.list_rnnzoo_specs
    extra: dict[str, ModelSpec] = {}

    help_map = {
        "base": "Base RNN architecture (paper-named; fixed per key)",
        "variant": "Architecture wrapper: direct, bidir, ln, attn, proj (fixed per key)",
        "lags": _LAG_WINDOW_HELP,
        "hidden_size": "Hidden size",
        "num_layers": "Number of stacked layers (only for torch RNN/LSTM/GRU bases)",
        "dropout": "Dropout probability in [0,1) (only if num_layers>1 for torch bases)",
        "proj_size": "Projection size for variant=proj",
        "attn_hidden": "Attention MLP hidden size for variant=attn",
        "clock_periods": "Clockwork periods as tuple or comma-separated string (clockwork base)",
        "qrnn_kernel_size": "QRNN Conv1d kernel size (qrnn base)",
        "rhn_depth": "RHN transition depth (rhn base)",
        "phased_tau": "Phased LSTM time-gate period (phased-lstm base)",
        "phased_r_on": "Phased LSTM open ratio in (0,1) (phased-lstm base)",
        "phased_leak": "Phased LSTM closed-phase leak in [0,1) (phased-lstm base)",
        **_TORCH_COMMON_PARAM_HELP,
    }

    base_defaults = {
        "lags": 24,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "proj_size": 16,
        "attn_hidden": 32,
        "clock_periods": (1, 2, 4, 8),
        "qrnn_kernel_size": 3,
        "rhn_depth": 2,
        "phased_tau": 32.0,
        "phased_r_on": 0.05,
        "phased_leak": 0.001,
        **_TORCH_COMMON_DEFAULTS,
    }

    for spec in list_rnnzoo_specs():
        extra[spec.key] = model_spec(
            key=spec.key,
            description=spec.description + ". Requires PyTorch.",
            factory=_factory_torch_rnnzoo_direct,
            default_params={**base_defaults, "base": spec.base, "variant": str(spec.variant)},
            param_help=dict(help_map),
            requires=("torch",),
            interface="local",
        )

    return extra


def _make_wave1_reservoir_specs(context: Any) -> dict[str, ModelSpec]:
    """Lane 01 ownership: ESN / reservoir / liquid-state lite families."""
    model_spec = context.ModelSpec
    _TORCH_COMMON_DEFAULTS = context._TORCH_COMMON_DEFAULTS
    _TORCH_COMMON_PARAM_HELP = context._TORCH_COMMON_PARAM_HELP
    np = context.np

    extra: dict[str, ModelSpec] = {}

    help_map = {
        "variant": "Reservoir family variant: esn, deep-esn, or liquid-state",
        "lags": _LAG_WINDOW_HELP,
        "hidden_size": "Reservoir hidden width",
        "spectral_radius": "Reservoir spectral radius (>0)",
        "leak": "Reservoir leaking rate in (0,1]",
        **_TORCH_COMMON_PARAM_HELP,
    }

    base_defaults = {
        "lags": 24,
        "hidden_size": 32,
        "spectral_radius": 0.9,
        "leak": 1.0,
        **_TORCH_COMMON_DEFAULTS,
    }

    descriptions = {
        "torch-esn-direct": (
            "Echo State Network / ESN lite local forecaster. Wraps the existing rnnpaper "
            "reservoir core behind a dedicated first-class model key. Requires PyTorch."
        ),
        "torch-deep-esn-direct": (
            "Deep ESN lite local forecaster. Wraps the existing rnnpaper deep reservoir core "
            "behind a dedicated first-class model key. Requires PyTorch."
        ),
        "torch-liquid-state-direct": (
            "Liquid State Machine lite local forecaster. Wraps the existing rnnpaper liquid-state "
            "core behind a dedicated first-class model key. Requires PyTorch."
        ),
    }
    variants = {
        "torch-esn-direct": "esn",
        "torch-deep-esn-direct": "deep-esn",
        "torch-liquid-state-direct": "liquid-state",
    }

    def _factory(
        *,
        variant: str,
        lags: int = 24,
        hidden_size: int = 32,
        spectral_radius: float = 0.9,
        leak: float = 1.0,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        seed: int = 0,
        normalize: bool = True,
        device: str = "cpu",
        patience: int = 10,
        loss: str = "mse",
        val_split: float = 0.0,
        grad_clip_norm: float = 0.0,
        optimizer: str = "adam",
        momentum: float = 0.9,
        scheduler: str = "none",
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1,
        restore_best: bool = True,
        **params: Any,
    ) -> ForecasterFn:
        variant_s = str(variant)
        lags_int = int(lags)
        hidden_size_int = int(hidden_size)
        spectral_radius_f = float(spectral_radius)
        leak_f = float(leak)
        epochs_int = int(epochs)
        lr_f = float(lr)
        weight_decay_f = float(weight_decay)
        batch_size_int = int(batch_size)
        seed_int = int(seed)
        normalize_bool = bool(normalize)
        device_s = str(device)
        patience_int = int(patience)
        loss_s = str(loss)
        val_split_f = float(val_split)
        grad_clip_norm_f = float(grad_clip_norm)
        optimizer_s = str(optimizer)
        momentum_f = float(momentum)
        scheduler_s = str(scheduler)
        scheduler_step_size_int = int(scheduler_step_size)
        scheduler_gamma_f = float(scheduler_gamma)
        restore_best_bool = bool(restore_best)
        extra_params = dict(params)

        def _f(train: Any, horizon: int) -> np.ndarray:
            from ..torch_reservoir import torch_reservoir_direct_forecast

            return torch_reservoir_direct_forecast(
                train,
                horizon,
                variant=variant_s,
                lags=lags_int,
                hidden_size=hidden_size_int,
                spectral_radius=spectral_radius_f,
                leak=leak_f,
                epochs=epochs_int,
                lr=lr_f,
                weight_decay=weight_decay_f,
                batch_size=batch_size_int,
                seed=seed_int,
                normalize=normalize_bool,
                device=device_s,
                patience=patience_int,
                loss=loss_s,
                val_split=val_split_f,
                grad_clip_norm=grad_clip_norm_f,
                optimizer=optimizer_s,
                momentum=momentum_f,
                scheduler=scheduler_s,
                scheduler_step_size=scheduler_step_size_int,
                scheduler_gamma=scheduler_gamma_f,
                restore_best=restore_best_bool,
                **extra_params,
            )

        return _f

    for key, description in descriptions.items():
        extra[key] = model_spec(
            key=key,
            description=description,
            factory=_factory,
            default_params={**base_defaults, "variant": variants[key]},
            param_help=dict(help_map),
            requires=("torch",),
            interface="local",
        )

    return extra


def _make_wave1_structured_rnn_specs(context: Any) -> dict[str, ModelSpec]:
    """Lane 02 ownership: structured / grid recurrent lite families."""
    model_spec = context.ModelSpec
    _TORCH_COMMON_DEFAULTS = context._TORCH_COMMON_DEFAULTS
    _TORCH_COMMON_PARAM_HELP = context._TORCH_COMMON_PARAM_HELP
    np = context.np

    extra: dict[str, ModelSpec] = {}

    help_map = {
        "variant": "Structured recurrent family variant: multidim-rnn, grid-lstm, or structural-rnn",
        "lags": _LAG_WINDOW_HELP,
        "hidden_size": "Hidden size for the wrapped structured recurrent core",
        **_TORCH_COMMON_PARAM_HELP,
    }

    base_defaults = {
        "lags": 24,
        "hidden_size": 32,
        **_TORCH_COMMON_DEFAULTS,
    }

    descriptions = {
        "torch-multidim-rnn-direct": (
            "Multi-Dimensional RNN lite local forecaster. Wraps the existing rnnpaper "
            "structured recurrent core behind a dedicated first-class model key. Requires PyTorch."
        ),
        "torch-grid-lstm-direct": (
            "Grid LSTM lite local forecaster. Wraps the existing rnnpaper grid-structured "
            "core behind a dedicated first-class model key. Requires PyTorch."
        ),
        "torch-structural-rnn-direct": (
            "Structural-RNN lite local forecaster. Wraps the existing rnnpaper structured "
            "spatiotemporal core behind a dedicated first-class model key. Requires PyTorch."
        ),
    }
    variants = {
        "torch-multidim-rnn-direct": "multidim-rnn",
        "torch-grid-lstm-direct": "grid-lstm",
        "torch-structural-rnn-direct": "structural-rnn",
    }

    def _factory(
        *,
        variant: str,
        lags: int = 24,
        hidden_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        seed: int = 0,
        normalize: bool = True,
        device: str = "cpu",
        patience: int = 10,
        loss: str = "mse",
        val_split: float = 0.0,
        grad_clip_norm: float = 0.0,
        optimizer: str = "adam",
        momentum: float = 0.9,
        scheduler: str = "none",
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1,
        restore_best: bool = True,
        **params: Any,
    ) -> ForecasterFn:
        variant_s = str(variant)
        lags_int = int(lags)
        hidden_size_int = int(hidden_size)
        epochs_int = int(epochs)
        lr_f = float(lr)
        weight_decay_f = float(weight_decay)
        batch_size_int = int(batch_size)
        seed_int = int(seed)
        normalize_bool = bool(normalize)
        device_s = str(device)
        patience_int = int(patience)
        loss_s = str(loss)
        val_split_f = float(val_split)
        grad_clip_norm_f = float(grad_clip_norm)
        optimizer_s = str(optimizer)
        momentum_f = float(momentum)
        scheduler_s = str(scheduler)
        scheduler_step_size_int = int(scheduler_step_size)
        scheduler_gamma_f = float(scheduler_gamma)
        restore_best_bool = bool(restore_best)
        extra_params = dict(params)

        def _f(train: Any, horizon: int) -> np.ndarray:
            from ..torch_structured_rnn import torch_structured_rnn_direct_forecast

            return torch_structured_rnn_direct_forecast(
                train,
                horizon,
                variant=variant_s,
                lags=lags_int,
                hidden_size=hidden_size_int,
                epochs=epochs_int,
                lr=lr_f,
                weight_decay=weight_decay_f,
                batch_size=batch_size_int,
                seed=seed_int,
                normalize=normalize_bool,
                device=device_s,
                patience=patience_int,
                loss=loss_s,
                val_split=val_split_f,
                grad_clip_norm=grad_clip_norm_f,
                optimizer=optimizer_s,
                momentum=momentum_f,
                scheduler=scheduler_s,
                scheduler_step_size=scheduler_step_size_int,
                scheduler_gamma=scheduler_gamma_f,
                restore_best=restore_best_bool,
                **extra_params,
            )

        return _f

    for key, description in descriptions.items():
        extra[key] = model_spec(
            key=key,
            description=description,
            factory=_factory,
            default_params={**base_defaults, "variant": variants[key]},
            param_help=dict(help_map),
            requires=("torch",),
            interface="local",
        )

    return extra


def _make_wave1_probabilistic_specs(context: Any) -> dict[str, ModelSpec]:
    """Lane 06 ownership: TimeGrad / TACTiS style probabilistic lite families."""
    model_spec = context.ModelSpec
    _TORCH_COMMON_DEFAULTS = context._TORCH_COMMON_DEFAULTS
    _TORCH_COMMON_PARAM_HELP = context._TORCH_COMMON_PARAM_HELP
    np = context.np

    extra: dict[str, ModelSpec] = {}

    help_map = {
        "variant": "Probabilistic lite family variant: timegrad or tactis",
        "lags": _LAG_WINDOW_HELP,
        "hidden_size": "Hidden width for the probabilistic backbone",
        "num_layers": "Recurrent depth for timegrad-lite",
        "num_heads": "Attention heads for tactis-lite (must divide hidden_size)",
        "dropout": "Dropout in [0,1)",
        **_TORCH_COMMON_PARAM_HELP,
        "loss": "Training loss: gaussian, mse, mae, huber",
    }

    base_defaults = {
        "lags": 24,
        "hidden_size": 32,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.0,
        **_TORCH_COMMON_DEFAULTS,
        "loss": "gaussian",
    }

    descriptions = {
        "torch-timegrad-direct": (
            "TimeGrad-style lite probabilistic local forecaster. Uses a compact recurrent "
            "Gaussian head and returns predictive means only; it does not implement full "
            "diffusion sampling semantics. Requires PyTorch."
        ),
        "torch-tactis-direct": (
            "TACTiS-style lite probabilistic local forecaster. Uses horizon-query attention "
            "with a Gaussian head and returns predictive means only; it does not implement "
            "full copula or trajectory sampling semantics. Requires PyTorch."
        ),
    }
    variants = {
        "torch-timegrad-direct": "timegrad",
        "torch-tactis-direct": "tactis",
    }

    def _factory(
        *,
        variant: str,
        lags: int = 24,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        seed: int = 0,
        normalize: bool = True,
        device: str = "cpu",
        patience: int = 10,
        loss: str = "gaussian",
        val_split: float = 0.0,
        grad_clip_norm: float = 0.0,
        optimizer: str = "adam",
        momentum: float = 0.9,
        scheduler: str = "none",
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1,
        restore_best: bool = True,
    ) -> ForecasterFn:
        variant_s = str(variant)
        lags_int = int(lags)
        hidden_size_int = int(hidden_size)
        num_layers_int = int(num_layers)
        num_heads_int = int(num_heads)
        dropout_f = float(dropout)
        epochs_int = int(epochs)
        lr_f = float(lr)
        weight_decay_f = float(weight_decay)
        batch_size_int = int(batch_size)
        seed_int = int(seed)
        normalize_bool = bool(normalize)
        device_s = str(device)
        patience_int = int(patience)
        loss_s = str(loss)
        val_split_f = float(val_split)
        grad_clip_norm_f = float(grad_clip_norm)
        optimizer_s = str(optimizer)
        momentum_f = float(momentum)
        scheduler_s = str(scheduler)
        scheduler_step_size_int = int(scheduler_step_size)
        scheduler_gamma_f = float(scheduler_gamma)
        restore_best_bool = bool(restore_best)

        def _f(train: Any, horizon: int) -> np.ndarray:
            from ..torch_probabilistic import torch_probabilistic_direct_forecast

            return torch_probabilistic_direct_forecast(
                train,
                horizon,
                variant=variant_s,
                lags=lags_int,
                hidden_size=hidden_size_int,
                num_layers=num_layers_int,
                num_heads=num_heads_int,
                dropout=dropout_f,
                epochs=epochs_int,
                lr=lr_f,
                weight_decay=weight_decay_f,
                batch_size=batch_size_int,
                seed=seed_int,
                normalize=normalize_bool,
                device=device_s,
                patience=patience_int,
                loss=loss_s,
                val_split=val_split_f,
                grad_clip_norm=grad_clip_norm_f,
                optimizer=optimizer_s,
                momentum=momentum_f,
                scheduler=scheduler_s,
                scheduler_step_size=scheduler_step_size_int,
                scheduler_gamma=scheduler_gamma_f,
                restore_best=restore_best_bool,
            )

        return _f

    for key, description in descriptions.items():
        extra[key] = model_spec(
            key=key,
            description=description,
            factory=_factory,
            default_params={**base_defaults, "variant": variants[key]},
            param_help=dict(help_map),
            requires=("torch",),
            interface="local",
        )

    return extra


_build_torch_local_catalog_base = build_torch_local_catalog


def build_torch_local_catalog(context: Any) -> dict[str, Any]:
    catalog = _build_torch_local_catalog_base(context)
    catalog.update(_make_torch_dl_variant_specs(context))
    catalog.update(_make_torch_rnnpaper_specs(context))
    catalog.update(_make_torch_rnnzoo_specs(context))
    catalog.update(_make_wave1_reservoir_specs(context))
    catalog.update(_make_wave1_structured_rnn_specs(context))
    catalog.update(_make_wave1_probabilistic_specs(context))
    return catalog
