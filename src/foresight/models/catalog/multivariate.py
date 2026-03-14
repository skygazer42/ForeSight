from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..specs import ModelSpec, MultivariateForecasterFn


LAG_WINDOW_PARAM_HELP = "Lag window length"
DROPOUT_PROBABILITY_PARAM_HELP = "Dropout probability in [0,1)"
DROPOUT_RANGE_PARAM_HELP = "Dropout in [0,1)"


def build_multivariate_catalog(context: Any) -> dict[str, Any]:
    model_spec = context.ModelSpec
    _TORCH_COMMON_DEFAULTS = context._TORCH_COMMON_DEFAULTS
    _TORCH_COMMON_PARAM_HELP = context._TORCH_COMMON_PARAM_HELP
    _factory_torch_graphwavenet_multivariate = context._factory_torch_graphwavenet_multivariate
    _factory_torch_stgcn_multivariate = context._factory_torch_stgcn_multivariate
    _factory_torch_stid_multivariate = context._factory_torch_stid_multivariate
    _factory_var = context._factory_var
    return {
    "var": model_spec(
        key="var",
        description="Vector autoregression via statsmodels on a multivariate target matrix. Optional dependency.",
        factory=_factory_var,
        default_params={"maxlags": 1, "trend": "c", "ic": None},
        param_help={
            "maxlags": "Maximum autoregressive lag order",
            "trend": "Deterministic trend: n, c, ct, ctt",
            "ic": "Optional lag-order selection criterion: aic, bic, hqic, fpe, or none",
        },
        requires=("stats",),
        interface="multivariate",
    ),
    "torch-stid-multivariate": model_spec(
        key="torch-stid-multivariate",
        description="Torch STID-style multivariate baseline (lite) on a wide target matrix. Requires PyTorch.",
        factory=_factory_torch_stid_multivariate,
        default_params={
            "lags": 24,
            "d_model": 64,
            "num_blocks": 2,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": LAG_WINDOW_PARAM_HELP,
            "d_model": "Node embedding / hidden dimension",
            "num_blocks": "Number of identity-style mixing blocks",
            "dropout": DROPOUT_PROBABILITY_PARAM_HELP,
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="multivariate",
    ),
    "torch-stgcn-multivariate": model_spec(
        key="torch-stgcn-multivariate",
        description="Torch STGCN-style spatiotemporal baseline (lite) on a wide target matrix. Requires PyTorch.",
        factory=_factory_torch_stgcn_multivariate,
        default_params={
            "lags": 24,
            "d_model": 64,
            "num_blocks": 2,
            "kernel_size": 3,
            "dropout": 0.1,
            "adj": "corr",
            "adj_path": "",
            "adj_top_k": 8,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": LAG_WINDOW_PARAM_HELP,
            "d_model": "Hidden dimension",
            "num_blocks": "Number of spatiotemporal blocks",
            "kernel_size": "Temporal convolution kernel size (>=1)",
            "dropout": DROPOUT_PROBABILITY_PARAM_HELP,
            "adj": "Adjacency spec: identity, ring, fully-connected, corr, or a numeric matrix",
            "adj_path": "Optional adjacency matrix path (.npy or .csv/.txt)",
            "adj_top_k": "If adj=corr, keep top-k neighbors per node (0 disables)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="multivariate",
    ),
    "torch-graphwavenet-multivariate": model_spec(
        key="torch-graphwavenet-multivariate",
        description="Torch Graph WaveNet-style baseline (lite) on a wide target matrix. Requires PyTorch.",
        factory=_factory_torch_graphwavenet_multivariate,
        default_params={
            "lags": 24,
            "d_model": 64,
            "num_blocks": 4,
            "kernel_size": 2,
            "dilation_base": 2,
            "dropout": 0.1,
            "adj": "corr",
            "adj_path": "",
            "adj_top_k": 8,
            "adaptive_adj": True,
            "adj_emb_dim": 8,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": LAG_WINDOW_PARAM_HELP,
            "d_model": "Hidden dimension",
            "num_blocks": "Number of dilated temporal graph blocks",
            "kernel_size": "Temporal convolution kernel size (>=1)",
            "dilation_base": "Dilation growth base (>=1). Each block uses dilation=base**i",
            "dropout": DROPOUT_PROBABILITY_PARAM_HELP,
            "adj": "Adjacency spec: identity, ring, fully-connected, corr, or a numeric matrix",
            "adj_path": "Optional adjacency matrix path (.npy or .csv/.txt)",
            "adj_top_k": "If adj=corr, keep top-k neighbors per node (0 disables)",
            "adaptive_adj": "Learn an additional adaptive adjacency (true/false)",
            "adj_emb_dim": "Adaptive adjacency embedding dimension (>=1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="multivariate",
    ),
    }


def _make_wave1_graph_attention_specs(context: Any) -> dict[str, ModelSpec]:
    """Lane 03 ownership: ASTGCN / GMAN style multivariate graph lite families."""
    model_spec = context.ModelSpec
    _TORCH_COMMON_DEFAULTS = context._TORCH_COMMON_DEFAULTS
    _TORCH_COMMON_PARAM_HELP = context._TORCH_COMMON_PARAM_HELP
    np = context.np

    extra: dict[str, ModelSpec] = {}

    help_map = {
        "variant": "Graph-attention lite family variant: astgcn or gman",
        "lags": LAG_WINDOW_PARAM_HELP,
        "d_model": "Latent width for the attention backbone",
        "num_blocks": "Number of stacked attention blocks",
        "num_heads": "Multi-head attention heads",
        "dropout": DROPOUT_RANGE_PARAM_HELP,
        "adj": "Adjacency source: identity, ring, fully-connected, corr, or matrix",
        "adj_path": "Optional adjacency file path (.npy or .csv/.txt)",
        "adj_top_k": "Top-k neighbors kept for correlation adjacency",
        **_TORCH_COMMON_PARAM_HELP,
    }

    base_defaults = {
        "lags": 24,
        "d_model": 64,
        "num_blocks": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "adj": "corr",
        "adj_path": "",
        "adj_top_k": 8,
        **_TORCH_COMMON_DEFAULTS,
    }

    descriptions = {
        "torch-astgcn-multivariate": (
            "ASTGCN-style lite multivariate forecaster. Uses adjacency-biased temporal/spatial "
            "attention over a wide target matrix; this is a compact proxy, not a paper-complete "
            "ASTGCN reproduction. Requires PyTorch."
        ),
        "torch-gman-multivariate": (
            "GMAN-style lite multivariate forecaster. Uses stacked spatial/temporal attention "
            "with node and horizon embeddings over a wide target matrix; this is a compact proxy, "
            "not a paper-complete GMAN reproduction. Requires PyTorch."
        ),
    }
    variants = {
        "torch-astgcn-multivariate": "astgcn",
        "torch-gman-multivariate": "gman",
    }

    def _factory(
        *,
        variant: str,
        lags: int = 24,
        d_model: int = 64,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        adj: Any = "corr",
        adj_path: str = "",
        adj_top_k: int = 8,
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
        **_params: Any,
    ) -> MultivariateForecasterFn:
        variant_s = str(variant)
        lags_int = int(lags)
        d_model_int = int(d_model)
        num_blocks_int = int(num_blocks)
        num_heads_int = int(num_heads)
        dropout_f = float(dropout)
        adj_value = adj
        adj_path_s = str(adj_path)
        adj_top_k_int = int(adj_top_k)
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
            from ..torch_graph_attention import torch_graph_attention_forecast

            return torch_graph_attention_forecast(
                train,
                horizon,
                variant=variant_s,
                lags=lags_int,
                d_model=d_model_int,
                num_blocks=num_blocks_int,
                num_heads=num_heads_int,
                dropout=dropout_f,
                adj=adj_value,
                adj_path=adj_path_s,
                adj_top_k=adj_top_k_int,
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
            interface="multivariate",
        )

    return extra


def _make_wave1_graph_structure_specs(context: Any) -> dict[str, ModelSpec]:
    """Lane 04 ownership: AGCRN / MTGNN style graph-structure lite families."""
    model_spec = context.ModelSpec
    _TORCH_COMMON_DEFAULTS = context._TORCH_COMMON_DEFAULTS
    _TORCH_COMMON_PARAM_HELP = context._TORCH_COMMON_PARAM_HELP
    np = context.np

    extra: dict[str, ModelSpec] = {}

    help_map = {
        "variant": "Graph-structure lite family variant: agcrn or mtgnn",
        "lags": LAG_WINDOW_PARAM_HELP,
        "d_model": "Latent width for the adaptive graph backbone",
        "num_blocks": "Number of temporal graph blocks",
        "kernel_size": "Temporal convolution kernel width",
        "dilation_base": "Temporal dilation growth factor",
        "dropout": DROPOUT_RANGE_PARAM_HELP,
        "adj": "Adjacency seed: identity, ring, fully-connected, corr, or matrix",
        "adj_path": "Optional adjacency file path (.npy or .csv/.txt)",
        "adj_top_k": "Top-k neighbors kept for correlation adjacency",
        "adaptive_adj": "Whether to learn an adaptive adjacency on top of the seed graph",
        "adj_emb_dim": "Embedding rank for adaptive adjacency factors",
        **_TORCH_COMMON_PARAM_HELP,
    }

    common_defaults = {
        "lags": 24,
        "d_model": 64,
        "kernel_size": 2,
        "dropout": 0.1,
        "adj_path": "",
        "adj_top_k": 8,
        "adaptive_adj": True,
        "adj_emb_dim": 8,
        **_TORCH_COMMON_DEFAULTS,
    }
    default_params = {
        "torch-agcrn-multivariate": {
            **common_defaults,
            "variant": "agcrn",
            "num_blocks": 2,
            "dilation_base": 1,
            "adj": "identity",
        },
        "torch-mtgnn-multivariate": {
            **common_defaults,
            "variant": "mtgnn",
            "num_blocks": 4,
            "dilation_base": 2,
            "adj": "corr",
        },
    }
    descriptions = {
        "torch-agcrn-multivariate": (
            "AGCRN-style lite multivariate forecaster. Reuses the existing adaptive "
            "Graph WaveNet-style backbone with an identity-seeded learned adjacency; this is "
            "a graph-structure proxy, not a paper-complete AGCRN reproduction. Requires PyTorch."
        ),
        "torch-mtgnn-multivariate": (
            "MTGNN-style lite multivariate forecaster. Reuses the existing adaptive "
            "Graph WaveNet-style backbone with a correlation-seeded learned adjacency and "
            "dilated temporal mixing; this is a graph-structure proxy, not a paper-complete "
            "MTGNN reproduction. Requires PyTorch."
        ),
    }

    def _factory(
        *,
        variant: str,
        lags: int = 24,
        d_model: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 2,
        dilation_base: int = 2,
        dropout: float = 0.1,
        adj: Any = "corr",
        adj_path: str = "",
        adj_top_k: int = 8,
        adaptive_adj: bool = True,
        adj_emb_dim: int = 8,
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
        **_params: Any,
    ) -> MultivariateForecasterFn:
        variant_s = str(variant)
        lags_int = int(lags)
        d_model_int = int(d_model)
        num_blocks_int = int(num_blocks)
        kernel_size_int = int(kernel_size)
        dilation_base_int = int(dilation_base)
        dropout_f = float(dropout)
        adj_value = adj
        adj_path_s = str(adj_path)
        adj_top_k_int = int(adj_top_k)
        adaptive_adj_bool = bool(adaptive_adj)
        adj_emb_dim_int = int(adj_emb_dim)
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
            from ..torch_graph_structure import torch_graph_structure_forecast

            return torch_graph_structure_forecast(
                train,
                horizon,
                variant=variant_s,
                lags=lags_int,
                d_model=d_model_int,
                num_blocks=num_blocks_int,
                kernel_size=kernel_size_int,
                dilation_base=dilation_base_int,
                dropout=dropout_f,
                adj=adj_value,
                adj_path=adj_path_s,
                adj_top_k=adj_top_k_int,
                adaptive_adj=adaptive_adj_bool,
                adj_emb_dim=adj_emb_dim_int,
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
            default_params=dict(default_params[key]),
            param_help=dict(help_map),
            requires=("torch",),
            interface="multivariate",
        )

    return extra


def _make_wave1_graph_spectral_specs(context: Any) -> dict[str, ModelSpec]:
    """Lane 05 ownership: StemGNN / FourierGNN style graph spectral lite families."""
    model_spec = context.ModelSpec
    _TORCH_COMMON_DEFAULTS = context._TORCH_COMMON_DEFAULTS
    _TORCH_COMMON_PARAM_HELP = context._TORCH_COMMON_PARAM_HELP
    np = context.np

    extra: dict[str, ModelSpec] = {}

    help_map = {
        "variant": "Graph-spectral lite family variant: stemgnn or fouriergnn",
        "lags": LAG_WINDOW_PARAM_HELP,
        "d_model": "Latent width for the spectral backbone",
        "num_blocks": "Number of spectral mixer blocks",
        "top_k_freq": "Number of FFT bins kept per node",
        "dropout": DROPOUT_RANGE_PARAM_HELP,
        **_TORCH_COMMON_PARAM_HELP,
    }

    base_defaults = {
        "lags": 24,
        "d_model": 64,
        "num_blocks": 2,
        "top_k_freq": 8,
        "dropout": 0.1,
        **_TORCH_COMMON_DEFAULTS,
    }
    descriptions = {
        "torch-stemgnn-multivariate": (
            "StemGNN-style lite multivariate forecaster. Fuses lag-history and FFT-derived "
            "spectral node features through compact node mixers; this is a spectral proxy, not "
            "a paper-complete StemGNN reproduction. Requires PyTorch."
        ),
        "torch-fouriergnn-multivariate": (
            "FourierGNN-style lite multivariate forecaster. Uses FFT-derived node tokens with "
            "learned horizon decoding; this is a spectral proxy, not a paper-complete FourierGNN "
            "reproduction. Requires PyTorch."
        ),
    }
    variants = {
        "torch-stemgnn-multivariate": "stemgnn",
        "torch-fouriergnn-multivariate": "fouriergnn",
    }

    def _factory(
        *,
        variant: str,
        lags: int = 24,
        d_model: int = 64,
        num_blocks: int = 2,
        top_k_freq: int = 8,
        dropout: float = 0.1,
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
        **_params: Any,
    ) -> MultivariateForecasterFn:
        variant_s = str(variant)
        lags_int = int(lags)
        d_model_int = int(d_model)
        num_blocks_int = int(num_blocks)
        top_k_freq_int = int(top_k_freq)
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
            from ..torch_graph_spectral import torch_graph_spectral_forecast

            return torch_graph_spectral_forecast(
                train,
                horizon,
                variant=variant_s,
                lags=lags_int,
                d_model=d_model_int,
                num_blocks=num_blocks_int,
                top_k_freq=top_k_freq_int,
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
            interface="multivariate",
        )

    return extra

_build_multivariate_catalog_base = build_multivariate_catalog


def build_multivariate_catalog(context: Any) -> dict[str, Any]:
    catalog = _build_multivariate_catalog_base(context)
    catalog.update(_make_wave1_graph_attention_specs(context))
    catalog.update(_make_wave1_graph_structure_specs(context))
    catalog.update(_make_wave1_graph_spectral_specs(context))
    return catalog
