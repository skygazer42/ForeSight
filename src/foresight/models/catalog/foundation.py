from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..specs import ForecasterFn, ModelSpec


def build_foundation_catalog(context: Any) -> dict[str, Any]:
    model_spec = context.ModelSpec
    _factory_hf_timeseries_transformer_direct = context._factory_hf_timeseries_transformer_direct
    return {
    "hf-timeseries-transformer-direct": model_spec(
        key="hf-timeseries-transformer-direct",
        description="Hugging Face TimeSeriesTransformer wrapper (lite) (direct local). Requires transformers + PyTorch.",
        factory=_factory_hf_timeseries_transformer_direct,
        default_params={
            "context_length": 48,
            "lags_sequence": (1, 2, 3, 4, 5, 6, 7),
            "d_model": 64,
            "nhead": 2,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "ffn_dim": 128,
            "dropout": 0.1,
            "num_time_features": 0,
            "num_samples": 100,
            "pretrained_model": "",
            "local_files_only": True,
            "normalize": True,
            "device": "cpu",
            "seed": 0,
            "epochs": 0,
        },
        param_help={
            "context_length": "Context length used by the model",
            "lags_sequence": "Comma-separated lags sequence (default: 1,2,3,4,5,6,7)",
            "d_model": "Transformer model dimension (ignored for pretrained_model)",
            "nhead": "Attention heads (ignored for pretrained_model)",
            "encoder_layers": "Encoder layers (ignored for pretrained_model)",
            "decoder_layers": "Decoder layers (ignored for pretrained_model)",
            "ffn_dim": "FFN hidden width (ignored for pretrained_model)",
            "dropout": "Dropout probability in [0,1) (ignored for pretrained_model)",
            "num_time_features": "Number of provided time features (0 disables time features)",
            "num_samples": "Number of parallel samples to generate; output is sample-mean",
            "pretrained_model": "Optional Hugging Face model name/path to load via from_pretrained()",
            "local_files_only": "If true, disallow downloads in from_pretrained() (true/false)",
            "normalize": "Z-score normalize the series before inference (true/false)",
            "device": "Torch device (cpu or cuda)",
            "seed": "Random seed (controls sampling)",
            "epochs": "Ignored (reserved for future fine-tuning support)",
        },
        requires=("transformers", "torch"),
        interface="local",
    ),
    }


def _make_wave1_foundation_wrapper_a_specs(context: Any) -> dict[str, ModelSpec]:
    """Lane 07 ownership: Lag-Llama / Chronos / TimesFM wrapper families."""
    model_spec = context.ModelSpec
    np = context.np

    extra: dict[str, ModelSpec] = {}

    help_map = {
        "backend": "Wrapper backend: auto, fixture-json, or hf-timeseries-transformer",
        "checkpoint_path": "Local fixture/checkpoint path used by fixture-json or as a backend source",
        "model_source": "Backend-specific model identifier or local model path",
        "local_files_only": "If true, disallow backend downloads where supported (true/false)",
        "context_length": "Context length passed to backend-backed inference when supported",
        "d_model": "Backend adapter hidden size when constructing a small fallback transformer",
        "num_samples": "Backend sample count where supported",
        "normalize": "Normalize the series before inference when supported (true/false)",
        "device": "Execution device for backend-backed inference",
        "seed": "Random seed for backend-backed inference",
    }

    base_defaults = {
        "backend": "auto",
        "checkpoint_path": "",
        "model_source": "",
        "local_files_only": True,
        "context_length": 48,
        "d_model": 16,
        "num_samples": 20,
        "normalize": True,
        "device": "cpu",
        "seed": 0,
    }

    families = {
        "lag-llama": "Lag-Llama wrapper scaffold (lite, inference-only). Requires explicit checkpoint_path or backend adapter; no native training.",
        "chronos": "Chronos wrapper scaffold (lite, inference-only). Requires explicit checkpoint_path or backend adapter; no native training.",
        "chronos-bolt": "Chronos-Bolt wrapper scaffold (lite, inference-only). Requires explicit checkpoint_path or backend adapter; no native training.",
        "timesfm": "TimesFM wrapper scaffold (lite, inference-only). Requires explicit checkpoint_path or backend adapter; no native training.",
    }

    def _factory(
        *,
        family: str,
        backend: str = "auto",
        checkpoint_path: str = "",
        model_source: str = "",
        local_files_only: bool = True,
        context_length: int = 48,
        d_model: int = 16,
        num_samples: int = 20,
        normalize: bool = True,
        device: str = "cpu",
        seed: int = 0,
        **params: Any,
    ) -> ForecasterFn:
        family_s = str(family).strip()
        backend_s = str(backend)
        checkpoint_path_s = str(checkpoint_path)
        model_source_s = str(model_source)
        local_files_only_bool = bool(local_files_only)
        context_length_int = int(context_length)
        d_model_int = int(d_model)
        num_samples_int = int(num_samples)
        normalize_bool = bool(normalize)
        device_s = str(device)
        seed_int = int(seed)
        extra_params = dict(params)

        def _f(train: Any, horizon: int) -> np.ndarray:
            from ..foundation_wrappers_a import foundation_wrapper_a_forecast

            return foundation_wrapper_a_forecast(
                train,
                horizon,
                family=family_s,
                backend=backend_s,
                checkpoint_path=checkpoint_path_s,
                model_source=model_source_s,
                local_files_only=local_files_only_bool,
                context_length=context_length_int,
                d_model=d_model_int,
                num_samples=num_samples_int,
                normalize=normalize_bool,
                device=device_s,
                seed=seed_int,
                **extra_params,
            )

        return _f

    for key, description in families.items():
        extra[key] = model_spec(
            key=key,
            description=description,
            factory=_factory,
            default_params={**base_defaults, "family": key},
            param_help=dict(help_map),
            requires=(),
            interface="local",
        )

    return extra


def _make_wave1_foundation_wrapper_b_specs(context: Any) -> dict[str, ModelSpec]:
    """Lane 08 ownership: Moirai / MOMENT / Time-MoE / Timer-S1 wrapper families."""
    model_spec = context.ModelSpec
    np = context.np

    extra: dict[str, ModelSpec] = {}

    help_map = {
        "backend": "Wrapper backend: auto, fixture-json, or hf-timeseries-transformer",
        "checkpoint_path": "Local fixture/checkpoint path used by fixture-json or as a backend source",
        "model_source": "Backend-specific model identifier or local model path",
        "local_files_only": "If true, disallow backend downloads where supported (true/false)",
        "context_length": "Context length passed to backend-backed inference when supported",
        "d_model": "Backend adapter hidden size when constructing a small fallback transformer",
        "num_samples": "Backend sample count where supported",
        "normalize": "Normalize the series before inference when supported (true/false)",
        "device": "Execution device for backend-backed inference",
        "seed": "Random seed for backend-backed inference",
    }

    base_defaults = {
        "backend": "auto",
        "checkpoint_path": "",
        "model_source": "",
        "local_files_only": True,
        "context_length": 48,
        "d_model": 16,
        "num_samples": 20,
        "normalize": True,
        "device": "cpu",
        "seed": 0,
    }

    families = {
        "moirai": "Moirai wrapper scaffold (lite, inference-only). Requires explicit checkpoint_path or backend adapter; no native training.",
        "moment": "MOMENT wrapper scaffold (lite, inference-only). Requires explicit checkpoint_path or backend adapter; no native training.",
        "time-moe": "Time-MoE wrapper scaffold (lite, inference-only). Requires explicit checkpoint_path or backend adapter; no native training.",
        "timer-s1": "Timer-S1 wrapper scaffold (lite, inference-only). Requires explicit checkpoint_path or backend adapter; no native training.",
    }

    def _factory(
        *,
        family: str,
        backend: str = "auto",
        checkpoint_path: str = "",
        model_source: str = "",
        local_files_only: bool = True,
        context_length: int = 48,
        d_model: int = 16,
        num_samples: int = 20,
        normalize: bool = True,
        device: str = "cpu",
        seed: int = 0,
        **params: Any,
    ) -> ForecasterFn:
        family_s = str(family).strip()
        backend_s = str(backend)
        checkpoint_path_s = str(checkpoint_path)
        model_source_s = str(model_source)
        local_files_only_bool = bool(local_files_only)
        context_length_int = int(context_length)
        d_model_int = int(d_model)
        num_samples_int = int(num_samples)
        normalize_bool = bool(normalize)
        device_s = str(device)
        seed_int = int(seed)
        extra_params = dict(params)

        def _f(train: Any, horizon: int) -> np.ndarray:
            from ..foundation_wrappers_b import foundation_wrapper_b_forecast

            return foundation_wrapper_b_forecast(
                train,
                horizon,
                family=family_s,
                backend=backend_s,
                checkpoint_path=checkpoint_path_s,
                model_source=model_source_s,
                local_files_only=local_files_only_bool,
                context_length=context_length_int,
                d_model=d_model_int,
                num_samples=num_samples_int,
                normalize=normalize_bool,
                device=device_s,
                seed=seed_int,
                **extra_params,
            )

        return _f

    for key, description in families.items():
        extra[key] = model_spec(
            key=key,
            description=description,
            factory=_factory,
            default_params={**base_defaults, "family": key},
            param_help=dict(help_map),
            requires=(),
            interface="local",
        )

    return extra

_build_foundation_catalog_base = build_foundation_catalog


def build_foundation_catalog(context: Any) -> dict[str, Any]:
    catalog = _build_foundation_catalog_base(context)
    catalog.update(_make_wave1_foundation_wrapper_a_specs(context))
    catalog.update(_make_wave1_foundation_wrapper_b_specs(context))
    return catalog
