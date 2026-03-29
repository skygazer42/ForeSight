from __future__ import annotations

from typing import Any

import numpy as np

from ..optional_deps import missing_dependency_message
from .torch_nn import _normalize_series, _require_torch


def _require_transformers() -> Any:
    try:
        import transformers  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            missing_dependency_message("transformers", subject="hf-timeseries-transformer-direct")
        ) from e
    return transformers


def _as_1d_float_array(train: Any) -> np.ndarray:
    arr = np.asarray(train, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {arr.shape}")
    return arr


def _parse_lags_sequence(value: Any) -> list[int]:
    if value is None:
        return [1, 2, 3, 4, 5, 6, 7]
    if isinstance(value, list | tuple):
        out = [int(v) for v in value]
    else:
        s = str(value).strip()
        if not s:
            return [1, 2, 3, 4, 5, 6, 7]
        out = [int(p.strip()) for p in s.split(",") if p.strip()]
    if not out:
        raise ValueError("lags_sequence must be non-empty")
    if any(v <= 0 for v in out):
        raise ValueError("lags_sequence values must be >= 1")
    return out


def hf_timeseries_transformer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    context_length: int = 48,
    lags_sequence: Any = (1, 2, 3, 4, 5, 6, 7),
    d_model: int = 64,
    nhead: int = 2,
    encoder_layers: int = 2,
    decoder_layers: int = 2,
    ffn_dim: int = 128,
    dropout: float = 0.1,
    num_time_features: int = 0,
    num_samples: int = 100,
    pretrained_model: str = "",
    local_files_only: bool = True,
    normalize: bool = True,
    device: str = "cpu",
    seed: int = 0,
    epochs: int = 0,
    **_params: Any,
) -> np.ndarray:
    """
    Hugging Face TimeSeriesTransformer wrapper (lite) for direct local forecasting.

    - If `pretrained_model` is provided, loads `TimeSeriesTransformerForPrediction.from_pretrained(...)`.
    - Otherwise, initializes a small model from config and runs inference.

    The output is the mean across generated samples: shape `(horizon,)`.
    """
    epochs_n = int(epochs)
    if epochs_n < 0:
        raise ValueError("epochs must be >= 0")

    torch = _require_torch()
    _require_transformers()
    from transformers import (  # type: ignore
        TimeSeriesTransformerConfig,
        TimeSeriesTransformerForPrediction,
    )

    x = _as_1d_float_array(train)
    h = int(horizon)
    ctx = int(context_length)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if ctx <= 0:
        raise ValueError("context_length must be >= 1")

    lags = _parse_lags_sequence(lags_sequence)
    max_lag = int(max(lags))
    past_length = ctx + max_lag
    if int(x.size) < past_length:
        raise ValueError(
            f"Need >= context_length+max(lags_sequence) observations "
            f"(context_length={ctx}, max_lag={max_lag}), got {int(x.size)}"
        )

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    torch.manual_seed(int(seed))
    dev = torch.device(str(device))

    pretrained = str(pretrained_model).strip()
    if pretrained and pretrained.lower() not in {"none", "null"}:
        model = TimeSeriesTransformerForPrediction.from_pretrained(
            pretrained,
            local_files_only=bool(local_files_only),
        )
        model.config.context_length = ctx
        model.config.prediction_length = h
        model.config.lags_sequence = list(lags)
        model.config.num_parallel_samples = int(num_samples)
        model.config.num_time_features = int(num_time_features)
    else:
        cfg = TimeSeriesTransformerConfig(
            prediction_length=h,
            context_length=ctx,
            lags_sequence=list(lags),
            d_model=int(d_model),
            encoder_layers=int(encoder_layers),
            decoder_layers=int(decoder_layers),
            encoder_attention_heads=int(nhead),
            decoder_attention_heads=int(nhead),
            encoder_ffn_dim=int(ffn_dim),
            decoder_ffn_dim=int(ffn_dim),
            dropout=float(dropout),
            attention_dropout=float(dropout),
            activation_dropout=float(dropout),
            num_time_features=int(num_time_features),
            num_parallel_samples=int(num_samples),
            input_size=1,
        )
        model = TimeSeriesTransformerForPrediction(cfg)

    model = model.to(dev)
    model.eval()

    window = x_work[-past_length:].astype(float, copy=False)
    past_values = torch.tensor(window.reshape(1, past_length), dtype=torch.float32, device=dev)

    nt = int(num_time_features)
    if nt > 0:
        age_past = np.linspace(0.0, 1.0, past_length, dtype=float).reshape(1, past_length, 1)
        age_future = np.linspace(1.0, 1.0 + (h / max(1, past_length)), h, dtype=float).reshape(1, h, 1)
        past_tf = np.repeat(age_past, nt, axis=2)
        fut_tf = np.repeat(age_future, nt, axis=2)
        past_time_features = torch.tensor(past_tf, dtype=torch.float32, device=dev)
        future_time_features = torch.tensor(fut_tf, dtype=torch.float32, device=dev)
    else:
        past_time_features = torch.zeros((1, past_length, 0), dtype=torch.float32, device=dev)
        future_time_features = torch.zeros((1, h, 0), dtype=torch.float32, device=dev)

    past_observed_mask = torch.ones((1, past_length), dtype=torch.bool, device=dev)
    out = model.generate(
        past_values=past_values,
        past_time_features=past_time_features,
        future_time_features=future_time_features,
        past_observed_mask=past_observed_mask,
    )

    seq = out.sequences  # (B, samples, h)
    yhat_t = seq.mean(dim=1).detach().cpu().numpy().reshape(-1)
    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)
