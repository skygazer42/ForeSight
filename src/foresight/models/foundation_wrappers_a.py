from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .foundation import normalize_foundation_source
from .hf_time_series import hf_timeseries_transformer_direct_forecast


def _as_1d_float_array(train: Any) -> np.ndarray:
    arr = np.asarray(train, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError("Foundation wrapper scaffolds require at least 2 observations")
    return arr


def _load_fixture_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    path = Path(str(checkpoint_path).strip())
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"checkpoint_path {str(path)!r} does not exist; pass a local JSON fixture/checkpoint."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixture-json checkpoint must contain a JSON object")
    return payload


def _forecast_from_fixture(x: np.ndarray, horizon: int, payload: dict[str, Any]) -> np.ndarray:
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")

    last = float(x[-1])
    prev = float(x[-2])
    trend = last - prev
    bias = float(payload.get("bias", 0.0))
    scale = float(payload.get("scale", 1.0))
    use_trend = bool(payload.get("use_trend", False))
    base = last * scale + bias

    out = np.empty(h, dtype=float)
    for idx in range(h):
        step = float(idx + 1)
        out[idx] = base + (trend * step if use_trend else 0.0)
    return out


def foundation_wrapper_a_forecast(
    train: Any,
    horizon: int,
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
) -> np.ndarray:
    x = _as_1d_float_array(train)
    family_s = str(family).strip() or "foundation-wrapper-a"
    backend_s = str(backend).strip().lower() or "auto"
    checkpoint_s = str(checkpoint_path).strip()
    source_s = normalize_foundation_source(model_source)

    if backend_s == "auto":
        if checkpoint_s:
            backend_s = "fixture-json"
        elif source_s:
            backend_s = "hf-timeseries-transformer"
        else:
            raise ValueError(
                f"{family_s} wrapper requires checkpoint_path for fixture-json mode or "
                "model_source for backend-backed inference."
            )

    if backend_s == "fixture-json":
        payload = _load_fixture_checkpoint(checkpoint_s)
        return _forecast_from_fixture(x, int(horizon), payload)

    if backend_s in {"hf-timeseries-transformer", "hf-tst"}:
        source_final = source_s or checkpoint_s
        if not source_final:
            raise ValueError(
                f"{family_s} wrapper with backend={backend_s!r} requires model_source or checkpoint_path."
            )
        return hf_timeseries_transformer_direct_forecast(
            x,
            int(horizon),
            context_length=int(context_length),
            d_model=int(d_model),
            num_samples=int(num_samples),
            pretrained_model=source_final,
            local_files_only=bool(local_files_only),
            normalize=bool(normalize),
            device=str(device),
            seed=int(seed),
            **params,
        )

    raise ValueError(
        f"Unsupported backend {backend_s!r} for {family_s}. "
        "Use auto, fixture-json, or hf-timeseries-transformer."
    )
