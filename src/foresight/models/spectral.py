from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Series contains NaN/inf")
    return x


def _validated_fft_topk_input(
    train: Any,
    *,
    horizon: int,
    top_k: int,
) -> tuple[np.ndarray, int, int]:
    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    k = int(top_k)
    if k <= 0:
        raise ValueError("top_k must be >= 1")
    if x.size < 4:
        raise ValueError("fft_topk_forecast requires at least 4 training points")
    return x, h, k


def _fit_linear_trend(x: np.ndarray) -> tuple[float, float]:
    """
    Fit y ~ a + b*t with t = 0..n-1. Returns (a, b).
    """
    n = int(x.size)
    t = np.arange(n, dtype=float)
    X = np.stack([np.ones((n,), dtype=float), t], axis=1)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)
    return float(coef[0]), float(coef[1])


def _harmonic_regression_design_matrix(t: np.ndarray, ws: np.ndarray) -> np.ndarray:
    tt = np.asarray(t, dtype=float)
    cols = [np.ones((int(tt.size),), dtype=float)]
    for w in np.asarray(ws, dtype=float):
        cols.append(np.sin(float(w) * tt))
        cols.append(np.cos(float(w) * tt))
    return np.stack(cols, axis=1)


def fft_topk_forecast(
    train: Any,
    horizon: int,
    *,
    top_k: int = 3,
    include_trend: bool = True,
) -> np.ndarray:
    """
    FFT-based extrapolation: keep top-K frequencies and extrapolate sinusoid fit.

    Roughly inspired by common "FFT forecaster" baselines used in TS libraries.
    """
    x, h, k = _validated_fft_topk_input(train, horizon=horizon, top_k=top_k)

    n = int(x.size)
    t = np.arange(n, dtype=float)

    a0 = 0.0
    b0 = 0.0
    resid = x
    if bool(include_trend):
        a0, b0 = _fit_linear_trend(x)
        resid = x - (a0 + b0 * t)

    # Pick dominant frequencies via FFT magnitude.
    fft = np.fft.rfft(resid)
    mag = np.abs(fft)
    if mag.size == 0:
        raise ValueError("Unexpected empty FFT result")
    mag[0] = 0.0  # ignore DC component

    k_eff = min(k, int(np.count_nonzero(mag > 0.0)))
    if k_eff <= 0:
        # Residual is (near) constant; fall back to trend-only.
        tf = np.arange(n, n + h, dtype=float)
        return np.asarray(a0 + b0 * tf, dtype=float)

    idx = np.argpartition(mag, -k_eff)[-k_eff:]
    idx = idx[np.argsort(mag[idx])[::-1]]

    freqs = np.fft.rfftfreq(n, d=1.0)[idx]
    ws = 2.0 * np.pi * freqs

    # Fit sin/cos regression on residuals using the selected frequencies.
    X = _harmonic_regression_design_matrix(t, ws)
    coef, *_ = np.linalg.lstsq(X, resid, rcond=None)

    tf = np.arange(n, n + h, dtype=float)
    x_future = _harmonic_regression_design_matrix(tf, ws)
    resid_fc = x_future @ coef

    trend_fc = a0 + b0 * tf
    return np.asarray(trend_fc + resid_fc, dtype=float)
